"""
Unit tests for the RunPod handler.
These tests can be run without loading the actual model.
"""

import unittest
from unittest.mock import patch

from src.handler import validate_input, InferenceInput
from src.utils import (
    ValidationError,
    sanitize_input,
    format_error_response,
    format_success_response,
    validate_generation_params,
)


class TestInputValidation(unittest.TestCase):
    """Test input validation logic."""

    def test_valid_input(self):
        """Test validation with valid input."""
        input_data = {
            "prompt": "Hello, how are you?",
            "max_new_tokens": 100,
            "temperature": 0.7,
        }

        result = validate_input(input_data)
        self.assertIsInstance(result, InferenceInput)
        self.assertEqual(result.prompt, "Hello, how are you?")
        self.assertEqual(result.max_new_tokens, 100)
        self.assertEqual(result.temperature, 0.7)

    def test_missing_prompt(self):
        """Test validation with missing prompt."""
        input_data = {
            "max_new_tokens": 100,
        }

        with self.assertRaises(ValidationError):
            validate_input(input_data)

    def test_empty_prompt(self):
        """Test validation with empty prompt."""
        input_data = {
            "prompt": "",
            "max_new_tokens": 100,
        }

        with self.assertRaises(ValidationError):
            validate_input(input_data)

    def test_invalid_temperature(self):
        """Test validation with invalid temperature."""
        input_data = {
            "prompt": "Test",
            "temperature": 3.0,  # Out of range
        }

        with self.assertRaises(ValidationError):
            validate_input(input_data)

    def test_invalid_top_p(self):
        """Test validation with invalid top_p."""
        input_data = {
            "prompt": "Test",
            "top_p": 1.5,  # Out of range
        }

        with self.assertRaises(ValidationError):
            validate_input(input_data)

    def test_default_values(self):
        """Test that default values are None when not provided."""
        input_data = {
            "prompt": "Test",
        }

        result = validate_input(input_data)
        self.assertIsNone(result.max_new_tokens)
        self.assertIsNone(result.temperature)
        self.assertIsNone(result.top_p)


class TestSanitizeInput(unittest.TestCase):
    """Test input sanitization."""

    def test_normal_input(self):
        """Test sanitization with normal input."""
        text = "Hello, world!"
        result = sanitize_input(text)
        self.assertEqual(result, "Hello, world!")

    def test_null_bytes_removed(self):
        """Test that null bytes are removed."""
        text = "Hello\x00World"
        result = sanitize_input(text)
        self.assertEqual(result, "HelloWorld")

    def test_empty_input_raises_error(self):
        """Test that empty input raises error."""
        with self.assertRaises(ValidationError):
            sanitize_input("")

    def test_non_string_raises_error(self):
        """Test that non-string input raises error."""
        with self.assertRaises(ValidationError):
            sanitize_input(123)

    def test_max_length_exceeded(self):
        """Test that exceeding max length raises error."""
        text = "a" * 10000
        with self.assertRaises(ValidationError):
            sanitize_input(text, max_length=1000)


class TestResponseFormatting(unittest.TestCase):
    """Test response formatting functions."""

    def test_error_response(self):
        """Test error response formatting."""
        error = ValueError("Test error")
        response = format_error_response(error, request_id="test-123")

        self.assertIn("error", response)
        self.assertEqual(response["error"]["type"], "ValueError")
        self.assertEqual(response["error"]["message"], "Test error")
        self.assertEqual(response["request_id"], "test-123")

    def test_success_response(self):
        """Test success response formatting."""
        generated_text = "This is generated text"
        metrics = {
            "tokens_generated": 10,
            "generation_time_ms": 1000,
        }
        response = format_success_response(generated_text, metrics, request_id="test-456")

        self.assertIn("output", response)
        self.assertEqual(response["output"]["generated_text"], generated_text)
        self.assertEqual(response["output"]["tokens_generated"], 10)
        self.assertEqual(response["request_id"], "test-456")


class TestGenerationParamsValidation(unittest.TestCase):
    """Test generation parameter validation."""

    def test_valid_params(self):
        """Test with valid parameters."""
        # Should not raise any exception
        validate_generation_params(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_new_tokens=100,
            repetition_penalty=1.1,
        )

    def test_invalid_temperature(self):
        """Test with invalid temperature."""
        with self.assertRaises(ValidationError):
            validate_generation_params(
                temperature=3.0,
                top_p=0.9,
                top_k=50,
                max_new_tokens=100,
                repetition_penalty=1.1,
            )

    def test_invalid_top_p(self):
        """Test with invalid top_p."""
        with self.assertRaises(ValidationError):
            validate_generation_params(
                temperature=0.7,
                top_p=1.5,
                top_k=50,
                max_new_tokens=100,
                repetition_penalty=1.1,
            )

    def test_negative_top_k(self):
        """Test with negative top_k."""
        with self.assertRaises(ValidationError):
            validate_generation_params(
                temperature=0.7,
                top_p=0.9,
                top_k=-1,
                max_new_tokens=100,
                repetition_penalty=1.1,
            )

    def test_zero_max_tokens(self):
        """Test with zero max_new_tokens."""
        with self.assertRaises(ValidationError):
            validate_generation_params(
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                max_new_tokens=0,
                repetition_penalty=1.1,
            )


class TestHandlerIntegration(unittest.TestCase):
    """Integration tests for the handler (with mocked model)."""

    @patch('src.handler.ModelLoader')
    @patch('src.handler.generate_text')
    def test_handler_success(self, mock_generate, mock_loader):
        """Test successful handler execution."""
        from src.handler import handler

        # Mock model loader
        mock_loader.is_loaded.return_value = True

        # Mock generate_text
        mock_generate.return_value = (
            "This is a test response",
            {
                "tokens_generated": 5,
                "generation_time_ms": 100,
                "tokens_per_second": 50.0,
            }
        )

        # Test event
        event = {
            "input": {
                "prompt": "Test prompt",
                "max_new_tokens": 50,
            }
        }

        result = handler(event)

        # Verify response structure
        self.assertIn("output", result)
        self.assertIn("generated_text", result["output"])
        self.assertEqual(result["output"]["generated_text"], "This is a test response")
        self.assertIn("request_id", result)

    @patch('src.handler.ModelLoader')
    def test_handler_missing_input(self, mock_loader):
        """Test handler with missing input field."""
        from src.handler import handler

        mock_loader.is_loaded.return_value = True

        event = {}  # No input field

        result = handler(event)

        # Should return error response
        self.assertIn("error", result)
        self.assertEqual(result["error"]["type"], "ValidationError")

    @patch('src.handler.ModelLoader')
    def test_handler_invalid_params(self, mock_loader):
        """Test handler with invalid parameters."""
        from src.handler import handler

        mock_loader.is_loaded.return_value = True

        event = {
            "input": {
                "prompt": "Test",
                "temperature": 5.0,  # Invalid
            }
        }

        result = handler(event)

        # Should return validation error
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()

