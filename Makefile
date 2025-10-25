.PHONY: help install test test-unit test-local build run push deploy clean lint format

# Variables
DOCKER_IMAGE ?= llama4-scout-runpod
DOCKER_TAG ?= latest
DOCKER_REGISTRY ?= docker.io
DOCKER_USERNAME ?= your-username

help:
	@echo "Available commands:"
	@echo "  make install        - Install Python dependencies"
	@echo "  make test          - Run all tests"
	@echo "  make test-unit     - Run unit tests only"
	@echo "  make test-local    - Run local integration test"
	@echo "  make build         - Build Docker image"
	@echo "  make run           - Run container locally with Docker Compose"
	@echo "  make push          - Push Docker image to registry"
	@echo "  make deploy        - Build and push Docker image"
	@echo "  make clean         - Clean up Docker resources"
	@echo "  make lint          - Run linting"
	@echo "  make format        - Format code with black"

install:
	pip install -r requirements.txt
	pip install pytest black flake8 pytest-cov

test: test-unit

test-unit:
	@echo "Running unit tests..."
	python -m pytest tests/ -v --cov=src --cov-report=term-missing

test-local:
	@echo "Running local integration test..."
	@if [ -z "$$HF_TOKEN" ]; then \
		echo "ERROR: HF_TOKEN not set. Please export HF_TOKEN=your_token"; \
		exit 1; \
	fi
	python scripts/test_local.py --mode simple

test-edge:
	@echo "Running edge case tests..."
	python scripts/test_local.py --mode edge

test-health:
	@echo "Testing health endpoint..."
	python scripts/test_local.py --mode health

benchmark:
	@echo "Running benchmark..."
	python scripts/test_local.py --mode benchmark --num-requests 10

build:
	@echo "Building Docker image..."
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

build-no-cache:
	@echo "Building Docker image (no cache)..."
	docker build --no-cache -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

run:
	@echo "Starting container with Docker Compose..."
	@if [ -z "$$HF_TOKEN" ]; then \
		echo "ERROR: HF_TOKEN not set. Please export HF_TOKEN=your_token"; \
		exit 1; \
	fi
	docker-compose up

run-detached:
	@echo "Starting container in background..."
	docker-compose up -d

stop:
	@echo "Stopping container..."
	docker-compose down

logs:
	@echo "Showing container logs..."
	docker-compose logs -f

tag:
	@echo "Tagging image for registry..."
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) \
		$(DOCKER_REGISTRY)/$(DOCKER_USERNAME)/$(DOCKER_IMAGE):$(DOCKER_TAG)

push: tag
	@echo "Pushing image to registry..."
	docker push $(DOCKER_REGISTRY)/$(DOCKER_USERNAME)/$(DOCKER_IMAGE):$(DOCKER_TAG)

deploy: build push
	@echo "Docker image built and pushed successfully!"
	@echo "Image: $(DOCKER_REGISTRY)/$(DOCKER_USERNAME)/$(DOCKER_IMAGE):$(DOCKER_TAG)"

clean:
	@echo "Cleaning up Docker resources..."
	docker-compose down -v
	docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG) 2>/dev/null || true

clean-cache:
	@echo "Cleaning Python cache..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

lint:
	@echo "Running linting..."
	flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503

format:
	@echo "Formatting code..."
	black src/ tests/ scripts/ --line-length=100

check-env:
	@echo "Checking environment..."
	@if [ -z "$$HF_TOKEN" ]; then \
		echo "WARNING: HF_TOKEN not set"; \
	else \
		echo "✓ HF_TOKEN is set"; \
	fi
	@echo "Python version: $$(python --version)"
	@echo "Docker version: $$(docker --version)"

shell:
	@echo "Opening shell in container..."
	docker-compose run --rm llama4-scout /bin/bash

gpu-check:
	@echo "Checking GPU availability..."
	docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Development shortcuts
dev-install: install
	pip install ipython jupyter

dev-test: test-unit test-local

dev-clean: clean clean-cache

