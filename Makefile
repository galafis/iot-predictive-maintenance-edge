.PHONY: install test lint run clean docker-build docker-up docker-down help

PYTHON ?= python
PIP ?= pip

help: ## Show this help message
	@echo "IoT Predictive Maintenance Edge - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install dev dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov flake8 black isort mypy

test: ## Run tests with coverage
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

test-quick: ## Run tests without coverage
	$(PYTHON) -m pytest tests/ -v --tb=short

lint: ## Run linters
	$(PYTHON) -m flake8 src/ tests/ main.py --max-line-length=120
	$(PYTHON) -m black --check src/ tests/ main.py
	$(PYTHON) -m isort --check-only src/ tests/ main.py

format: ## Auto-format code
	$(PYTHON) -m black src/ tests/ main.py
	$(PYTHON) -m isort src/ tests/ main.py

run: ## Run the demo
	$(PYTHON) main.py

clean: ## Clean build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage dist build *.egg-info

docker-build: ## Build Docker image
	docker build -f docker/Dockerfile -t iot-predictive-maintenance-edge .

docker-up: ## Start with Docker Compose
	docker-compose -f docker/docker-compose.yml up -d

docker-down: ## Stop Docker Compose
	docker-compose -f docker/docker-compose.yml down

typecheck: ## Run type checking
	$(PYTHON) -m mypy src/ --ignore-missing-imports
