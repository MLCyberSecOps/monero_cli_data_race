.PHONY: install format lint test type-check clean build publish

# Variables
PYTHON = python3
PIP = pip3
PACKAGE = threadguard_enhanced.py

# Install development dependencies
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

# Install pre-commit hooks
install-hooks:
	pre-commit install

# Format code
format:
	black $(PACKAGE) tests
	isort $(PACKAGE) tests

# Lint code
lint:
	flake8 $(PACKAGE) tests
	pylint $(PACKAGE) tests

# Run tests
test:
	pytest -v --cov=$(PACKAGE) --cov-report=term-missing --cov-report=xml

# Run type checking
type-check:
	mypy $(PACKAGE) tests

# Clean up build artifacts
clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ htmlcov/ .coverage coverage.xml
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

# Build package
build: clean
	$(PYTHON) -m build

# Run all checks
check: format lint test type-check

# Publish to PyPI (requires twine)
publish: build
	twine upload dist/*

# Run all checks and build package
all: check build

# Show help
help:
	@echo "Available targets:"
	@echo "  install      - Install development dependencies"
	@echo "  install-hooks - Install pre-commit hooks"
	@echo "  format       - Format code with black and isort"
	@echo "  lint         - Run linters (flake8 and pylint)"
	@echo "  test         - Run tests with coverage"
	@echo "  type-check   - Run static type checking with mypy"
	@echo "  clean        - Remove build and cache files"
	@echo "  build        - Build package"
	@echo "  check        - Run all checks (format, lint, test, type-check)"
	@echo "  publish      - Build and publish package to PyPI"
	@echo "  all          - Run all checks and build package"
	@echo "  help         - Show this help message"
