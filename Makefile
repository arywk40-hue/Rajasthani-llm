.PHONY: all setup test lint format clean

setup:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	black --check src tests
	isort --check-only src tests
	flake8 src tests

format:
	black src tests
	isort src tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
