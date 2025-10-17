.PHONY: check format test test-slow test-all clean docs lint type-check help

check: lint type-check
	@echo "All checks passed!"

lint:
	poetry run ruff check EMFieldML
	poetry run isort --check-only EMFieldML
	poetry run black --check EMFieldML

format:
	poetry run ruff check --fix EMFieldML
	poetry run isort EMFieldML
	poetry run black EMFieldML

type-check:
	poetry run mypy EMFieldML

test:
	poetry run pytest tests/

test-slow:
	poetry run pytest tests/test_visualize.py -m slow

test-all:
	poetry run pytest tests/ -m ""

docs:
	cd docs && poetry run make html

clean:
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf docs/_build/

help:
	@echo "Available targets:"
	@echo "  test        - Run normal tests (excludes slow tests)"
	@echo "  test-slow   - Run only slow visualization tests"
	@echo "  test-all    - Run all tests including slow ones"
	@echo "  check       - Run linting and type checking"
	@echo "  format      - Format code with ruff, isort, and black"
	@echo "  lint        - Check code style with ruff, isort, and black"
	@echo "  type-check  - Run mypy type checking"
	@echo "  docs        - Build documentation"
	@echo "  clean       - Clean cache and build directories"
