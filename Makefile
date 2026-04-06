.PHONY: help install install-dev format lint test smoke demo docs-check

help:
	@echo "Available targets:"
	@echo "  install      Install project in editable mode"
	@echo "  install-dev  Install project with dev extras"
	@echo "  format       Run ruff format"
	@echo "  lint         Run ruff checks"
	@echo "  test         Run pytest"
	@echo "  smoke        Run predeploy smoke test"
	@echo "  demo         Run demo launcher"

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e .[dev]

format:
	ruff format .

lint:
	ruff check .

test:
	pytest -q

smoke:
	python test_predeploy.py

demo:
	python run_demo.py
