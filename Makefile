.PHONY: help setup test train predict clean

# Display available commands
help:
	@echo "Available commands:"
	@echo "  setup    - set up the project environment"
	@echo "  test     - run unit tests"
	@echo "  run      - run application, train models, predict on test data"
	@echo "  clean    - clean up build artifacts"

# Set up the project environment using Pipenv
setup:
	pipenv install --dev

# Run unit tests
test:
	pipenv run pytest

# Train models
run:
	pipenv run python3 src/main.py 

# Clean up build artifacts
clean:
	rm -rf build/
	rm -rf workspace.egg-info/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

