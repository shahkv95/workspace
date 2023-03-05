.PHONY: help setup test train predict clean

# Display available commands
help:
	@echo "Available commands:"
	@echo "  setup    - set up the project environment"
	@echo "  test     - run unit tests"
	@echo "  train    - train models"
	@echo "  predict  - make predictions using trained models"
	@echo "  clean    - clean up build artifacts"

# Set up the project environment using Pipenv
setup:
	pipenv install --dev

# Run unit tests
test:
	pipenv run pytest

# Train models
train:
	pipenv run python3 src/main.py train

# Make predictions using trained models
predict:
	pipenv run python3 src/main.py predict

# Clean up build artifacts
clean:
	rm -rf build/
	rm -rf workspace.egg-info/
	find . -type f -name "*.pyc" -delete

