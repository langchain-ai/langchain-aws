######################
# VARIABLES
######################

PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
PACKAGE_NAME=langgraph_checkpoint_aws

.PHONY: help lint format install_dev install_test install_lint install_typing install_codespell check_imports spell_check spell_fix

######################
# LINTING AND FORMATTING
######################

# Define different lint targets
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=$(PACKAGE_NAME)
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint: ## Run linter
	poetry run ruff check $(PYTHON_FILES)

lint_fix: ## Run linter and fix issues
	poetry run ruff check --fix $(PYTHON_FILES)

lint_diff: ## Run linter on changed files
	poetry run ruff check $(PYTHON_FILES)

lint_package: ## Run linter on package
	poetry run ruff check --select I $(PYTHON_FILES)

lint_tests: ## Run type checking on tests
	mkdir -p $(MYPY_CACHE); poetry run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format: ## Run code formatter
	poetry run ruff format $(PYTHON_FILES)

format_diff: ## Run code formatter and show differences
	poetry run ruff format $(PYTHON_FILES) --diff

spell_check: ## Run code spell check
	poetry run codespell --toml pyproject.toml

spell_fix: ## Run code spell fix
	poetry run codespell --toml pyproject.toml -w

######################
# TESTING
######################

# Define a variable for the test file path.
test test_watch tests: TEST_FILE ?= tests/unit_tests/
integration_test integration_tests: TEST_FILE = tests/integration_tests/

# Define a variable for Python and notebook files.
PYTHON_FILES=.

tests: ## Run all unit tests
	poetry run pytest $(TEST_FILE)

test:  ## Run individual unit test: make test TEST_FILE=tests/unit_test/test.py
	poetry run pytest $(TEST_FILE)

integration_tests: ## Run all integration tests
	poetry run pytest $(TEST_FILE)

integration_test: ## Run individual integration test: make integration_test TEST_FILE=tests/integration_tests/integ_test.py
	poetry run pytest $(TEST_FILE)

test_watch: ## Run and interactively watch unit tests
	poetry run ptw --snapshot-update --now . -- -vv $(TEST_FILE)

######################
# DEPENDENCIES
######################

install: ## Install package
	@pip install --no-cache -U poetry
	@poetry install

install_dev: ## Install development environment
	@pip install --no-cache -U poetry
	@poetry install --with dev

install_test: ## Install test dependencies
	@pip install --no-cache -U poetry
	@poetry install --with test

install_lint: ## Install lint dependencies
	@pip install --no-cache -U poetry
	@poetry install --with lint

install_typing: ## Install typing dependencies
	@pip install --no-cache -U poetry
	@poetry install --with typing

install_codespell: ## Install codespell dependencies
	@pip install --no-cache -U poetry
	@poetry install --with codespell

install_all: ## Install all dependencies including optional groups
	@pip install --no-cache -U poetry
	@poetry install --with dev,test,lint,typing,codespell

check_imports: $(shell find $(PACKAGE_NAME) -name '*.py') ## Check missing imports
	@poetry run python ./scripts/check_imports.py $^

######################
# CLEANING
######################

clean: ## Clean all generated files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +

######################
# HELP
######################

help: ## Print this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
