######################
# NON FILE TARGETS
######################
.PHONY: all format lint test tests integration_tests docker_tests help extended_tests

######################
# ALL TARGETS
######################

all: help ## Default target = help

######################
# TEST CASES
######################

# Define a variable for the test file path.
test test_watch tests: TEST_FILE ?= tests/unit_tests/
integration_test integration_tests: TEST_FILE = tests/integration_tests/

# Define a variable for Python and notebook files.
PYTHON_FILES=.

tests: ## Run all Tests
	poetry run pytest $(TEST_FILE)

test:  ## Run Individual Test: make test TEST_FILE=test.py
	poetry run pytest $(TEST_FILE)

integration_tests: ## Run all Integration Tests
	poetry run pytest $(TEST_FILE)

integration_test: ## Run Individual Integration Test: make integration_test TEST_FILE=integ_test.py
	poetry run pytest $(TEST_FILE)

test_watch: ## Run and Interactively Watch Tests
	poetry run ptw --snapshot-update --now . -- -vv $(TEST_FILE)

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/aws --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_aws
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint:         ## Run linter
	poetry run ruff check

lint_diff:    ## Run linter
	poetry run ruff format $(PYTHON_FILES) --diff

lint_package: ## Run linter on Package
	poetry run ruff check --select I $(PYTHON_FILES)

lint_tests:   ## Run linter tests
	mkdir -p $(MYPY_CACHE); poetry run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff: ## Run code formatter
	poetry run ruff format $(PYTHON_FILES)
	poetry run ruff check --select I --fix $(PYTHON_FILES)

spell_check: ## Run code spell check
	poetry run codespell --toml pyproject.toml

spell_fix: ## Run code spell fix
	poetry run codespell --toml pyproject.toml -w

######################
# DEPENDENCIES
######################

install_deps: ## Install tool dependencies
	@pip install --no-cache -U poetry
	@poetry install --with test,codespell,lint

check_imports: $(shell find langchain_aws -name '*.py') ## Check code for Import breakage
	@poetry run python ./scripts/check_imports.py $^

######################
# HELP
######################

help: ## Print this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
