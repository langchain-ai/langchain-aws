######################
# NON FILE TARGETS
######################
.PHONY: all format lint test tests integration_tests docker_tests help extended_tests coverage_integration_tests coverage_integration_test coverage_tests coverage_report coverage_html

######################
# ALL TARGETS
######################

all: help ## Default target = help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

######################
# TEST CASES
######################

# Define a variable for the test file path.
test test_watch tests: TEST_FILE ?= tests/unit_tests/
integration_test integration_tests: TEST_FILE = tests/integration_tests/

# Define a variable for Python and notebook files.
PYTHON_FILES=.

tests: ## Run all unit tests
	env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN -u BEDROCK_AWS_ACCESS_KEY_ID -u BEDROCK_AWS_SECRET_ACCESS_KEY uv run --group test pytest --disable-socket --allow-unix-socket $(TEST_FILE)

test:  ## Run individual unit test: make test TEST_FILE=tests/unit_test/test.py
	env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN -u BEDROCK_AWS_ACCESS_KEY_ID -u BEDROCK_AWS_SECRET_ACCESS_KEY uv run --group test pytest --disable-socket --allow-unix-socket $(TEST_FILE)

integration_tests: ## Run all integration tests
	uv run --group test --group test_integration pytest -n auto $(TEST_FILE)

integration_test: ## Run individual integration test: make integration_test TEST_FILE=tests/integration_tests/integ_test.py
	uv run --group test --group test_integration pytest -n auto $(TEST_FILE)

test_watch: ## Run and interactively watch unit tests
	uv run --group test ptw --snapshot-update --now . -- -vv $(TEST_FILE)

coverage_integration_tests: ## Run integration tests with coverage
	uv run --group test --group test_integration pytest --cov=langchain_aws --cov-report=html --cov-report=term-missing --cov-branch tests/integration_tests/

coverage_integration_test: ## Run specific integration test with coverage: make coverage_integration_test TEST_FILE=tests/integration_tests/integ_test.py
	uv run --group test --group test_integration pytest --cov=langchain_aws --cov-report=html --cov-report=term-missing --cov-branch $(TEST_FILE)

coverage_tests: ## Run unit tests with coverage
	env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN -u BEDROCK_AWS_ACCESS_KEY_ID -u BEDROCK_AWS_SECRET_ACCESS_KEY uv run --group test pytest --cov=langchain_aws --cov-report=html --cov-report=term-missing --cov-branch tests/unit_tests/

coverage_report: ## Generate coverage report
	uv run --group test coverage report

coverage_html: ## Generate HTML coverage report
	uv run --group test coverage html

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

lint lint_diff lint_package lint_tests: ## Run linter
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && uv run --all-groups mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff: ## Run code formatter
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check --fix $(PYTHON_FILES)

######################
# DEPENDENCIES
######################

check_imports: $(shell find langchain_aws -name '*.py') ## Check missing imports
	uv run --all-groups python ./scripts/check_imports.py $^

######################
# HELP
######################

help: ## Print this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'