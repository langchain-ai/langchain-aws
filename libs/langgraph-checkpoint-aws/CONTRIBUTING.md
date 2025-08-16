# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.


## Reporting Bugs/Feature Requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features.

When filing an issue, please check existing open, or recently closed, issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment


## Contributing via Pull Requests
Contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

1. You are working against the latest source on the *main* branch.
2. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
3. You open an issue to discuss any significant work - we would hate for your time to be wasted.


## Development Setup

This section provides detailed instructions for setting up your development environment and running tests locally.

### Prerequisites

This project utilizes [Poetry](https://python-poetry.org/) v1.7.1+ as a dependency manager.

❗Note: *Before installing Poetry*, if you use `Conda`, create and activate a new Conda env (e.g. `conda create -n langgraph-checkpoint-aws python=3.9`)

Install Poetry: **[documentation on how to install it](https://python-poetry.org/docs/#installation)**.

❗Note: If you use `Conda` or `Pyenv` as your environment/package manager, after installing Poetry,
tell Poetry to use the virtualenv python environment (`poetry config virtualenvs.prefer-active-python true`)

### Installation

All commands should be run from the `libs/langgraph-checkpoint-aws` directory:

```bash
cd libs/langgraph-checkpoint-aws
```

Install all development dependencies:

```bash
poetry install --with dev,test,lint,typing,codespell,test_integration
```

### Testing

#### Unit Tests

Unit tests cover modular logic that does not require calls to outside APIs:

```bash
make tests
```

To run a specific unit test:

```bash
make test TEST_FILE=tests/unit_tests/specific_test.py
```

#### Integration Tests

Integration tests cover end-to-end functionality with AWS services:

```bash
make integration_tests
```

To run a specific integration test:

```bash
make integration_test TEST_FILE=tests/integration_tests/specific_test.py
```

### Code Coverage

This project uses [coverage.py](https://github.com/nedbat/coveragepy) to track code coverage during testing.

#### Running Tests with Coverage

To run unit tests with coverage:

```bash
make coverage_tests
```

To run all integration tests with coverage:

```bash
make coverage_integration_tests
```

To run a specific integration test with coverage:

```bash
make coverage_integration_test TEST_FILE=tests/integration_tests/specific_test.py
```

#### Viewing Coverage Reports

**Terminal Report:**
```bash
make coverage_report
```

**HTML Report:**
```bash
make coverage_html
```

The HTML report will be generated in the `htmlcov/` directory. Open `htmlcov/index.html` in your browser for detailed analysis.

### Code Quality

#### Formatting

Code formatting is done via [ruff](https://docs.astral.sh/ruff/rules/):

```bash
make format
```

#### Linting

Linting is done via [ruff](https://docs.astral.sh/ruff/rules/):

```bash
make lint
```

To automatically fix linting issues:

```bash
make lint_fix
```

#### Type Checking

Type checking is done via [mypy](http://mypy-lang.org/):

```bash
make lint_tests
```

#### Spell Checking

Spell checking is done via [codespell](https://github.com/codespell-project/codespell):

```bash
make spell_check
```

To fix spelling issues:

```bash
make spell_fix
```

### Clean Up

To clean generated files and caches:

```bash
make clean
```

## Finding contributions to work on
Looking at the existing issues is a great way to find something to contribute on. As our projects, by default, use the default GitHub issue labels (enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any 'help wanted' issues is a great place to start.


## Code of Conduct
This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Security issue notifications
If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.


## Licensing

See the [LICENSE](LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.