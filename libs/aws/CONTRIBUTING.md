# Contribute Code

To contribute to this project, please follow the ["fork and pull request"](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) workflow. Please do not try to push directly to this repo.

Note related issues and tag relevant maintainers in pull requests.

Pull requests cannot land without passing the formatting, linting, and testing checks first. See [Testing](#testing) and
[Formatting and Linting](#formatting-and-linting) for how to run these checks locally.

It's essential that we maintain great documentation and testing. Add or update relevant unit or integration test when possible.
These live in `tests/unit_tests` and `tests/integration_tests`. Example notebooks and documentation lives in `/docs` inside the
[LangChain repo](https://github.com/langchain-ai/langchain/tree/master/docs).

We are a small, progress-oriented team. If there's something you'd like to add or change, opening a pull request is the
best way to get our attention.

## 🚀 Quick Start

This quick start guide explains how to setup the repository locally for development.

### Dependency Management: uv and other env/dependency managers

This project utilizes [uv](https://docs.astral.sh/uv/) as a dependency manager.

❗Note: *Before installing uv*, if you use `Conda`, create and activate a new Conda env (e.g. `conda create -n langchain python=3.10`)

Install uv: **[documentation on how to install it](https://docs.astral.sh/uv/getting-started/installation/)**.

The instructions here assume that you run all commands from the `libs/aws` directory.

```bash
cd libs/aws
```

### Install for development

```bash
uv sync --group lint --group typing --group test --group test_integration --group dev
```

Then verify the installation.

```bash
make test
```

If during installation you encounter any issues with dependency installation, please make sure you are using the latest version of uv.
If you continue to see installation issues, please file an issue with the details of your environment.

### Testing

Unit tests cover modular logic that does not require calls to outside APIs.
If you add new logic, please add a unit test.

To run unit tests:

```bash
make test
```

Integration tests cover the end-to-end service calls as much as possible.
However, in certain cases this might not be practical, so you can mock the
service response for these tests. There are examples of this in the repo,
that can help you write your own tests. If you have suggestions to improve
this, please get in touch with us.

To run the integration tests:

```bash
make integration_test
```

### Code Coverage

This project uses [coverage.py](https://github.com/nedbat/coveragepy) to track code coverage during testing. Coverage reports help identify untested code paths and ensure comprehensive test coverage.

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

After running tests with coverage, you can view the results in several ways:

**Terminal Report:**

```bash
make coverage_report
```

**HTML Report:**

```bash
make coverage_html
```

The HTML report will be generated in the `htmlcov/` directory. Open `htmlcov/index.html` in your browser to view detailed line-by-line coverage analysis.

#### Coverage Configuration

Coverage settings are configured in `pyproject.toml`:

- **Source tracking**: Only code in `langchain_aws/` is measured
- **Branch coverage**: Tracks both line and branch coverage for comprehensive analysis
- **Exclusions**: Test files and common patterns (like `pragma: no cover`) are excluded
- **Reports**: Both terminal and HTML reports show missing lines and coverage percentages

#### Coverage Best Practices

- Aim for high coverage on new code you add
- Use coverage reports to identify untested edge cases
- Add tests for uncovered lines when practical
- Use `# pragma: no cover` sparingly for truly untestable code (like debug statements)

### Formatting and Linting

Formatting ensures that the code in this repo has consistent style so that the
code looks more presentable and readable. It corrects these errors when you run
the formatting command. Linting finds and highlights the code errors and helps
avoid coding practices that can lead to errors.

Run both of these locally before submitting a PR. The CI scripts will run these
when you submit a PR, and you won't be able to merge changes without fixing
issues identified by the CI.

#### Code Formatting

Formatting for this project is done via [ruff](https://docs.astral.sh/ruff/rules/).

To run format:

```bash
make format
```

Additionally, you can run the formatter only on the files that have been modified in your current branch
as compared to the master branch using the `format_diff` command. This is especially useful when you have
made changes to a subset of the project and want to ensure your changes are properly formatted without
affecting the rest of the codebase.

```bash
make format_diff
```

#### Linting

Linting for this project is done via a combination of [ruff](https://docs.astral.sh/ruff/rules/) and [mypy](http://mypy-lang.org/).

To run lint:

```bash
make lint
```

In addition, you can run the linter only on the files that have been modified in your current branch as compared to the master branch using the `lint_diff` command. This can be very helpful when you've made changes to only certain parts of the project and want to ensure your changes meet the linting standards without having to check the entire codebase.

```bash
make lint_diff
```

In addition, you can run the linter only tests.

```bash
make lint_tests
```

We recognize linting can be annoying - if you do not want to do it, please contact a project maintainer, and they can help you with it. We do not want this to be a blocker for good code getting contributed.
