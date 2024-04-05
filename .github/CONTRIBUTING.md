# Contribute Code

To contribute to this project, please follow the ["fork and pull request"](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) workflow. Please do not try to push directly to this repo.

Note related issues and tag relevant maintainers in pull requests.

Pull requests cannot land without passing the formatting, linting, and testing checks first. See [Testing](#testing) and
[Formatting and Linting](#formatting-and-linting) for how to run these checks locally.

It's essential that we maintain great documentation and testing. Add or update relevant unit or integration test when possible. 
These live in `tests/unit_tests` and `tests/integration_tests`. Example notebooks and documentation lives in `/docs` inside the
LangChain repo [here](https://github.com/langchain-ai/langchain/tree/master/docs).

We are a small, progress-oriented team. If there's something you'd like to add or change, opening a pull request is the
best way to get our attention.

## 🚀 Quick Start

This quick start guide explains how to setup the repository locally for development.

### Dependency Management: Poetry and other env/dependency managers

This project utilizes [Poetry](https://python-poetry.org/) v1.7.1+ as a dependency manager.

❗Note: *Before installing Poetry*, if you use `Conda`, create and activate a new Conda env (e.g. `conda create -n langchain python=3.9`)

Install Poetry: **[documentation on how to install it](https://python-poetry.org/docs/#installation)**.

❗Note: If you use `Conda` or `Pyenv` as your environment/package manager, after installing Poetry,
tell Poetry to use the virtualenv python environment (`poetry config virtualenvs.prefer-active-python true`)

The instructions here assume that you run all commands from the `libs/aws` directory. 

```bash
cd libs/aws
```

### Install for development

```bash
poetry install --with lint,typing,test,test_integration,dev
```

Then verify the installation.

```bash
make test
```

If during installation you receive a `WheelFileValidationError` for `debugpy`, please make sure you are running
Poetry v1.6.1+. This bug was present in older versions of Poetry (e.g. 1.4.1) and has been resolved in newer releases.
If you are still seeing this bug on v1.6.1+, you may also try disabling "modern installation"
(`poetry config installer.modern-installation false`) and re-installing requirements.
See [this `debugpy` issue](https://github.com/microsoft/debugpy/issues/1246) for more details.

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

### Formatting and Linting

Formatting ensures that the code in this repo has consistent style so that the
code looks more presentable and readable. It corrects these errors when you run
the formatting command. Linting finds and highlights the code errors and helps 
avoid coding practicies that can lead to errors. 

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

We recognize linting can be annoying - if you do not want to do it, please contact a project maintainer, and they can help you with it. We do not want this to be a blocker for good code getting contributed.

#### Spellcheck

Spellchecking for this project is done via [codespell](https://github.com/codespell-project/codespell).
Note that `codespell` finds common typos, so it could have false-positive (correctly spelled but rarely used) and false-negatives (not finding misspelled) words.

To check spelling for this project:

```bash
make spell_check
```

To fix spelling in place:

```bash
make spell_fix
```

If codespell is incorrectly flagging a word, you can skip spellcheck for that word by adding it to the codespell config in the `pyproject.toml` file.

```python
[tool.codespell]
...
# Add here:
ignore-words-list = 'momento,collison,ned,foor,reworkd,parth,whats,aapply,mysogyny,unsecure'
```
