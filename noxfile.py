import nox


@nox.session
def format(session):
    """Format the code using pre-commit hooks."""
    # Install pre-commit
    session.install("pre-commit")

    # Run pre-commit to format the code
    session.run("pre-commit", "run", "--all-files", "--show-diff-on-failure")


@nox.session
def tests(session):
    """Run the tests using the tests extra defined in pyproject.toml."""
    # Install the package with the test extras
    session.install(".[test]")

    # Run the tests using pytest
    session.run("pytest")


@nox.session
def docs(session):
    """Build the documentation and optionally open it in a browser."""
    # Install Sphinx and the documentation dependencies
    session.install(".[docs]")

    # Build the documentation
    session.run("sphinx-build", "doc", "_build")

    # Optional: Open the documentation in a web browser
    session.run("open", "_build/index.html")  # On macOS
    # For Linux, you might use `xdg-open`
    # For Windows, you might use `start`
