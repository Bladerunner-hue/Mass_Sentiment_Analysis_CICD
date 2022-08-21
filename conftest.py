# content of conftest.py


def pytest_configure(config):
    """
    Allows plugins and conftest files to perform initial configuration.
    This hook is called for every plugin and initial conftest
    file after command line options have been parsed.
    """


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """


def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    """


def pytest_unconfigure(config):
    """
    called before test process is exited.
    """

# content of conftest.py

pytest_plugins = [
    'plugins.example_plugin',
]


# content of plugins/example_plugin.py
def pytest_configure(config):
    pass


def pytest_unconfigure(config):
    pass