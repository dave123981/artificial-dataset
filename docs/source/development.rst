Development
===========

Setup
-----

Clone the repository over SSH and install the package with all development
dependencies:

.. code-block:: bash

   git clone git@github.com:dcintlab/artificial-dataset.git
   cd artificial-dataset
   pip install -e ".[dev]"

Pre-commit hooks
----------------

The project uses `pre-commit <https://pre-commit.com>`_ to enforce code style
and linting before every commit.  Install the hooks once after cloning:

.. code-block:: bash

   pre-commit install

From that point on, the hooks run automatically on the files staged for each
commit.  To run them manually against every file in the repository (useful
after changing hook configuration or when setting up a new environment):

.. code-block:: bash

   pre-commit run --all-files

Running the tests
-----------------

The full test command mirrors what runs in CI:

.. code-block:: bash

   pytest --cov=artificial_dataset \
       --durations=0 \
       --cov-report term \
       --cov-report html:coverage-html \
       --cov-report xml:coverage-report.xml \
       --junitxml=junit-report.xml
   coverage-badge -o coverage.svg

Explanation of the flags:

``--cov=artificial_dataset``
    Measure coverage for the ``artificial_dataset`` package only.

``--durations=0``
    Print the runtime of every test at the end of the run, slowest first.
    Useful for spotting unexpectedly slow tests.

``--cov-report term``
    Print a coverage summary to the terminal after the test run.

``--cov-report html:coverage-html``
    Write a browsable HTML coverage report to the ``coverage-html/`` directory.

``--cov-report xml:coverage-report.xml``
    Write a machine-readable XML coverage report, consumed by CI and coverage
    tools such as Codecov.

``--junitxml=junit-report.xml``
    Write a JUnit-compatible XML test report, consumed by CI dashboards.

``coverage-badge -o coverage.svg``
    Generate an SVG badge from the coverage data produced by the preceding run.

Building the documentation
--------------------------

Install the documentation dependencies first:

.. code-block:: bash

   pip install -e ".[docs]"

**Using Make** (recommended, run from the ``docs/`` directory):

.. code-block:: bash

   cd docs
   make html

The rendered HTML is written to ``docs/build/html/index.html``.
Other targets (``make clean``, ``make linkcheck``, …) are available;
run ``make help`` for the full list.

**Using sphinx-build directly** (run from the project root):

.. code-block:: bash

   sphinx-build -W -T -b html docs/source docs/_build/html

``-W`` promotes all warnings to errors so documentation issues are caught
early; ``-T`` prints the full traceback on errors.
