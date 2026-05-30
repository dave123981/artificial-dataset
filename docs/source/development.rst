Development
===========

Setup
-----

Clone the repository and install the package with all development dependencies:

.. code-block:: bash

   git clone https://github.com/dcintlab/artificial-dataset.git
   cd artificial-dataset
   pip install -e ".[dev]"

Install the pre-commit hooks so linting runs automatically before each commit:

.. code-block:: bash

   pre-commit install

Running the tests
-----------------

.. code-block:: bash

   pytest

To also measure code coverage:

.. code-block:: bash

   pytest --cov=artificial_dataset --cov-report=term-missing

Code style
----------

The project uses `Black <https://black.readthedocs.io>`_ for formatting,
`Ruff <https://docs.astral.sh/ruff>`_ for linting, and
`mypy <https://mypy.readthedocs.io>`_ for static type checking.
All three run automatically via pre-commit, but can also be invoked manually:

.. code-block:: bash

   black .
   ruff check .
   mypy artificial_dataset

Building the documentation
--------------------------

Install the documentation dependencies and build with Sphinx:

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build -b html docs/source docs/_build/html

The rendered HTML is then available at ``docs/_build/html/index.html``.
