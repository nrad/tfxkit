Usage Guide
===========

This guide explains how to use `tfxkit` for training and tuning.

Prequisites
------------
- Python 3.12 or higher is required.

Check your Python version:

.. code-block:: bash

   python --version

Install tfxkit from TestPyPI:

.. code-block:: bash

   pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple tfxkit


Installation
------------

Install via pip:


Install from source:

.. code-block:: bash

   git clone git@github.com:nrad/tfxkit.git
   cd tfxkit
   pip install -e .

Run the Example
---------------

Before running the example, download the example data:

.. code-block:: bash

   tfxkit --task download_example_data

Then you can run training with the default configuration:

.. code-block:: bash

   tfxkit


Basic Workflow
--------------

.. code-block:: bash

   tfxkit 


1. Define your config file.
2. Select a model builder.
3. Run training or tuning.
