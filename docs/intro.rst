
Introduction
============

This package provides a simple Python framework to implement Expectation
Maxmimization (EM) based massive parallel machine leraning algorithms.

ToDo: The focus of this library is to allow rapid prototyping of algorithms
while at the same time providing a high degree of scalability.


Installation 
============

The recommended approch to install the framework is to obtain 
the most recent stable version from `github.com`:

.. code-block:: bash

    git checkout https://github.com/ml-uol/prosper.git
    cd prosper
    python setup.py develop --user

After installation you should run the testsuite to ensure all neccessary 
dependencies are installed correctly and that everyting works as expected:

.. code-block:: bash

    nosetests -v


Running examples 
================

You are now ready to run your first dictionary learning experiments on artificial 
data.

Create some artifical training data by running `bars-create-data.py`:

.. code-block:: bash

    cd examples/barstests
    python bars-create-data.py

ToDo: this does not work right now.
