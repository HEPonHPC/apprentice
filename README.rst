======================================================
Quickstart
======================================================

Overview of Apprentice
~~~~~~~~~~~~~~~~~~~~~~~~

The core functionality of Apprentice is construction of a multivariate analytic
surrogate model to computationally expensive Monte-Carlo predictions.
The surrogate model is used for numerical optimization of a prediction function
since it can be prohibitively expensive to perform optimization over functions
with the Monte-Carlo predictions.
To summarize, Apprentice can be used for performing three tasks:

  1. Construct :ref:`surrogate models<apprentice_surrogatemodels>` to computationally expensive Monte-Carlo predictions
  2. Formulate a :ref:`prediction function<apprentice_functions>` with surrogate models
  3. Perform :ref:`numerical optimziation<apprentice_optimization>` over the prediction function

.. _apprentice_dependencies:

Dependencies
~~~~~~~~~~~~

Required dependencies:

* Python_ 3.7
* NumPy_ 1.15.0 or above
* SciPy_ 1.7.2 or above
* Pyomo_ 6.4.0 or above

Optional dependencies:

* numba_ 0.40.0 or above
* h5py_ 2.8.0 or above
* matplotlib_ 3.0.0 or above
* GPy_ 1.9.9 or above (required if gaussian process surrogate model object is constructed)

For running with the mpi4py parallelism:

* A functional MPI 1.x/2.x/3.x implementation, such as MPICH_, built with shared/dynamic libraries
* mpi4py_ v3.0.0 or above

For compiling this documentation:

* Sphinx-RTD_

.. _apprentice_initial_install:

Installation
~~~~~~~~~~~~

To install apprentice_, execute the following commands::

    git clone git@github.com:HEPonHPC/apprentice.git
    cd  apprentice/
    pip install .
    cd ..

.. _apprentice_test_the_install:

Testing the installation
~~~~~~~~~~~~~~~~~~~~~~~~

To test the install, run the unit tests_ over all modules of apprentice::

    cd  apprentice/apprentice
    python -m unittest discover .

.. _tests: https://github.com/HEPonHPC/apprentice/tree/master/apprentice
.. _apprentice: https://github.com/HEPonHPC/apprentice
.. _Pyomo: http://www.pyomo.org
.. _h5py: https://www.h5py.org
.. _numba: https://numba.pydata.org
.. _sklearn: https://scikit-learn.org/stable/
.. _matplotlib: https://matplotlib.org
.. _pyDOE: https://pythonhosted.org/pyDOE/
.. _pyDOE2: https://pypi.org/project/pyDOE2/
.. _pandas: https://pandas.pydata.org
.. _Conda: https://docs.conda.io/en/latest/
.. _mpi4py: https://bitbucket.org/mpi4py/mpi4py
.. _MPICH: http://www.mpich.org/
.. _NumPy: http://www.numpy.org
.. _PyPI: https://pypi.org
.. _SciPy: http://www.scipy.org
.. _Python: http://www.python.org
.. _GPy: https://gpy.readthedocs.io/en/deploy/
.. _Sphinx-RTD: https://sphinx-rtd-tutorial.readthedocs.io/en/latest/install.html
