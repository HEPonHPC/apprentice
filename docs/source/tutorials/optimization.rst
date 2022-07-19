.. _apprentice_tutorial_optimization:

======================================================
Using apprentice to perform function optimization
======================================================

In apprentice, we can perform optimization on predition functions so as to minimize
the loss function
For this purpose, a base class ``Minimizer`` is provided in apprentice.

In this tutorial, we describe the ``Minimizer`` base class, the available implementation
of the ``Minimizer`` base class, and how you can create your own implementation of the
``Minimizer`` base class.
More specifically, in this tutorial, we will:

* Test the install
* Learn about the ``Minimizer`` base class
* Learn about the available implementation of the ``Minimizer`` base class
* Learn how to construct a different implementations of the ``Minimizer`` base class

Getting started
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install apprentice_, execute the following commands::

    git clone git@github.com:HEPonHPC/apprentice.git
    cd  apprentice/
    pip install .
    cd ..

Then, test the installation as described in the
:ref:`test installation documentation<apprentice_test_the_install>`.

``Minimizer`` base class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, the functions of the ``Minimizer`` base class are discussed. More details
can be found in :ref:`minimizer code documentation<apprentice_code_documentation_optimization_minimizer>`.


.. _apprentice_tutorial_optimization_construction:

Construction methods
************************************************************************

To construct a ``Minimizer`` object, call the ``__init__`` function::

  def __init__(self, function, **kwargs)

where function is the object of a class that inherits the ``Function`` base class.

.. _apprentice_tutorial_optimization_abstract:

Abstract (unimplemented) methods
************************************************************************

``Minimizer`` has one abstract (unimplemented) method.

* ``def minimize(self,x0)``: Minimize the prediction function with starting point ``x0``.
  This function is required to be implemented in the inheriting class.

Available implementations of the ``Function`` base class in apprentice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ScipyMinimizer
************************************************************************

The implementation of the ``minimize`` function in ScipyMinimizer minimizes
the prediction function using SciPy_. The signature of the abstract function is::

  def minimize(self, x0=<array>, nrestart=<int>, method=<str>, tol=<float>)

where

* ``x0``: starting point
* ``nrestart``: starting point
* ``method``: solver (method) to use. The allowed values include

  * "tnc": `Truncated Newton (TNC) algorithm`_
  * "lbfgsb": `L-BFGS-B algorithm`_
  * "slsqp": `Sequential Least Squares Programming (SLSQP)`_
  * "ncg": `Newton conjugate gradient trust-region algorithm`_

* ``tol``: tolerance value

See :ref:`ScipyMinimizer code documentation<apprentice_code_documentation_optimization_scipymin>`
for more details. Additionally, the `ScipyMinimizer unit test script`_ contains the
construction and usage of the operations over the ``ScipyMinimizer`` object.

Construct your own implementation of the ``Minimizer`` base class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To implement your own minimizer (optimization) function, all you have to do is to
implement the :ref:`abstract function<apprentice_tutorial_optimization_abstract>`.
Then you can construct your object using the
:ref:`construction methods<apprentice_tutorial_optimization_construction>`.
To override the ``__init__`` constructor method, use the template in the code snippet below::

  def __init__(self, function, **kwargs):
      super(<Your class name>, self).__init__(function, **kwargs)
      """
      add additional construction code here
      """
      # ...

.. _`Newton conjugate gradient trust-region algorithm`: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustncg.html
.. _`Sequential Least Squares Programming (SLSQP)`: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html
.. _`L-BFGS-B algorithm`: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
.. _`Truncated Newton (TNC) algorithm`: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-tnc.html
.. _apprentice: https://github.com/HEPonHPC/apprentice/tree/main
.. _SciPy: https://docs.scipy.org/doc/scipy/reference/optimize.html
.. _`ScipyMinimizer unit test script`: https://github.com/HEPonHPC/apprentice/blob/main/apprentice/test_scipyminimizer.py
