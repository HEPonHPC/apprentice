.. _apprentice_tutorial_function:

======================================================
Using apprentice to construct prediction functions
======================================================

In apprentice, prediction function objects can be constructed that describe a loss
function between the surrogate models and some data subject to noise (variance) and
other metrics.
For this purpose, a base class ``Function`` is provided in apprentice.

In this tutorial, we describe the ``Function`` base class, the available implementations
of the ``Function`` base class, and how you can create your own implementation of the
``Function`` base class.
More specifically, in this tutorial, we will:

* Test the install
* Learn about the ``Function`` base class
* Learn about the available implementations of the ``Function`` base class
* Learn how to construct a different implementations of the ``Function`` base class

Getting started
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install apprentice_, execute the following commands::

    git clone git@github.com:HEPonHPC/apprentice.git
    cd  apprentice/
    pip install .
    cd ..

Then, test the installation as described in the
:ref:`test installation documentation<apprentice_test_the_install>`.

``Function`` base class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, the functions of the ``Function`` base class are discussed. More details
can be found in :ref:`function code documentation<apprentice_code_documentation_function_function>`.

.. _apprentice_tutorial_function_function_construction:

Construction methods
************************************************************************

There are multiple ways to construct a function:

* ``def from_space(cls, spc, **kwargs)``: construct the function object from parameter
  domain space. The argument ``spc`` should be an object of a class that inherits the
  ``Space`` class. See :ref:`space code documentation<apprentice_code_documentation_space>`
  for more details. Additionally, the `Space unit test script`_ contains the
  construction and usage of the operations over the ``Space`` object.
* ``def from_surrogates(cls, surrogates)``: construct the function object from
  :ref:`surrogate models<apprentice_surrogatemodels>`. The argument ``surrogates``
  is a list of objects of a class that inherits the ``SurrogateModel`` class.

.. _apprentice_tutorial_function_function_abstract:

Abstract (unimplemented) methods
************************************************************************

``Function`` has one abstract (unimplemented) method.

* ``def objective(self,x)``: Compute the objective function value at point ``x``.
  This function is required to be implemented in the inheriting class.

.. _apprentice_tutorial_function_function_properties:

Properties
************************************************************************

``Function`` has several properties. We discuss a few key ones here.

* ``def has_gradient(self)``: returns true if the object inheriting the
  ``Function`` base class has a method by the name ``gradient``. This function,
  if implemented should compute the gradient of the function at a certain point.
* ``def has_hessian(self)``: returns true if the object inheriting the
  ``Function`` base class has a method by the name ``hessian``. This function,
  if implemented should compute the hessian of the function at a certain point.

Available implementations of the ``Function`` base class in apprentice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Least Squares
************************************************************************

The objective function of least squares implementation class ``LeastSquares`` is:

.. math::

   L_2(p) = \sum_{t=0}^{N_t} w_t^2 \frac{ (M_t(p)-D_t)^2 }{\widetilde{M_t}(p)^2 + \widetilde{D_t}^2}

where

* :math:`N_t`: number of terms e.g., term1, term2, ...
* :math:`w_t`: weight for term t
* :math:`M_t(p)`: surrogate model of mean value or the MC mean value for term t evaluated at parameter value p
* :math:`D_t`: data (mean) value for term t
* :math:`\widetilde{M_t}(p)`: surrogate model of error value or the MC error value for term t evaluated at parameter value p
* :math:`\widetilde{D_t}`: data error for term t

Additionally, the gradient and hessian of this function is also implemented in the
``LeastSquares`` class. See :ref:`least squares code documentation<apprentice_code_documentation_function_lsq>`
for more details. Additionally, the `least squares unit test script`_ contains the
construction and usage of the operations over the ``LeastSquares`` object.

Generator Tuning
************************************************************************

Generator Tuning function implementation ``GeneratorTuning`` inherits ``LeastSquares`` class.
So the objective function, gradient and hessian of this implementation is the same as those of the
``LeastSquares`` implementation. This class provides additional methods to interact with the weights
in the objective function and to remove terms from the objective function.
See :ref:`generator tuning code documentation<apprentice_code_documentation_function_gt>`
for more details. Additionally, the `generator tuning unit test script`_ contains the
construction and usage of the operations over the ``GeneratorTuning`` object.

Construct your own implementation of the ``Function`` base class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To implement your own prediction function, all you have to do is to
implement the :ref:`abstract function<apprentice_tutorial_function_function_abstract>` and
optional :ref:`gradient and hessian functions<apprentice_tutorial_function_function_properties>`.
Then you can construct your object using the
:ref:`construction methods<apprentice_tutorial_function_function_construction>`.
To override the ``__init__`` constructor method, use the template in the code snippet below::

  def __init__(self, dim, fnspace, <optional additional args>, **kwargs):
      super(<Your class name>, self).__init__(dim, fnspace, **kwargs)
      """
      add additional construction code here
      """
      # ...

.. _`Space unit test script`: https://github.com/HEPonHPC/apprentice/blob/main/apprentice/test_space.py
.. _`least squares unit test script`: https://github.com/HEPonHPC/apprentice/blob/main/apprentice/test_leastsquares.py
.. _`generator tuning unit test script`: https://github.com/HEPonHPC/apprentice/blob/main/apprentice/test_generatortuning.py
.. _apprentice: https://github.com/HEPonHPC/apprentice/tree/main
