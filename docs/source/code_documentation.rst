

.. _apprentice_code_documentation:

======================================================
Code Documentation
======================================================

Here we give detailed code documentation of the classes, constructors, and functions
within APPRENTICE. Here is an index of all the code documentation presented in this page.

* Surrogate model

  * :ref:`Surrogate model base class<apprentice_code_documentation_surrogate_model>`
  * :ref:`Available implementation of surrogate model: Polynomial Approximation<apprentice_code_documentation_surrogate_model_pa>`
  * :ref:`Available implementation of surrogate model: Rational Approximation<apprentice_code_documentation_surrogate_model_ra>`
  * :ref:`Available implementation of surrogate model: Gaussian Process<apprentice_code_documentation_surrogate_model_gp>`

* Predition function

  * :ref:`Function base class<apprentice_code_documentation_function_function>`
  * :ref:`Available implementation of Function: LeastSquares<apprentice_code_documentation_function_lsq>`
  * :ref:`Available implementation of Function: GeneratorTuning<apprentice_code_documentation_function_gt>`

* Function Optimization

  * :ref:`Minimizer base class<apprentice_code_documentation_optimization_minimizer>`

* :ref:`Monomial<apprentice_code_documentation_monomial>`
* :ref:`Function parameter space<apprentice_code_documentation_space>`
* :ref:`Utility<apprentice_code_documentation_utility>`

Surrogate models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _apprentice_code_documentation_surrogate_model:

Base class
************************************************************************

SurrogateModel (apprentice/surrogatemodel.py)
------------------------------------------------------

.. autoclass:: apprentice.surrogatemodel.SurrogateModel
   :member-order: bysource
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__


Available implementation
************************************************************************

.. _apprentice_code_documentation_surrogate_model_pa:

PolynomialApproximation (apprentice/polynomialapproximation.py)
----------------------------------------------------------------------

.. autoclass:: apprentice.polynomialapproximation.PolynomialApproximation
   :member-order: bysource
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__


.. _apprentice_code_documentation_surrogate_model_ra:

RationalApproximation (apprentice/rationalapproximation.py)
----------------------------------------------------------------------

.. autoclass:: apprentice.rationalapproximation.RationalApproximation
   :member-order: bysource
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

.. _apprentice_code_documentation_surrogate_model_gp:

GaussianProcess (apprentice/gaussianprocess.py)
----------------------------------------------------------------------

.. autoclass:: apprentice.gaussianprocess.GaussianProcess
   :member-order: bysource
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

Prediction function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _apprentice_code_documentation_function_function:

Base class
************************************************************************

Function (apprentice/function.py)
----------------------------------------------------------------------

.. autoclass:: apprentice.function.Function
   :member-order: bysource
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

Available implementation
************************************************************************

.. _apprentice_code_documentation_function_lsq:

LeastSquares (apprentice/leastsquares.py)
----------------------------------------------------------------------

.. autoclass:: apprentice.leastsquares.LeastSquares
   :member-order: bysource
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

.. _apprentice_code_documentation_function_gt:

GeneratorTuning (apprentice/generatortuning.py)
----------------------------------------------------------------------

.. autoclass:: apprentice.generatortuning.GeneratorTuning
   :member-order: bysource
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

Function Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _apprentice_code_documentation_optimization_minimizer:

Base class
************************************************************************

Minimizer (apprentice/minimizer.py)
----------------------------------------------------------------------

.. autoclass:: apprentice.minimizer.Minimizer
   :member-order: bysource
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__


Available implementation
************************************************************************

.. _apprentice_code_documentation_optimization_scipymin:

ScipyMinimizer (apprentice/minimizer.py)
----------------------------------------------------------------------

.. autoclass:: apprentice.scipyminimizer.ScipyMinimizer
   :member-order: bysource
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

.. _apprentice_code_documentation_monomial:

Monomial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: apprentice.monomial
   :member-order: bysource
   :members:
   :undoc-members:
   :show-inheritance:


.. _apprentice_code_documentation_space:

Function parameter space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: apprentice.space.Space
   :member-order: bysource
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__



Utility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _apprentice_code_documentation_utility:

.. autoclass:: apprentice.util.Util
   :member-order: bysource
   :members:
   :undoc-members:
   :show-inheritance:
