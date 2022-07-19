.. _apprentice_tutorial_surrogatemodels:

======================================================
Using apprentice to construct surrogate models
======================================================

In apprentice, surrogate models can be constructed that to approximate
a continuous function. For this purpose,
a base class ``SurrogateModel`` is provided in apprentice.

In this tutorial, we describe the ``SurrogateModel`` base class, the available implementations
of the ``SurrogateModel`` base class, and how you can create your own implementation of the
``SurrogateModel`` base class.
More specifically, in this tutorial, we will:

* Test the install
* Learn about the ``SurrogateModel`` base class
* Learn about the available implementations of the ``SurrogateModel`` base class
* Learn how to construct a different implementations of the ``SurrogateModel`` base class

Getting started
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install apprentice_, execute the following commands::

    git clone git@github.com:HEPonHPC/apprentice.git
    cd  apprentice/
    pip install .
    cd ..

Then, test the installation as described in the
:ref:`test installation documentation<apprentice_test_the_install>`.

``SurrogateModel`` base class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, the functions of the ``SurrogateModel`` base class are discussed. More details
can be found in :ref:`function code documentation<apprentice_code_documentation_surrogate_model>`.

.. _apprentice_tutorial_surrogatemodels_construction:

Construction methods
************************************************************************

There are multiple ways to construct a function:

* ``def from_interpolation_points(cls,X, Y,**kwargs)``: construct the surrogate
  model object from data points. ``X`` is 2-D an array of size :math:`d \times N_p`
  and it is the x data values to fit. ``Y`` is 1-D an array of size :math:`N_p` and
  it is the y data values to fit.
* ``def from_data_structure(cls,data_structure,**kwargs)``: construct the
  surrogate model object previous surrogate model fit from a ``data_structure``
  dictionary.
* ``def from_file(cls,filepath,**kwargs)``: construct the
  surrogate model object previous surrogate model fit saved into a file at location
  ``filepath``.


.. _apprentice_tutorial_surrogatemodels_abstract:

Abstract (unimplemented) methods
************************************************************************

``SurrogateModel`` has the following abstract (unimplemented) method.

* ``def fit(self,X,Y)``: Fit the surrogate model from data points. ``X`` is 2-D
  an array of size :math:`d \times N_p` and it is the x data values to fit.
  ``Y`` is 1-D an array of size :math:`N_p` and it is the y data values to fit.
* ``def f_x(self,x)``: compute the surrogate model at a single point ``x``,
  an array of size :math:`d`.
* ``def f_X(self,x)``: compute the surrogate model at multiple points ``X``,
  an array of size :math:`d \times N_p`.

.. _apprentice_tutorial_surrogatemodels_properties:

Properties
************************************************************************

``SurrogateModel`` has several properties. We discuss a few key ones here.

* ``def has_gradient(self)``: returns true if the object inheriting the
  ``SurrogateModel`` base class has a method by the name ``gradient``. This function,
  if implemented should compute the gradient of the model at a certain point.
* ``def has_hessian(self)``: returns true if the object inheriting the
  ``SurrogateModel`` base class has a method by the name ``hessian``. This function,
  if implemented should compute the hessian of the model at a certain point.

Available implementations of the ``SurrogateModel`` base class in apprentice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following implementations of ``SurrogateModel`` is available in apprentice.
For more details about these models, see the tutorials of the respective model.


* :ref:`Polynomial Approximation<apprentice_tutorial_surrogatemodels_pa>`
* :ref:`Rational Approximation<apprentice_tutorial_surrogatemodels_ra>`
* :ref:`Gaussian Process<apprentice_tutorial_surrogatemodels_gp>`

Construct your own implementation of the ``SurrogateModel`` base class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To implement your own surrogate model, all you have to do is to
implement the :ref:`abstract function<apprentice_tutorial_surrogatemodels_abstract>` and
optional :ref:`gradient and hessian functions<apprentice_tutorial_function_function_properties>`.
Then you can construct your object using the
:ref:`construction methods<apprentice_tutorial_surrogatemodels_construction>`.
To override the ``__init__`` constructor method, use the template in the code snippet below::

  def __init__(self, dim, fnspace=None, **kwargs: dict):
      super().__init__(dim, fnspace)
      """
      add additional construction code
      """
      # ...

.. _apprentice: https://github.com/HEPonHPC/apprentice/tree/main
