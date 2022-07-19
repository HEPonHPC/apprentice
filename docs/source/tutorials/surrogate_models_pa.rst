
.. _apprentice_tutorial_surrogatemodels_pa:

====================================================================================
Using apprentice to construct and use Polynomial Approximation Surrogate Model
====================================================================================

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Polynomial approximation models are surrogate models that can be used to approximate
a continuous function. The polynomial approximation can be constructed with
order :math:`m\in\mathbb{Z}^+`

In this tutorial, we describe how to setup the surrogate model construction problem,
the options to construct the polynomial approximation model, how to store and
use the the polynomial approximation model. More specifically, in this tutorial, you will
be shown how to:

* Test the install
* Set up the polynomial approximation surrogate model construction problem
* Construct the polynomial approximation surrogate model
* Use the polynomial approximation surrogate model

Getting started
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install apprentice_, execute the following commands::

    git clone git@github.com:HEPonHPC/apprentice.git
    cd  apprentice/
    pip install .
    cd ..

Then, test the installation as described in the
:ref:`test installation documentation<apprentice_test_the_install>`.

Construct polynomial approximation surrogate model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are multiple ways to construct a polynomial
approximation object.

From interpolation points
************************************************************************

To construct a polynomial approximation object using ``from_interpolation_points``,
we need data of size :math:`d \times N_p`,
where :math:`d` is the dimension and the :math:`N_p` is the number of data points.
Note that :math:`N_p \ge {d + m \choose m}`.
Additionally, we need arguments that describe the strategy, order and other
relevant model parameters::

  from apprentice import PolynomialApproximation
  P = PolynomialApproximation.from_interpolation_points(X,Y,
                                                        m = <int>,
                                                        strategy = <int>,
                                                        pnames=[
                                                                <str>,
                                                                <str>,
                                                                ...
                                                        ])

In this call,

* X is 2-D an array of size :math:`d \times N_p` and it is the x data values to fit
* Y is 1-D an array of size :math:`N_p` and it is the y data values to fit
* m is the order of the polynomial
* pnames are the names of the dimension and it is an array of size :math:`d`.
* strategy is the strategy to use

  * ``strategy = 1``: polynomial regression using Singular Value Decomposition (SVD)
  * ``strategy = 2``: solving for polynomial coefficients using least squares regression

Once the polynomial approximation is constructed using ``from_interpolation_points``,
the coefficients and other metrics of the polynomial approximation fit can be obtained
into a variable as a ``dict`` using::

  P_output = P.as_dict

Additionally, to save the coefficients and other metrics of the polynomial approximation fit
to a file at location ``<file location>`` use::

  P.save(<file location>)


From data structure
********************************

Construct a saved polynomial approximation object using ``from_data_structure`` from a variable
``P_output``::

  P_from_data_structure = PolynomialApproximation.from_data_structure(P_output)


From file
********************************

Construct a saved polynomial approximation object using ``from_file`` from file location
``<file location>`` using::

  P_from_file = PolynomialApproximation.from_file(tmp_file)


Operations allowed on the polynomial approximation object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following operations can be performed on a constructed polynomial approximation object ``P``.

* y = P(x): compute the polynomial approximation at a single point ``x``,
  an array of size :math:`d`. ``y`` is a single value of type ``float``
* Y = P.f_X(X): compute the polynomial approximation at multiple points ``X``,
  an array of size :math:`d \times N_p`. ``Y`` is an array of size :math:`N_p`.
* P_output = P.as_dict: get the coefficients and other metrics of the polynomial
  approximation fit into a variable. ``P_output`` is a ``dict``.
* P.save(<file location>): save the coefficients and other metrics of the polynomial
  approximation fit into a file at location ``<file location>``
* v = P.coeff_norm: get 1-norm of the coefficients. ``v`` is a single value of type ``float``
* v2 = P.coeff2_norm: get 2-norm of the coefficients. ``v2`` is a single value of type ``float``
* g = P.gradient(x): get the gradient of the polynomial approximation at point ``x``,
  an array of size :math:`d`. ``g`` is an array of size :math:`d`.
* h = P.hessian(x): get the hessian of the polynomial approximation at point ``x``,
  an array of size :math:`d`. ``h`` is an array of size :math:`d \times d`.

More information about the code is in the :ref:`code documentation<apprentice_code_documentation_surrogate_model_pa>`
for polynomial approximation. Additionally, the `polynomial approximation unit test script`_ contains the
construction and usage of the operations over the polynomial approximation object.

.. _`polynomial approximation unit test script`: https://github.com/HEPonHPC/apprentice/blob/main/apprentice/test_polynomialapproximation.py
.. _apprentice: https://github.com/HEPonHPC/apprentice/tree/main
