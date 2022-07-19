
.. _apprentice_tutorial_surrogatemodels_ra:
.. _apprentice_tutorial_surrogatemodels_ra_mn:
.. _apprentice_tutorial_surrogatemodels_ra_m1:

====================================================================================
Using apprentice to construct and use Rational Approximation Surrogate Model
====================================================================================

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rational approximation models are surrogate models that can be used to approximate
a continuous function. The rational approximation can be constructed with
numerator order :math:`m\in\mathbb{Z}^+` and denominator order :math:`n\in\mathbb{Z}^+`.

In this tutorial, we describe how to setup the surrogate model construction problem,
the options to construct the rational approximation model, how to store and
use the the rational approximation model. More specifically, in this tutorial, you will
be shown how to:

* Test the install
* Set up the rational approximation surrogate model construction problem
* Construct the rational approximation surrogate model
* Use the rational approximation surrogate model

Getting started
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install apprentice_, execute the following commands::

    git clone git@github.com:HEPonHPC/apprentice.git
    cd  apprentice/
    pip install .
    cd ..

Then, test the installation as described in the
:ref:`test installation documentation<apprentice_test_the_install>`.

Construct rational approximation surrogate model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are multiple ways to construct a rational
approximation object.

From interpolation points
************************************************************************

Before constructing the object
========================================

To construct a rational approximation object using ``from_interpolation_points``,
we need data of size :math:`d \times N_p`,
where :math:`d` is the dimension and the :math:`N_p` is the number of data points.
Note that :math:`N_p \ge {d + m \choose m} + {d + n \choose n}`.
Additionally, we need arguments that describe the strategy, order and other
relevant model parameters::

  from apprentice import RationalApproximation
  R = RationalApproximation.from_interpolation_points(X,Y,
                                                      m = <int>,
                                                      n = <int>,
                                                      strategy = <int>,
                                                      pnames=[
                                                              <str>,
                                                              <str>,
                                                              ...
                                                      ])

In this call,

* X is 2-D an array of size :math:`d \times N_p` and it is the x data values to fit
* Y is 1-D an array of size :math:`N_p` and it is the y data values to fit
* m is the order of the numerator polynomial
* n is the order of the denominator polynomial
* pnames are the names of the dimension and it is an array of size :math:`d`.
* strategy is the strategy to use

  * ``strategy = "1.1"``: find pole-free coefficients of the rational approximation
    with numerator order :math:`m` and denominator order :math:`1`
  * ``strategy = "2.1"``: find coefficients of the rational approximation
    with numerator order :math:`m` and denominator order :math:`n`. The resulting
    rational approximation may contain poles using the linear algebra technique
    proposed in our paper `practical algorithms for multivariate rational approximation`_
    [arXiv_].
  * ``strategy = "2.2"``: find coefficients of the rational approximation
    with numerator order :math:`m` and denominator order :math:`n`.
    F = p/q is reformulated as 0 = p - qF using the Vandermonde matrices.
    That defines the problem Ax = b and we solve for x using using Singular
    Value Decomposition (SVD), exploiting A = U x S x V.T. There is an
    additional manipulation exploiting on setting the constant coefficient in
    q to 1. The resulting rational approximation may contain poles.
  * ``strategy = "2.3"``: find coefficients of the rational approximation
    with numerator order :math:`m` and denominator order :math:`n`.
    F = p/q is reformulated as 0 = p - qF using the Vandermonde matrices.
    That defines the problem Ax = b and we solve for x using using Singular
    Value Decomposition (SVD), exploiting A = U x S x V.T. We get the solution
    as the last column in V (corresponds to the smallest singular value).
    The resulting rational approximation may contain poles.
  * ``strategy = "3.1"``: find coefficients of the rational approximation
    with numerator order :math:`m` and denominator order :math:`n` with minimal
    number of poles. The number of poles is reduced using the semi-infinite
    programming technique proposed in our paper
    `practical algorithms for multivariate rational approximation`_ [arXiv_].

For each strategy above, there maybe additional parameters, some of which
are described below. A complete list of available parameters is described in
:ref:`code documentation<apprentice_code_documentation>`

* ``strategy = "1.1"``

  * ``fit_solver=<str>``: fit solver to use. Examples include:

    * ipopt_
    * filter_
    * scipy_

    If ipopt_ or filter_ is used, then the optimization problem is solved using
    AMPL_ models via Pyomo_. Additionally, you can use any solver that can optimize
    a quadratic objective function with linear constraints and compatible with Pyomo_.
    Please note that if you use ipopt_ or filter_ or any other
    solver as described above, then the executable of the
    solver needs to be set up in your bash environment for Pyomo_ to find.

  * ``use_abstract_model=<bool>``: if the fit solver used requires the construction of AMPL models,
    then using a ``True`` value in this parameter, will construct abstract Pyomo_ models.
    On the other hand, if a false value is used then a concrete Pyomo_ model will be constructed.
    See `Pyomo documentation`_ for more details

* ``strategy = "3.1"``

  * fit_solver=<str>: solver to use in the outer optimization. Examples include:

    * ipopt_
    * filter_
    * scipy_

    If ipopt or filter is used, then the optimization problem is solved using
    AMPL_ models via Pyomo_. Additionally, you can use any solver that can optimize
    a quadratic objective function with linear constraints and compatible with Pyomo_.
    Please note that if you use ipopt_ or filter_ or any other
    solver as described above, then the executable of the
    solver needs to be set up in your bash environment for Pyomo_ to find.
  * local_solver=<str>: solver to use in the inner (local) optimization. Examples include:

    * ipopt_
    * scipy_

    If ipopt_ is used, then the optimization problem is solved using
    AMPL_ models via Pyomo_. Additionally, you can use any solver that can optimize
    a nonlinear objective function and compatible with Pyomo_.
    Please note that if you use ipopt_ or filter_ or any other
    solver as described above, then the executable of the
    solver needs to be set up in your bash environment for Pyomo_ to find.


After constructing the object
========================================

Once the rational approximation is constructed using ``from_interpolation_points``,
the coefficients and other metrics of the rational approximation fit can be obtained
into a variable as a ``dict`` using::

  R_output = R.as_dict

Additionally, to save the coefficients and other metrics of the rational approximation fit
to a file at location ``<file location>`` use::

  R.save(<file location>)


From data structure
********************************

Construct a saved rational approximation object using ``from_data_structure`` from a variable
``R_output``::

  R_from_data_structure = RationalApproximation.from_data_structure(R_output)


From file
********************************

Construct a saved rational approximation object using ``from_file`` from file location
``<file location>`` using::

  R_from_file = RationalApproximation.from_file(tmp_file)


Operations allowed on the rational approximation object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following operations can be performed on a constructed rational approximation object ``R``.

* y = R(x): compute the rational approximation at a single point ``x``,
  an array of size :math:`d`. ``y`` is a single value of type ``float``
* Y = R.f_X(X): compute the rational approximation at multiple points ``X``,
  an array of size :math:`d \times N_p`. ``Y`` is an array of size :math:`N_p`.
* R_output = R.as_dict: get the coefficients and other metrics of the rational
  approximation fit into a variable. ``R_output`` is a ``dict``.
* R.save(<file location>): save the coefficients and other metrics of the rational
  approximation fit into a file at location ``<file location>``
* v = R.coeff_norm: get 1-norm of the coefficients. ``v`` is a single value of type ``float``
* v2 = R.coeff2_norm: get 2-norm of the coefficients. ``v2`` is a single value of type ``float``
* g = R.gradient(x): get the gradient of the rational approximation at point ``x``,
  an array of size :math:`d`. ``g`` is an array of size :math:`d`.


More information about the code is in the :ref:`code documentation<apprentice_code_documentation_surrogate_model_ra>`
for rational approximation. Additionally, the `rational approximation unit test script`_ contains the
construction and usage of the operations over the rational approximation object.

.. _`rational approximation unit test script`: https://github.com/HEPonHPC/apprentice/blob/main/apprentice/test_rationalapproximation.py
.. _apprentice: https://github.com/HEPonHPC/apprentice/tree/main
.. _`practical algorithms for multivariate rational approximation`: https://www.sciencedirect.com/science/article/abs/pii/S0010465520303222
.. _arXiv: https://arxiv.org/abs/1912.02272
.. _AMPL: https://ampl.com
.. _Pyomo: http://www.pyomo.org
.. _`Pyomo documentation`: https://pyomo.readthedocs.io/en/stable/pyomo_overview/abstract_concrete.html
.. _filter: https://epubs.siam.org/doi/10.1137/S105262340038081X
.. _scipy: https://docs.scipy.org/doc/scipy/reference/optimize.html
.. _ipopt: https://coin-or.github.io/Ipopt/
