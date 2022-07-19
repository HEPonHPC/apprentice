
.. _apprentice_tutorial_surrogatemodels_gp:

============================================================
Using apprentice to construct and use Gaussian Process
============================================================

The Bayesian regression approach is a probabilistic approach to find the
posterior of the coefficients of a function given the input-output data points.
This approach provides a distribution over the coefficients
that gets updated whenever new data points are observed.

The Gaussian Process (GP) approach, in contrast, consists of a collection of
latent functions such that it describes a Gaussian distribution over all functions
that are consistent with the observed data.
A GP is like an infinite dimensional (in coefficients) and multivariate Gaussian
distribution where any finite collection of r.v.'s i.e., labels are jointly distributed.

A GP begins with a prior distribution and updates this with the observed data
points, producing the posterior distribution over functions.
Here the prior is specified on the function space that is converted to a
posterior distribution using the observed data. From this distribution,
we can obtain predictions on a point of interest as a joint distribution over
the trained labels and the label at the point of interest.

In this tutorial, we describe how to setup the surrogate model construction problem,
the options to construct the GP model, how to store and
use the the GP model. More specifically, in this tutorial, you will
be shown how to:

* Test the install
* Set up the GP surrogate model construction problem
* Construct the GP surrogate model
* Use the GP surrogate model

Getting started
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install apprentice_, execute the following commands::

    git clone git@github.com:HEPonHPC/apprentice.git
    cd  apprentice/
    pip install .
    cd ..

Then, test the installation as described in the
:ref:`test installation documentation<apprentice_test_the_install>`.

Construct gaussian process surrogate model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are multiple ways to construct a gaussian process (GP) object.

From interpolation points
************************************************************************

To construct a GP object using ``from_interpolation_points``,
we need data of size :math:`d \times N_p`,
where :math:`d` is the dimension and the :math:`N_p` is the number of data points.
Additionally, we need arguments that describe the strategy, kernel, and other
relevant model parameters::

  GP = GaussianProcess.from_interpolation_points(X,Y,
                                                seed=<int>,
                                                kernel=<str>,
                                                max_restarts=<int>,
                                                keepout_percentage=<float>,
                                                mean_surrogate_model=<object of apprentice.SurrogateModel>,
                                                error_surrogate_model=<object of apprentice.SurrogateModel>,
                                                sample_size=<int>,
                                                stopping_bound=<float>,
                                                strategy=<str>
                                              )


In this call,

* X is 2-D an array of size :math:`d \times N_p` and it is the x data values to fit
* Y is 1-D an array of size :math:`N_p` and it is the y data values to fit
* kernel is the GP kernel to use. Allowed kernels include:

  * sqe: Squared exponential kernel
  * ratquad: Rational quadratic kernel
  * matern32: Matern 3/2 kernel
  * matern52: Matern 5/2 kernel
  * poly: Polynomial kernel
  * or: Hybrid OR kernel (all of the above kernels summed together)

* max_restarts is the maximum number of restarts to use in the hyperparameter tuning
  problem
* keepout_percentage is the value in percent of the amount of holdout data
  to be used for testing. So (100-keepout_percentage)% of data will be used
  for training the GP
* mean_surrogate_model: surrogate model over the prior mean
* error_surrogate_model: surrogate model over the prior heteroschedastic variance
* sample_size: number of samples of training dataset at each data point
* stopping_bound: stopping condition for heteroschedastic GP tuning
* strategy is the strategy to use

  * ``strategy = "1"``: Most Likely Heteroscedastic Gaussian Process (HeGP-ML)
  * ``strategy = "2"``: Heteroscedastic Gaussian Process using Stochastic Kriging (HeGP-SK)
  * ``strategy = "3"``: Homoscedastic Gaussian Process (HoGP)

Once the GP is constructed using ``from_interpolation_points``,
the coefficients and other metrics of the GP fit can be obtained
into a variable as a ``dict`` using::

  GP_output = GP.as_dict

Additionally, to save the coefficients and other metrics of the GP fit
to a file at location ``<file location>`` use::

  GP.save(<file location>)


From data structure
********************************

Construct a saved GP object using ``from_data_structure`` from a variable
``GP_output``::

  GP_from_data_structure = GaussianProcess.from_data_structure(GP_output)


From file
********************************

Construct a saved GP object using ``from_file`` from file location
``<file location>`` using::

  GP_from_file = GaussianProcess.from_file(tmp_file)


Operations allowed on the GP object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following operations can be performed on a constructed GP object ``GP``.

* y = GP(x): compute the GP at a single point ``x``,
  an array of size :math:`d`. ``y`` is a single value of type ``float``
* Y = GP.f_X(X): compute the GP at multiple points ``X``,
  an array of size :math:`d \times N_p`. ``Y`` is an array of size :math:`N_p`.
* GP_output = GP.as_dict: get the hyperparameters and other metrics of the GP
  fit into a variable. ``GP_output`` is a ``dict``.
* GP.save(<file location>): save the hyperparameters and other metrics of the GP
  fit into a file at location ``<file location>``

More information about the code is in the :ref:`code documentation<apprentice_code_documentation_surrogate_model_gp>`
for GP. Additionally, the `GP unit test script`_ contains the
construction and usage of the operations over the GP object.

.. _`GP unit test script`: https://github.com/HEPonHPC/apprentice/blob/main/apprentice/test_gaussianprocess.py
.. _apprentice: https://github.com/HEPonHPC/apprentice/tree/main
