from apprentice.space import Space

class Minimizer(object):
    """
    Minimizer (Optimization) base class
    """
    def __init__(self, function, **kwargs):
        """

        Initialize minimizer object

        :param function: function
        :type function: apprentice.function.Function

        """
        self.function_ = function
        self.gradient_ = function.gradient if function.has_gradient else None
        self.hessian_  = function.hessian  if function.has_hessian else None
        self.bounds_ = self.function_.bounds
        self.constraints_ = ()

    def set_bounds(self, bounds):
        """

        Set bounds of the optimization

        :param bounds: bounds of optimization
        :type bounds: np.array

        """
        self.bounds__ = bounds

    def minimize(self, x0):
        """

        Minimize.
        This function needs to be implemented in a class that
        inherits this class

        :param x0: starting point
        :type x0: np.array

        """
        raise Exception("This must be implemented in the derived class")

    def sample(self, npoints, **kwargs):
        """
        Sample the domain

        :param npoints: number of points in the sample
        :type npoints: int
        :return: sample of the domain
        :rtype: np.array

        """
        return Space.sample_main(self.bounds_[:,0], self.bounds_[:,1], npoints, **kwargs)

