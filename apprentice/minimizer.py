from apprentice.space import Space

class Minimizer(object):
    def __init__(self, function, **kwargs):
        self.function_ = function
        self.gradient_ = function.gradient if function.has_gradient else None
        self.hessian_  = function.hessian  if function.has_hessian else None
        self.bounds_ = self.function_.bounds
        self.constraints_ = ()

    def set_bounds(self, bounds):
        self.bounds__ = bounds

    def minimize(self, x0):
        """
        """
        raise Exception("This must be implemented in the derived class")

    def sample(self, npoints, **kwargs):
        return Space.sample_main(self.bounds_[:,0], self.bounds_[:,1], npoints, **kwargs)

