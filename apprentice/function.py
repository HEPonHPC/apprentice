import numpy as np
from apprentice import space
from apprentice import surrogatemodel

from apprentice.util import Util



class Function(object):
    """
    Function base class
    """
    def __init__(self, dim, fnspace=None, **kwargs):
        """

        Initialize the Function class

        :param dim: parameter dimension
        :type dim: int
        :param fnspace: function space object
        :type fnspace: apprentice.space.Space

        """
        self.dim_ = dim # raise Exception if no dim given
        if fnspace is None:
            self.fnspace_ = space.Space(dim, [-np.inf for d in range(dim)], [np.inf for d in range(dim)])
        else:
            if isinstance(fnspace, space.Space):
                self.fnspace_ = fnspace
            else:
                try:
                    self.fnspace_ = space.Space(fnspace)
                except Exception as e:
                    print("Unable to interpret space argument:", e)

        # The currentpoint where we evaluate the function at.
        self.currpoint_ = np.empty(self.dim_, dtype=np.float64)

        # Set up bounds for the parameter space
        self.bounds_ = np.zeros((dim,2))
        if 'bounds' in kwargs:    self.set_bounds(kwargs['bounds'][0], kwargs['bounds'][1])
        else:                     self.set_bounds(self.fnspace_.a_   , self.fnspace_.b_)

        # Fix parameters
        self.fixed_indices_ = ([],)
        self.fixed_values_  = []
        if 'fixed' in kwargs: self.set_fixed_parameters(kwargs["fixed"])
        else:                 self.set_fixed_parameters([])


    def set_bounds(self, bmin, bmax):
        """

        Set minimum and maximum bounds

        :param bmin: bound minimum
        :type bmin: list
        :param bmax: bound maximum
        :type bmax: list

        """

        for d in range(self.dim_):
            self.bounds_[d][0] = bmin[d]
            self.bounds_[d][1] = bmax[d]

    def set_fixed_parameters(self, fixed):
        """

        Set fixed dimension onf the function

        :param fixed: fixed dimensions of the function
        :type fixed: list

        """
        self.fixed_indices_ = ([fx[0] for fx in fixed],)
        self.fixed_values_  =  [fx[1] for fx in fixed]

        for fx in fixed: self.currpoint_[fx[0]] = fx[1]

        self.free_indices_  = ([i for i in range(self.dim_) if not i in self.fixed_indices_[0]],)

        # TODO add debug message!



    @classmethod
    def mk_empty(cls, dim):
        """

        A class function to make an empty function

        :param dim: parameter dimension
        :type dim: int

        """
        return cls(dim)

    @classmethod
    def from_space(cls, spc, **kwargs):
        """

        A class function to make a function from parameter space

        :param spc: parameter space
        :type spc: apprentice.space.Space

        """
        if Util.inherits_from(spc, 'Space'):
            return cls(spc.dim, spc, **kwargs)
        else:
            try:
                return cls(len(spc), space.Space.fromList(spc), **kwargs)
            except Exception as e:
                print("Unable to interpret list argument as Space")


    @classmethod
    def from_surrogates(cls, surrogates):
        """

        A class function to make a function from surrogates

        :param surrogates: surrogate models for all terms for the function
        :type surrogates: list

        """
        checkdim = None
        for s in surrogates:
            if not Util.inherits_from(s, 'SurrogateModel'):
                raise Exception("Object of type {} does not derive from SurrogateModel".format(type(s)))
            if checkdim is None:
                checkdim = s.dim
            else:
                if not s.dim == checkdim:
                    raise Exception("Encountered objects of different dimensions: {} and {} ".format(s.dim, checkdim))

        return cls(checkdim, surrogates=surrogates)


    @property
    def dim(self):
        """

        Get the parameter dimension value

        :return: parameter dimension
        :rtype: int

        """
        return self.dim_

    @property
    def bounds(self):
        """

        Get bounds of the parameter space

        :return: parameter bounds
        :rtype: np.array

        """
        return self.bounds_

    @property
    def has_gradient(self):
        """

        Check if gradient is implemented

        :return: true if an implementation of gradient is found.
        :rtype: bool

        """
        return hasattr(self, "gradient")

    @property
    def has_hessian(self):
        """

        Check if hessian is implemented

        :return: true if an implementation of hessian is found.
        :rtype: bool

        """
        return hasattr(self, "hessian")

    def objective(self,x):
        """

        Compute the function objective at a new data point.
        This function needs to be implemented in a class that
        inherits this class


        :param x: a new x point, an araay of size :math:`dim` where :math:`dim` is the parameter dimension
        :type x: list
        :return: objective value of the function at the new point
        :rtype: float

        """
        raise Exception("The function objective must be implemented in the derived class")

    def __call__(self, x):
        """

        Compute the function objective at a free indices at a new data point

        :param x: a new x point, an array of size :math:`dim` where :math:`dim` is the parameter dimension
        :type x: list
        :return: objective value of the function at the new point
        :rtype: float

        """
        self.currpoint_[self.free_indices_] = x
        return self.objective(self.currpoint_)

