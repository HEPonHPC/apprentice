import numpy as np
from apprentice.space import Space

class SurrogateModel(object):
    """
    Surrogate model base class
    """
    def __init__(self,dim,fnspace=None,**kwargs):
        """

        Initialize surrogate models

        :param dim: parameter dimension
        :type dim: int
        :param fnspace: function space
        :type fnspace: apprentice.space.Space

        """
        if fnspace is None:
            self.fnspace_ = Space(dim,
                                        [-np.inf for d in range(dim)],
                                        [np.inf for d in range(dim)],
                                        pnames=kwargs['pnames'] if 'pnames' in kwargs else None)
        else:
            if isinstance(fnspace, Space):
                self.fnspace_ = fnspace
            else:
                raise Exception("Unable to interpret space argument")

    @classmethod
    def from_interpolation_points(cls,X=None, Y=None,**kwargs):
        """

        A class method to construct surrogate model from interpolation points

        :param X: a 2-D array of size :math:`dim \times N_p` and it is the x data values to fit where
            :math:`dim` is the parameter dimension and :math:`N_p` is the the number of data points
        :type X: list
        :param Y: an array of size :math:`N_p` and it is the y data values to fit where
            :math:`N_p` is the the number of data points
        :type Y: list

        """
        X = np.array(X)
        Y = np.array(Y)
        dim = X[0].shape[0]
        sa = kwargs['scale_min'] if 'scale_min' in kwargs else None
        sb = kwargs['scale_max'] if 'scale_max' in kwargs else None
        pnames = kwargs['pnames'] if 'pnames' in kwargs else None
        fnspace = Space(dim,
                    np.amin(X, axis=0),
                    np.amax(X, axis=0),
                    sa=sa,
                    sb=sb,
                    pnames=pnames)
        SM = cls(dim,fnspace,**kwargs)
        SM.training_size_ = len(X)
        SM.fit(X,Y)
        return SM

    @classmethod
    def from_data_structure(cls,data_structure,**kwargs):
        """

        A class method to construct surrogate model from data structure.
        This function needs to be implemented in a class that inherits this class

        :param data_structure: previously fit surrogate model saved as a data structure
        :type data_structure: dict

        """
        raise Exception("The function objective must be implemented in the derived class")

    @classmethod
    def from_file(cls,filepath,**kwargs):
        """

        A class method to construct surrogate model from data structure saved in a file.
        This function needs to be implemented in a class that inherits this class

        :param filepath: file path of a previously fit surrogate model saved as a data structure
        :type filepath: str

        """
        raise Exception("The function objective must be implemented in the derived class")

    def fit(self,X,Y):
        """

        Surrogate model fitting method. This function needs to be implemented in a class that
        inherits this class

        :param X: a 2-D array of size :math:`dim \times N_p` and it is the x data values to fit where
            :math:`dim` is the parameter dimension and :math:`N_p` is the the number of data points
        :type X: list
        :param Y: an array of size :math:`N_p` and it is the y data values to fit where
            :math:`N_p` is the the number of data points
        :type Y: list

        """
        raise Exception("The function objective must be implemented in the derived class")

    def f_x(self,x):
        """

        Calculate the surrogate model at a new point. This function needs to be implemented in a class that
        inherits this class

        :param x: a new x point, an araay of size :math:`dim` where :math:`dim` is the parameter dimension
        :type x: list
        :return: surrogate model value at the new point
        :rtype: float

        """
        raise Exception("The function objective must be implemented in the derived class")

    def f_X(self,x):
        """

        Calculate the surrogate model at a multiple date points.
        This function needs to be implemented in a class that inherits this class

        :param X: multiple x point, an araay of size :math:`dim \times n`
            where :math:`dim` is the parameter dimension and :math:`n` is the number of new data points
        :type X: list
        :return: surrogate model value at the new points an araay of size :math:`n`
            where :math:`n` is the number of new data points
        :rtype: list

        """
        raise Exception("The function objective must be implemented in the derived class")

    @property
    def training_size(self):
        """

        Get the training size

        :return: the training size :math:`N_p`
        :rtype: int

        """
        ts = self.training_size_ if hasattr(self, "training_size_") else 0
        return ts

    @property
    def function_space(self):
        """

        Get the function space object

        :return: function space object
        :rtype: apprentice.space.Space

        """
        return self.fnspace_

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

    @property
    def fnspace(self):
        """

        Get the function space object

        :return: function space object
        :rtype: apprentice.space.Space

        """
        return self.fnspace_

    @property
    def bounds(self):
        """

        Get function space bounds

        :return:  2-D array of size :math:`dim \time 2` containing bounds of parameter space
        :rtype: list

        """
        bounds_ = np.zeros((self.fnspace_.dim,2))
        for d in range(self.fnspace_.dim):
            bounds_[d][0] = self.fnspace_.a_[d]
            bounds_[d][1] = self.fnspace_.b_[d]
        return bounds_

    @property
    def dim(self):
        """

        Get the dimension of the parameter space

        :return: dimension value
        :rtype: int
        """
        return self.fnspace_.dim

    def __call__(self, x):
        """

        Calculate the surrogate model at a new point. This function needs to be implemented in a class that
        inherits this class

        :param x: a new x point, an araay of size :math:`dim` where :math:`dim` is the parameter dimension
        :type x: list
        :return: surrogate model value at the new point
        :rtype: float

        """
        return self.f_x(x)






