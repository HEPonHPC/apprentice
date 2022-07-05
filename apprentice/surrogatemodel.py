import numpy as np
from apprentice.space import Space

class SurrogateModel(object):
    def __init__(self,dim,fnspace=None,**kwargs):
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
        raise Exception("The function objective must be implemented in the derived class")

    @classmethod
    def from_file(cls,filepath,**kwargs):
        raise Exception("The function objective must be implemented in the derived class")

    def fit(self,X,Y):
        raise Exception("The function objective must be implemented in the derived class")

    def f_x(self,x):
        raise Exception("The function objective must be implemented in the derived class")

    def f_X(self,x):
        raise Exception("The function objective must be implemented in the derived class")

    @property
    def training_size(self):
        ts = self.training_size_ if hasattr(self, "training_size_") else 0
        return ts

    @property
    def function_space(self):
        return self.fnspace_

    @property
    def has_gradient(self):
        """
        Return true if an implementation of gradient is found.
        """
        return hasattr(self, "gradient")

    @property
    def has_hessian(self):
        """
        Return true if an implementation of hessian is found.
        """
        return hasattr(self, "hessian")

    @property
    def fnspace(self):
        return self.fnspace_

    @property
    def bounds(self):
        bounds_ = np.zeros((self.fnspace_.dim,2))
        for d in range(self.fnspace_.dim):
            bounds_[d][0] = self.fnspace_.a_[d]
            bounds_[d][1] = self.fnspace_.b_[d]
        return bounds_

    @property
    def dim(self):
        return self.fnspace_.dim

    def __call__(self, x):
        return self.f_x(x)






