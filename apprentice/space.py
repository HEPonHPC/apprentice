import numpy as np
class Space(object):
    def __init__(self, dim, a, b, sa=None,sb=None,pnames=None):
        assert(dim == len(a))
        assert(dim == len(b))
        self.dim_ = dim
        self.a_ = np.array(a)
        self.b_ = np.array(b)
        self.pnames_ = pnames
        self.sa_ = np.array(-1*np.ones(dim)) if sa is None else np.array(sa)
        self.sb_ = np.array(np.ones(dim)) if sb is None else np.array(sb)
        self.scaleTerm_ = (self.sb_ - self.sa_)/(self.b_ - self.a_)
        self.jacfac_ = (self.box_scaled[:,1] - self.box_scaled[:,0])/(self.box[:,1] - self.box[:,0])

    @classmethod
    def fromList(cls, lst, pnames=None):
        return cls(len(lst), [l[0] for l in lst], [l[1] for l in lst], pnames)

    @property
    def box(self):
        """
        The scaled world box
        """
        return np.column_stack((self.a_, self.b_))

    @property
    def box_scaled(self):
        """
        The scaled world box
        """
        return np.column_stack((self.sa_, self.sb_))

    @property
    def dim(self): return self.dim_

    @property
    def center(self): return [self.a_[d] + 0.5*(self.b_[d]-self.a_[d]) for d in range(self.dim)]

    @property
    def center_scaled(self): return [self.sa_[d] + 0.5*(self.sb_[d]-self.sa_[d]) for d in range(self.dim)]

    @property
    def jacfac(self):
        return self.jacfac_

    @property
    def as_dict(self):
        d = {}
        d['dim_'] = self.dim
        d['a_'] = list(self.a_)
        d['b_'] = list(self.b_)
        d['sa_'] = list(self.sa_)
        d['sb_'] = list(self.sb_)
        d['pnames_'] = self.pnames
        return d

    def scale(self, x):
        """
        Scale the point x from the observed range _Xmin, _Xmax to the interval _interval
        (newmax-newmin)/(oldmax-oldmin)*(x-oldmin)+newmin
        """
        return self.scaleTerm_*(x - self.a_) + self.sa_

    def unscale(self, x):
        """
        Convert a point from the scaled world back to the real world.
        """
        return self.a_ + (x-self.sa_)/self.scaleTerm_

    @property
    def pnames(self): return self.pnames_

    def __eq__(self, other):
        return (self.dim == other.dim) and np.all(np.isclose(self.sa_, other.sa_)) and np.all(np.isclose(self.scaleTerm_, other.scaleTerm_)) and np.all(np.isclose(self.a_, other.a_))

    def __repr__(self):
        s = "{} dimensional space\n".format(self.dim)
        for d in range(self.dim):
            if self.pnames is not None:
                s += "{} ".format(self.pnames[d])

            s += "[{} {}] ".format(self.a_[d], self.b_[d])
            s += "[{} {}]\n".format(self.sa_[d], self.sb_[d])

        return s

    def mkSubSpace(self, dims: list[int]):
        """
        Return a Space using only the dimensions specified in dims.
        Useful when fixing parameters.
        """
        newdim = len(dims)
        newnames = [self.pnames[d] for d in dims] if not self.pnames is None else None
        return Space(newdim, [self.a_[d] for d in dims], [self.b_[d] for d in dims], pnames=newnames)

    @staticmethod
    def sample_main(b_min, b_max, npoints: int, method="uniform", seed=None):
        """
        Sample npoints self.dim_-dimensional pointsrandomly from within this space's bounds.
        Provided methods: uniform,lhs,sobol
        With seed=None, it is guaranteed that successive calls yield different points.
        """

        import numpy as np
        from scipy.stats import qmc
        if method== "uniform":
            if seed is not None:
                np.random.seed(seed)
            points = np.random.uniform(low=b_min, high=b_max,size=(npoints, len(b_min)))
        elif method== "lhs":
            sampler = qmc.LatinHypercube(len(b_min), seed=seed)
            sample = sampler.random(n=npoints)
            points = qmc.scale(sample, b_min, b_max)
        elif method== "sobol":
            sampler = qmc.Sobol(len(b_min), seed=seed)
            sample = sampler.random(n=npoints)
            points = qmc.scale(sample, b_min, b_max)
        else:
            raise Exception("Requested sampling method {} not implemented".format(method))

        return points

    def sample(self, npoints: int, method="uniform", seed=None):
        return self.sample_main(b_min=self.a_,b_max=self.b_,npoints=npoints,method=method,seed=seed)

    def sample_scaled(self, npoints: int, method="uniform", seed=None):
        return self.sample_main(b_min=self.sa_,b_max=self.sb_,npoints=npoints,method=method,seed=seed)

