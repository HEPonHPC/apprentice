import numpy as np
import apprentice
from apprentice.surrogatemodel import SurrogateModel
from apprentice.space import Space
from apprentice.tools import gradientRecursion
import pprint

# https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
def timeit(method):
    import time
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed

class PolynomialApproximation(SurrogateModel):
    """
    Polynomial approximation surrogate model
    """
    __allowed = ("m_", "m",
                 "pcoeff_","pcoeff",
                 "training_size_","training_size_",
                 "strategy_","strategy",
                 'cov_'"cov",
                 "compute_cov_",'compute_cov'
                 )

    def __init__(self, dim, fnspace=None, **kwargs: dict):
        """

        Polynomial approximation surrogate model construction function

        :param dim: parameter dimension
        :type dim: int
        :param fnspace: function space object
        :type fnspace: apprentice.space.Space

        """
        super().__init__(dim, fnspace)
        for k, v in kwargs.items():
            if k in ['m','training_size',"pcoeff",'strategy','cov','compute_cov']:
                k+="_"
            elif k in ['pnames','pnames_','scale_min','scale_max','scale_min_', 'scale_max_']: continue
            assert (k in self.__class__.__allowed)
            setattr(self,k, v)

        self.set_structures()

    # def __init__(self, X=None, Y=None, order=2, fname=None, initDict=None, strategy=2, scale_min=-1, scale_max=1, pnames=None, set_structures=True, computecov=False):
    #     """
    #     Multivariate polynomial approximation
    #
    #     kwargs:
    #         fname --- to read in previously calculated Pade approximation
    #
    #         X        --- anchor points
    #         Y        --- function values
    #         fname    --- JSON file to read pre-calculated info from
    #         initDict --- dict to read pre-calculated info from
    #         order    --- int being the order of the polynomial
    #     """
    #     self._vmin=None
    #     self._vmax=None
    #     self._xmin=None
    #     self._xmax=None
    #     if initDict is not None:
    #         self.mkFromDict(initDict, set_structures=set_structures)
    #     elif fname is not None:
    #         self.mkFromJSON(fname, set_structures=set_structures)
    #     elif X is not None and Y is not None:
    #         self._m=order
    #         self._scaler = apprentice.Scaler(np.atleast_2d(np.array(X, dtype=np.float64)), a=scale_min, b=scale_max, pnames=pnames)
    #         self._X   = self._scaler.scaledPoints
    #         self._dim = self._X[0].shape[0]
    #         self._Y   = np.array(Y, dtype=np.float64)
    #         self._trainingsize=len(X)
    #         if self._dim==1: self.recurrence=apprentice.monomial.recurrence1D
    #         else           : self.recurrence=apprentice.monomial.recurrence
    #         self.fit(strategy=strategy, computecov=computecov)
    #     else:
    #         raise Exception("Constructor not called correctly, use either fname, initDict or X and Y")

    @property
    def dim(self):
        """

        Get the parameter dimension value

        :return: parameter dimension
        :rtype: int

        """
        return self.fnspace.dim

    @property
    def pnames(self):
        """

        Get names of the parameter dimension names

        :return: array of parameter dimension string names
        :rtype: list

        """
        return self.fnspace.pnames

    @property
    def order_numerator(self):
        """

        Get the numerator order

        :return: numerator order
        :rtype: int

        """
        if hasattr(self, 'm_'):
            return self.m_
        return 1

    @property
    def fit_strategy(self):
        """

        Get the fit strategy

        :return: fit strategy
        :rtype: int

        """
        if hasattr(self, 'strategy_'):
            return self.strategy_
        return 1

    @property
    def to_compute_covariance(self):
        """

        Get the choice of whether covariance should be computed

        :return: true if covariance should be computed
        :rtype: bool

        """
        if hasattr(self, 'compute_cov_'):
            return self.compute_cov_
        return False

    @property
    def coeff_numerator(self):
        """

        Get the numerator coefficients. The order of coefficients is as in
        https://people.math.sc.edu/Burkardt/py_src/polynomial/polynomial.html

        :return: list of numerator coefficients.
        :rtype: list

        """
        if hasattr(self, 'pcoeff_'):
            return self.pcoeff_
        raise Exception("Numerator coeffecients cannot be found. Perform a fit first")

    def set_structures(self):
        """

        Set monomial structures into self. The order is as in
        https://people.math.sc.edu/Burkardt/py_src/polynomial/polynomial.html

        """
        m = self.order_numerator
        self.struct_p_ = apprentice.monomialStructure(self.dim, m)
        from apprentice import tools
        self.M_ = tools.numCoeffsPoly(self.dim, m)
        self.nnz_ = self.struct_p_ > 0

    # @timeit
    def coeff_solve(self, VM, Y):
        """

        Solve coefficients using Singular Value Decomposition (SVD)

        :param VM: Vandermonde matrix
        :type VM: np.array
        :param Y: an array of size :math:`N_p` and it is the y data values to fit where
            :math:`N_p` is the the number of data points
        :type Y: np.array

        """

        # TODO check for singular values below threshold and raise Exception
        U, S, V = np.linalg.svd(VM)
        # SM dixit: manipulations to solve for the coefficients
        # Given A = U Sigma VT, for A x = b, x = V Sigma^-1 UT b
        temp = np.dot(U.T, Y.T)[0:S.size]
        self.pcoeff_ = np.dot(V.T, 1. / S * temp)

    # @timeit
    def coeff_solve2(self, VM, Y):
        """

        Solve coefficients using least squares regression

        :param VM: Vandermonde matrix
        :type VM: np.array
        :param Y: an array of size :math:`N_p` and it is the y data values to fit where
            :math:`N_p` is the the number of data points
        :type Y: np.array

        """
        rcond = -1 if np.version.version < "1.15" else None
        x, res, rank, s = np.linalg.lstsq(VM, Y, rcond=rcond)
        self.pcoeff_ = x

    def fit(self, X, Y):
        """

        Surrogate model fitting method.

        :param X: a 2-D array of size :math:`dim \times N_p` and it is the x data values to fit where
            :math:`dim` is the parameter dimension and :math:`N_p` is the the number of data points
        :type X: list
        :param Y: an array of size :math:`N_p` and it is the y data values to fit where
            :math:`N_p` is the the number of data points
        :type Y: np.array

        """
        m = self.order_numerator
        from apprentice import tools
        n_required = tools.numCoeffsPoly(self.dim, m)
        if n_required > Y.shape[0]:
            raise Exception(
                "Not enough inputs: got %i but require %i to do m=%i" % (Y.shape[0], n_required, m))

        self.set_structures()
        X = self.fnspace.scale(X)
        from apprentice import monomial
        VM = monomial.vandermonde(X, m)
        strategy = self.fit_strategy
        if self.to_compute_covariance is not False:
            self.cov_ = np.linalg.inv(2 * VM.T @ VM)
        if strategy == 1:
            self.coeff_solve(VM,Y)
        elif strategy == 2:
            self.coeff_solve2(VM,Y)
        # NOTE, strat 1 is faster for smaller problems (Npoints < 250)
        else:
            raise Exception("fit() strategy %i not implemented" % strategy)

    def f_x_slow(self, x):
        """

        Calculate the surrogate model at a new point. This is a slower version.

        :param x: a new x point, an araay of size :math:`dim` where :math:`dim` is the parameter dimension
        :type x: list
        :return: surrogate model value at the new point
        :rtype: float

        """
        x = self.fnspace.scale(np.array(x))
        if self.dim==1: recurrence=apprentice.monomial.recurrence1D
        else           :recurrence=apprentice.monomial.recurrence
        rec_p = np.array(recurrence(x, self.struct_p_))
        return self.coeff_numerator.dot(rec_p)

    def f_x(self, x):
        """

        Calculate the surrogate model at a new point.

        :param x: a new x point, an araay of size :math:`dim` where :math:`dim` is the parameter dimension
        :type x: list
        :return: surrogate model value at the new point
        :rtype: float

        """
        x = self.fnspace.scale(np.array(x))
        if self.dim==1: rec_p=apprentice.monomial.recurrence1D(x, self.struct_p_)
        else           :rec_p=apprentice.monomial.recurrence2(x, self.struct_p_, self.nnz_)
#        if type(self.coeff_numerator) == 'list' :
        if isinstance(self.coeff_numerator,list):
            return np.array(self.coeff_numerator).dot(rec_p)
            #return self.np.array(coeff_numerator).dot(rec_p)
        return self.coeff_numerator.dot(rec_p)

    def f_X(self, X):
        """

        Calculate the surrogate model at a multiple date points.

        :param X: multiple x point, an araay of size :math:`dim \times n`
            where :math:`dim` is the parameter dimension and :math:`n` is the number of new data points
        :type X: list
        :return: surrogate model value at the new points an araay of size :math:`n`
            where :math:`n` is the number of new data points
        :rtype: list

        """
        return [self.f_x(x) for x in X]

    def __repr__(self):
        """

        Print-friendly representation.

        """
        return "<PolynomialApproximation dim:{} order:{}>".format(self.dim, self.order_numerator)

    @property
    def as_dict(self):
        """

        Get the polynomial approximation fit as a dictionary

        :return: polynomial approximation fit dictionary
        :rtype: dict

        """
        d = {}
        d["m"] = self.order_numerator
        d["training_size"] = self.training_size
        d["strategy"] = self.fit_strategy
        d["pcoeff"] = list(self.coeff_numerator)
        d["fnspace"] = self.fnspace.as_dict
        d['compute_cov'] = self.to_compute_covariance
        if self.to_compute_covariance is not False:
            d['cov'] = self.cov_
        # if hasattr(self,'vmin') and self.vmin is not None: d["vmin"] = self.vmin
        # if hasattr(self,'vmax') and self.vmax is not None: d["vmax"] = self.vmax
        # if hasattr(self,'xmin') and self.xmin is not None: d["xmin"] = self.xmin
        # if hasattr(self,'xmax') and self.xmax is not None: d["xmax"] = self.xmax
        return d

    def save(self, fname):
        """

        Save the polynomial approximation fit into a file

        :param fname: file path to save polynomial approximation fit
        :type fname: str

        """
        import json
        with open(fname, "w") as f:
            json.dump(self.as_dict, f,indent=4)

    @classmethod
    def from_data_structure(cls,data_structure,**kwargs):
        """

        A class method to construct surrogate model from data structure.

        :param data_structure: previously fit surrogate model saved as a data structure
        :type data_structure: dict

        """
        if not isinstance(data_structure, dict):
            raise Exception("data_structure has to be a dictionary")
        dim = data_structure['fnspace']['dim_']
        a = data_structure['fnspace']['a_']
        b = data_structure['fnspace']['b_']
        sa = data_structure['fnspace']['sa_']
        sb = data_structure['fnspace']['sb_']
        pnames = data_structure['fnspace']['pnames_']
        data_structure.pop('fnspace')
        fnspace = Space(dim,
                        a=a,
                        b=b,
                        sa=sa,
                        sb=sb,
                        pnames=pnames)

        return cls(dim,fnspace,**data_structure)

    @classmethod
    def from_file(cls, filepath,**kwargs):
        """

        A class method to construct surrogate model from data structure saved in a file.

        :param filepath: file path of a previously fit surrogate model saved as a data structure
        :type filepath: str

        """
        import json
        with open(filepath,'r') as f:
            d = json.load(f)
        return cls.from_data_structure(d)

    # def fmin(self, nsamples=1, nrestart=1, use_grad=False):
    #     return apprentice.tools.extreme(self, nsamples, nrestart, use_grad, mode="min")
    #
    # def fmax(self, nsamples=1, nrestart=1, use_grad=False):
    #     return apprentice.tools.extreme(self, nsamples, nrestart, use_grad, mode="max")

    @property
    def coeff_norm(self):
        """

        Get 1-norm of the polynomial approximation fit coefficients

        :return: 1-norm of the polynomial approximation fit coefficients
        :rtype: float

        """
        nrm = 0
        for p in self.coeff_numerator:
            nrm += abs(p)
        return nrm

    @property
    def coeff2_norm(self):
        """

        Get 2-norm of the polynomial approximation fit coefficients

        :return: 2-norm of the polynomial approximation fit coefficients
        :rtype: float

        """
        nrm = 0
        for p in self.coeff_numerator:
            nrm += p * p
        return np.sqrt(nrm)

    def gradient(self, X):
        """

        Get gradient of the polynomial approximation

        :param X: a new point, an araay of size :math:`dim` where :math:`dim` is the parameter dimension
        :type X: list
        :return: gradient of the polynomial approximation at a new point
        :rtype: list

        """

        struct = np.array(self.struct_p_, dtype=float)
        X = self.function_space.scale(np.array(X))

        if self.dim == 1:
            struct[1:] = self.function_space.jacfac[0] * struct[1:] * np.power(X, struct[1:] - 1)
            return np.dot(np.atleast_2d(struct), self.coeff_numerator)

        GREC = gradientRecursion(X, struct, self.function_space.jacfac)

        return np.sum(GREC * self.coeff_numerator, axis=1)

    def hessian(self, X):
        """

        Get hessian of the polynomial approximation

        :param X: a new point, an araay of size :math:`dim` where :math:`dim` is the parameter dimension
        :type X: list
        :return: hessian of the polynomial approximation at a new point
        :rtype: list

        """
        X = self.function_space.scale(np.array(X))
        S = self.struct_p_

        HH = np.ones((self.dim, self.dim, len(S)), dtype=np.float64)
        EE = np.full((self.dim, self.dim, len(S), self.dim), S, dtype=np.int32)

        for numx in range(self.dim):
            for numy in range(self.dim):
                if numx == numy:
                    HH[numx][numy] = S[:, numx] * (S[:, numx] - 1)
                else:
                    HH[numx][numy] = S[:, numx] * S[:, numy]
                EE[numx][numy][:, numx] -= 1
                EE[numx][numy][:, numy] -= 1

        NONZ = np.empty((self.dim, self.dim), dtype=tuple)
        for numx in range(self.dim):
            for numy in range(self.dim):
                NONZ[numx][numy] = np.where(HH[numx][numy] > 0)

        JF = self.function_space.jacfac
        for numx in range(self.dim):
            for numy in range(self.dim):
                HH[numx][numy][NONZ[numx][numy]] *= (JF[numx] * JF[numy])

        HESS = np.empty((self.dim, self.dim), dtype=np.float64)
        for numx in range(self.dim):
            for numy in range(self.dim):
                if numy >= numx:
                    HESS[numx][numy] = np.sum(HH[numx][numy][NONZ[numx][numy]] * np.prod(
                        np.power(X, EE[numx][numy][NONZ[numx][numy]]), axis=1) * self.coeff_numerator[
                                                  NONZ[numx][numy]])
                else:
                    HESS[numx][numy] = HESS[numy][numx]

        return HESS

    # def __call__(self, x):
    #     super.__call__(x)
    # def wraps(self, v):
    #     dec = True
    #     if self.vmin is not None and self.vmax is not None:
    #         if self.vmin > v or self.vmax < v: dec = False
    #     return dec


# if __name__ == "__main__":
#
#     import sys
#
#
#     def mkTestData(NX, dim=1):
#         def anthonyFunc(x):
#             return x ** 3 - x ** 2
#
#         NR = 1
#         np.random.seed(555)
#         X = 10 * np.random.rand(NX, dim) - 1
#         Y = np.array([anthonyFunc(*x) for x in X])
#         return X, Y
#
#
#     X, Y = mkTestData(200)
#
#     p1 = PolynomialApproximation(X=X, Y=Y, order=3, strategy=1)
#     p2 = PolynomialApproximation(X=X, Y=Y, order=3, strategy=2)
#     p2.save("polytest.json")
#     p3 = PolynomialApproximation(fname="polytest.json")
#
#     import pylab
#
#     pylab.clf()
#     pylab.plot(X, Y, marker="*", linestyle="none", label="Data")
#     TX = sorted(X)
#     Y1 = [p1(p) for p in TX]
#     Y2 = [p2(p) for p in TX]
#     Y3 = [p3(p) for p in TX]
#
#     try:
#         import autograd.numpy as np
#         from autograd import hessian, grad
#
#         have_autograd = True
#     except:
#         print("No autograd...")
#         have_autograd = False
#
#
#     def f(x):
#         return 2 * x ** 2 + 1
#
#
#     def fp(x):
#         return 4 * x
#
#
#     X = np.linspace(0, 10, 11)
#     # X=np.linspace(-1,1,11)
#     Y = f(X)
#     pp = PolynomialApproximation(X=[[x] for x in X], Y=Y, order=2, strategy=1)
#     pylab.clf()
#     pylab.plot(X, Y, marker="*", linestyle="none", label="Data")
#
#     pylab.plot(X, [pp(x) for x in X], label="Polynomial approx m={} strategy 1".format(3))
#     pylab.plot(X, pp.predictArray(X), label="Polynomial approx array  m={} strategy 1".format(3))
#
#     myg = [pp.gradient(x) for x in X]
#
#     if have_autograd:
#         g = grad(pp)
#         G = [g(x) for x in X]
#         pylab.plot(X, G, label="auto gradient")
#     FP = [fp(x) for x in X]
#
#     pylab.plot(X, FP, marker="s", linestyle="none", label="analytic gradient")
#     pylab.plot(X, myg, label="manual gradient")
#     pylab.legend()
#
#     pylab.show()
#
#     pylab.plot(TX, Y1, label="Polynomial approx m={} strategy 1".format(3))
#     pylab.plot(TX, Y2, label="Polynomial approx m={} strategy 2".format(3))
#     pylab.plot(TX, Y3, "m--", label="Restored polynomial approx m={} strategy 2".format(3))
#     pylab.legend()
#     pylab.xlabel("x")
#     pylab.ylabel("f(x)")
#     pylab.savefig("demopoly.pdf")
#     sys.exit(0)
