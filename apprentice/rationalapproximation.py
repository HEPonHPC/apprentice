import numpy as np

#https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
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

from sklearn.base import BaseEstimator, RegressorMixin
class RationalApproximation(BaseEstimator, RegressorMixin):
    def __init__(self, *args, **kwargs):
        """
        Multivariate rational approximation f(x)_mn =  g(x)_m/h(x)_n

        kwargs:
            fname --- to read in previously calculated Pade approximation

            X     --- anchor points
            Y     --- function values
            order --- tuple (m,n) m being the order of the numerator polynomial --- if omitted: auto
        """
        import os
        if len(args) == 0:
            pass
        else:
            if type(args[0])==dict:
                self.mkFromDict(args[0])
            elif type(args[0]) == str:
                self.mkFromJSON(args[0])
            else:
                self._X   = np.array(args[0], dtype=np.float64)
                self._dim = X[0].shape[0]
                self._Y   = np.array(args[1], dtype=np.float64)
                self.mkFromData(kwargs=kwargs)

    @property
    def dim(self): return self._dim
    @property
    def M(self): return self._M
    @property
    def N(self): return self._N
    @property
    def m(self): return self._m
    @property
    def n(self): return self._n

    def mkFromJSON(self, fname):
        import json
        d = json.load(open(fname))
        self.mkFromDict(d)

    def mkFromDict(self, pdict):
        self._acoeff     = np.array(pdict["acoeff"])
        self._bcoeff     = np.array(pdict["bcoeff"])
        self.setStructures(pdict["m"], pdict["n"])

    def mkFromData(self, kwargs):
        """
        Calculate the Pade approximation
        """
        order=kwargs["order"]
        debug=kwargs["debug"] if kwargs.get("debug") is not None else False
        strategy=int(kwargs["strategy"]) if kwargs.get("strategy") is not None else 2
        self.fit(order[0], order[1], debug=debug, strategy=strategy)

    def setStructures(self, m, n):
        from apprentice import monomial
        self._struct_g = monomial.monomialStructure(self.dim, m)
        self._struct_h = monomial.monomialStructure(self.dim, n)
        from apprentice import tools
        self._M        = tools.numCoeffsPoly(self.dim, m)
        self._N        = tools.numCoeffsPoly(self.dim, n)
        self._m        = m
        self._n        = n
        self._K=m+n+1

    def mkVandermonde(self, params, order):
        """
        Construct the Vandermonde matrix.
        """
        from apprentice import tools
        PM = np.zeros((len(params), tools.numCoeffsPoly(self.dim, order)), dtype=np.float64)

        from apprentice import monomial
        s = monomial.monomialStructure(self.dim, order)
        for a, p in enumerate(params): PM[a]=monomial.recurrence(p, s)

        return PM

    # @timeit
    def coeffSolve(self, VM, VN):
        """
        This does the solving for the numerator and denominator coefficients
        following Anthony's recipe.
        """
        Fmatrix=np.diag(self._Y)
        # rcond changes from 1.13 to 1.14
        rcond = -1 if np.version.version < "1.15" else None
        MM, res, rank, s  = np.linalg.lstsq(VM, Fmatrix, rcond=rcond)
        Zmatrix = MM.dot(VN)
        U, S, V = np.linalg.svd(VM.dot(Zmatrix) - Fmatrix.dot(VN))
        self._bcoeff = V[-1]
        self._acoeff = Zmatrix.dot(self._bcoeff)

    # @timeit
    def coeffSolve2(self, VM, VN):
        """
        This does the solving for the numerator and denominator coefficients
        following Steve's recipe.
        """
        Feps = - (VN.T * self._Y).T
        # the full left side of the equation
        y = np.hstack([ VM, Feps[:,1:self._Y.size ] ])
        U, S, V = np.linalg.svd(y)
        # manipulations to solve for the coefficients
        # Given A = U Sigma VT, for A x = b, x = V Sigma^-1 UT b
        tmp1 = np.transpose( U ).dot( np.transpose( self._Y ))[0:S.size]
        Sinv = np.linalg.inv( np.diag(S) )
        x = np.transpose(V).dot( Sinv.dot(tmp1) )
        self._acoeff = x[0:self._M]
        self._bcoeff = np.concatenate([np.array([1.00]),x[self._M:self._M+self._N+1]])

    def fit(self, m, n, debug=False, strategy=2):
        """
        Do everything
        """
        # Set M, N, K, polynomial structures
        # n_required=self.numCoeffs(self.dim, m+n+1)
        from apprentice import tools
        n_required = tools.numCoeffsRapp(self.dim, (m,n))
        if n_required > self._Y.shape[0]:
            raise Exception("Not enough inputs: got %i but require %i to do m=%i n=%i"%(n_required, Fmatrix.shape[0], m,n))


        self.setStructures(m,n)
        if debug:
            print("structure setting took {} seconds".format(te-ts))

        VanderMonde=self.mkVandermonde(self._X, self._K)
        VM = VanderMonde[:, 0:(self._M)]
        VN = VanderMonde[:, 0:(self._N)]
        if debug:
            print("VM took {} seconds".format(te-ts))

        if   strategy==1: self.coeffSolve( VM, VN)
        elif strategy==2: self.coeffSolve2(VM, VN)
        else: raise Exception("fit() strategy %i not implemented"%strategy)

    def denom(self, X):
        """
        Evaluation of the denom poly at X.
        """
        from apprentice import monomial
        lv_h = np.array(monomial.recurrence(X, self._struct_h))
        h=self._bcoeff.dot(lv_h)
        return h

    def numer(self, X):
        """
        Evaluation of the numer poly at X.
        """
        from apprentice import monomial
        lv_g = np.array(monomial.recurrence(X, self._struct_g))
        g=self._acoeff.dot(lv_g)
        return g

    def predict(self, X):
        """
        Return the prediction of the RationalApproximation at X.
        """
        den = self.denom(X)
        if den==0: return 0 # TODO why is this here?
        else:
            return self.numer(X)/self.denom(X)

    def __call__(self, X):
        """
        Operator version of predict.
        """
        return self.predict(X)

    @property
    def asDict(self):
        """
        Store all info in dict as basic python objects suitable for JSON
        """
        d={}
        d["m"]    = self.m
        d["n"]    = self.n
        d["acoeff"] = list(self._acoeff)
        d["bcoeff"] = list(self._bcoeff)
        return d

    def save(self, fname):
        import json
        with open(fname, "w") as f:
            json.dump(self.asDict, f)

if __name__=="__main__":

    import sys

    def mkTestData(NX, dim=1):
        def anthonyFunc(x):
            return (10*x)/(x**3 - 4* x + 5)
        NR = 1
        np.random.seed(555)
        X = np.random.rand(NX, dim)
        Y = np.array([anthonyFunc(*x) for x in X])
        return X, Y

    X, Y = mkTestData(500)
    r=RationalApproximation(X,Y, order=(1,3))

    import pylab
    pylab.plot(X, Y, marker="*", linestyle="none", label="Data")
    TX = sorted(X)
    YW = [r(p) for p in TX]

    pylab.plot(TX, YW, label="Rational approx m={} n={}".format(1,3))
    pylab.legend()
    pylab.xlabel("x")
    pylab.ylabel("f(x)")
    pylab.savefig("demo.pdf")

    sys.exit(0)
