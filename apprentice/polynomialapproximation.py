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
class PolynomialApproximation(BaseEstimator, RegressorMixin):
    def __init__(self, X=None, Y=None, order=2, fname=None, initDict=None, strategy=2):
        """
        Multivariate polynomial approximation

        kwargs:
            fname --- to read in previously calculated Pade approximation

            X        --- anchor points
            Y        --- function values
            fname    --- JSON file to read pre-calculated info from
            initDict --- dict to read pre-calculated info from
            order    --- int being the order of the polynomial
        """
        if initDict is not None:
            self.mkFromDict(initDict)
        elif fname is not None:
            self.mkFromJSON(fname)
        elif X is not None and Y is not None:
            self._m=order
            self._X   = np.array(X, dtype=np.float64)
            self._dim = self._X[0].shape[0]
            self._Y   = np.array(Y, dtype=np.float64)
            self._trainingsize=len(X)
            self.fit(strategy=strategy)
        else:
            raise Exception("Constructor not called correctly, use either fname, initDict or X and Y")

    @property
    def dim(self): return self._dim
    @property
    def trainingsize(self): return self._trainingsize
    @property
    def M(self): return self._M
    @property
    def m(self): return self._m

    def setStructures(self):
        from apprentice import monomial
        self._struct_p = monomial.monomialStructure(self.dim, self.m)
        from apprentice import tools
        self._M        = tools.numCoeffsPoly(self.dim, self.m)

    # @timeit
    def coeffSolve(self, VM):
        """
        SVD solve coefficients.
        """
        # TODO check for singular values below threshold and raise Exception
        U, S, V = np.linalg.svd(VM)
        # SM dixit: manipulations to solve for the coefficients
        # Given A = U Sigma VT, for A x = b, x = V Sigma^-1 UT b
        temp = np.dot(U.T, self._Y.T)[0:S.size]
        self._pcoeff = np.dot(V.T, 1./S * temp)

    # @timeit
    def coeffSolve2(self, VM):
        """
        Least square solve coefficients.
        """
        rcond = -1 if np.version.version < "1.15" else None
        x, res, rank, s  = np.linalg.lstsq(VM, self._Y, rcond=None)
        self._pcoeff = x

    def fit(self, **kwargs):
        """
        Do everything
        """
        from apprentice import tools
        n_required = tools.numCoeffsPoly(self.dim, self.m)
        if n_required > self._Y.shape[0]:
            raise Exception("Not enough inputs: got %i but require %i to do m=%i"%(n_required, self._Y.shape, self.m))

        self.setStructures()

        from apprentice import monomial
        VM = monomial.vandermonde(self._X, self.m)
        strategy=kwargs["strategy"] if kwargs.get("strategy") is not None else 1
        if   strategy==1: self.coeffSolve( VM)
        elif strategy==2: self.coeffSolve2(VM)
        # NOTE, strat 1 is faster for smaller problems (Npoints < 250)
        else: raise Exception("fit() strategy %i not implemented"%strategy)

    def predict(self, X):
        """
        Evaluation of the numer poly at X.
        """
        X=np.array(X)
        from apprentice import monomial
        rec_p = np.array(monomial.recurrence(X, self._struct_p))
        p=self._pcoeff.dot(rec_p)
        return p

    def __call__(self, X):
        """
        Operator version of predict.
        """
        return self.predict(X)

    def __repr__(self):
        """
        Print-friendly representation.
        """
        return "<PolynomialApproximation dim:{} order:{}>".format(self.dim, self.m)

    @property
    def asDict(self):
        """
        Store all info in dict as basic python objects suitable for JSON
        """
        d={}
        d["dim"]    = self.dim
        d["trainingsize"] = self.trainingsize
        d["m"]      = self.m
        d["pcoeff"] = list(self._pcoeff)
        return d

    def save(self, fname):
        import json
        with open(fname, "w") as f:
            json.dump(self.asDict, f)

    def mkFromJSON(self, fname):
        import json
        d = json.load(open(fname))
        self.mkFromDict(d)

    def mkFromDict(self, pdict):
        self._pcoeff     = np.array(pdict["pcoeff"])
        self._m = int(pdict["m"])
        self._dim=int(pdict["dim"])
        try:
            self._trainingsize = int(pdict["trainingsize"])
        except:
            pass
        self.setStructures()


if __name__=="__main__":

    import sys

    def mkTestData(NX, dim=1):
        def anthonyFunc(x):
            return x**5-x**2
        NR = 1
        np.random.seed(555)
        X = np.random.rand(NX, dim)
        Y = np.array([anthonyFunc(*x) for x in X])
        return X, Y

    X, Y = mkTestData(200)
    # from apprentice import scaler
    # S=scaler.Scaler(X)


    # r=PolynomialApproximation(X=S.scaledPoints, Y=Y, order=3)

    # r.save("testpoly.json")
    # r=PolynomialApproximation(fname="testpoly.json")

    # import pylab
    # pylab.plot(S.scaledPoints, Y, marker="*", linestyle="none", label="Data")
    # TX = sorted(S.scaledPoints)
    # YW = [r(p) for p in TX]

    # pylab.plot(TX, YW, label="Polynomial approx m={}".format(3))
    # pylab.legend()
    # pylab.xlabel("x")
    # pylab.ylabel("f(x)")
    # pylab.savefig("demopoly.pdf")

    r =PolynomialApproximation(X=X, Y=Y, order=3, strategy=1)
    r2=PolynomialApproximation(X=X, Y=Y, order=3, strategy=2)

    import pylab
    pylab.clf()
    pylab.plot(X, Y, marker="*", linestyle="none", label="Data")
    TX = sorted(X)
    YW  = [r(p)  for p in TX]
    YW2 = [r2(p) for p in TX]

    pylab.plot(TX, YW, label="Polynomial approx m={} strategy 1".format(3))
    pylab.plot(TX, YW, label="Polynomial approx m={} strategy 2".format(3))
    pylab.legend()
    pylab.xlabel("x")
    pylab.ylabel("f(x)")
    pylab.savefig("demopolynoscale.pdf")
    sys.exit(0)
