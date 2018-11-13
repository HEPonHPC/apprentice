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
    def __init__(self, *args, **kwargs):
        """
        Multivariate polynomial approximation

        kwargs:
            fname --- to read in previously calculated Pade approximation

            X     --- anchor points
            Y     --- function values
            order --- int being the order of the polynomial
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
                self._m=kwargs["order"]
                self._X   = np.array(args[0], dtype=np.float64)
                self._dim = self._X[0].shape[0]
                self._Y   = np.array(args[1], dtype=np.float64)
                self.fit()

    @property
    def dim(self): return self._dim
    @property
    def M(self): return self._M
    @property
    def m(self): return self._m

    def mkFromJSON(self, fname):
        import json
        d = json.load(open(fname))
        self.mkFromDict(d)

    def mkFromDict(self, pdict):
        self._acoeff     = np.array(pdict["acoeff"])
        self._m = int(pdict["m"])
        self._dim=int(pdict["dim"])
        self.setStructures()

    def setStructures(self):
        from apprentice import monomial
        self._struct_g = monomial.monomialStructure(self.dim, self.m)
        from apprentice import tools
        self._M        = tools.numCoeffsPoly(self.dim, self.m)

    def mkVandermonde(self, params, order):
        """
        Construct the Vandermonde matrix.
        """
        from apprentice import tools
        PM = np.zeros((len(params), self.M), dtype=np.float64)

        from apprentice import monomial
        s = monomial.monomialStructure(self.dim, order)
        for a, p in enumerate(params): PM[a]=monomial.recurrence(p, s)

        return PM

    def coeffSolve(self, VM):
        """
        SVD solve coefficients.
        """
        U, S, V = np.linalg.svd(VM)
        # TODO check for singular values below threshold and raise Exception

        # manipulations to solve for the coefficients
        # Given A = U Sigma VT, for A x = b, x = V Sigma^-1 UT b
        tmp1 = np.transpose( U ).dot( np.transpose( self._Y ))[0:S.size]
        Sinv = np.linalg.inv( np.diag(S) )
        x = np.transpose(V).dot( Sinv.dot(tmp1) )
        self._acoeff = x[0:self._M]

    def fit(self):
        """
        Do everything
        """
        from apprentice import tools
        n_required = tools.numCoeffsPoly(self.dim, self.m)
        if n_required > self._Y.shape[0]:
            raise Exception("Not enough inputs: got %i but require %i to do m=%i"%(n_required, self._Y.shape, self.m))

        self.setStructures()

        VM = self.mkVandermonde(self._X, self.m)
        self.coeffSolve(VM)

    def predict(self, X):
        """
        Evaluation of the numer poly at X.
        """
        from apprentice import monomial
        lv_g = np.array(monomial.recurrence(X, self._struct_g))
        g=self._acoeff.dot(lv_g)
        return g

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
        d["m"]      = self.m
        d["dim"]    = self.dim
        d["acoeff"] = list(self._acoeff)
        return d

    def save(self, fname):
        import json
        with open(fname, "w") as f:
            json.dump(self.asDict, f)

if __name__=="__main__":

    import sys

    def mkTestData(NX, dim=1):
        def anthonyFunc(x):
            return x**3-x**2
        NR = 1
        np.random.seed(555)
        X = np.random.rand(NX, dim)
        Y = np.array([anthonyFunc(*x) for x in X])
        return X, Y

    X, Y = mkTestData(500)
    from apprentice import scaler
    S=scaler.Scaler(X)


    r=PolynomialApproximation(S.scaledPoints, Y, order=3)

    r.save("testpoly.json")
    r=PolynomialApproximation("testpoly.json")

    import pylab
    pylab.plot(S.scaledPoints, Y, marker="*", linestyle="none", label="Data")
    TX = sorted(S.scaledPoints)
    YW = [r(p) for p in TX]

    pylab.plot(TX, YW, label="Polynomial approx m={}".format(3))
    pylab.legend()
    pylab.xlabel("x")
    pylab.ylabel("f(x)")
    pylab.savefig("demopoly.pdf")

    r=PolynomialApproximation(X, Y, order=3)

    import pylab
    pylab.clf()
    pylab.plot(X, Y, marker="*", linestyle="none", label="Data")
    TX = sorted(X)
    YW = [r(p) for p in TX]

    pylab.plot(TX, YW, label="Polynomial approx m={}".format(3))
    pylab.legend()
    pylab.xlabel("x")
    pylab.ylabel("f(x)")
    pylab.savefig("demopolynoscale.pdf")
    sys.exit(0)
