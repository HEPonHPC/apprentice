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
                self._trainingsize=len(args[0])
                self.fit()

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

    def fit(self):
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
        self.coeffSolve(VM)

    def predict(self, X):
        """
        Evaluation of the numer poly at X.
        """
        from apprentice import monomial
        rec_p = np.array(monomial.recurrence(X, self._struct_p))
        p=self._pcoeff.dot(rec_p)
        return p

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
            return x**3-x**2
        NR = 1
        np.random.seed(555)
        X = np.random.rand(NX, dim)
        Y = np.array([anthonyFunc(*x) for x in X])
        return X, Y

    X, Y = mkTestData(20)
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
