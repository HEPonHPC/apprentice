import numpy as np
import apprentice

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
    def __init__(self, X=None, Y=None, order=2, fname=None, initDict=None, strategy=2, xmin=-1, xmax=1, scale_min=-1, scale_max=1, pnames=None):
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
            self._scaler = apprentice.Scaler(np.atleast_2d(np.array(X, dtype=np.float64)), a=scale_min, b=scale_max, pnames=pnames)
            self._X   = self._scaler.scaledPoints
            self._X   = self._scaler.scaledPoints
            self._dim = self._X[0].shape[0]
            self._Y   = np.array(Y, dtype=np.float64)
            self._trainingsize=len(X)
            if self._dim==1: self.recurrence=apprentice.monomial.recurrence1D
            else           : self.recurrence=apprentice.monomial.recurrence
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
        self._struct_p = apprentice.monomialStructure(self.dim, self.m)
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
        X=self._scaler.scale(np.array(X))
        rec_p = np.array(self.recurrence(X, self._struct_p))
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
        d["scaler"] = self._scaler.asDict
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
        self._scaler = apprentice.Scaler(pdict["scaler"])
        if self._dim==1: self.recurrence=apprentice.monomial.recurrence1D
        else           : self.recurrence=apprentice.monomial.recurrence
        try:
            self._trainingsize = int(pdict["trainingsize"])
        except:
            pass
        self.setStructures()

    def fmin(self, multistart=None):
        from scipy import optimize
        if multistart is None:
            fmin = optimize.minimize(lambda x:self.predict(x), self._scaler.center, bounds=self._scaler.box)
            return fmin["fun"]
        else:
            _P = self._scaler.drawSamples(multistart)
            _fmin = [optimize.minimize(lambda x:self.predict(x), pstart, bounds=self._scaler.box)["fun"] for pstart in _P]
            return min(_fmin)

    def fmax(self, multistart=None):
        from scipy import optimize
        if multistart is None:
            fmax = optimize.minimize(lambda x:-self.predict(x), self._scaler.center, bounds=self._scaler.box)
            return -fmax["fun"]
        else:
            _P = self._scaler.drawSamples(multistart)
            _fmax = [optimize.minimize(lambda x:-self.predict(x), pstart, bounds=self._scaler.box)["fun"] for pstart in _P]
            return -min(_fmax)

    @property
    def coeffNorm(self):
        nrm = 0
        for p in self._pcoeff:
            nrm+= abs(p)
        return nrm

    @property
    def coeff2Norm(self):
        nrm = 0
        for p in self._pcoeff:
            nrm+= p*p
        return np.sqrt(nrm)

if __name__=="__main__":

    import sys

    def mkTestData(NX, dim=1):
        def anthonyFunc(x):
            return x**3-x**2
        NR = 1
        np.random.seed(555)
        X = 10*np.random.rand(NX, dim) -1
        Y = np.array([anthonyFunc(*x) for x in X])
        return X, Y

    X, Y = mkTestData(200)

    p1 =PolynomialApproximation(X=X, Y=Y, order=3, strategy=1)
    p2 =PolynomialApproximation(X=X, Y=Y, order=3, strategy=2)
    p2.save("polytest.json")
    p3 =PolynomialApproximation(fname="polytest.json")

    import pylab
    pylab.clf()
    pylab.plot(X, Y, marker="*", linestyle="none", label="Data")
    TX = sorted(X)
    Y1  = [p1(p) for p in TX]
    Y2 =  [p2(p) for p in TX]
    Y3 =  [p3(p) for p in TX]

    pylab.plot(TX, Y1, label="Polynomial approx m={} strategy 1".format(3))
    pylab.plot(TX, Y2, label="Polynomial approx m={} strategy 2".format(3))
    pylab.plot(TX, Y3, "m--", label="Restored polynomial approx m={} strategy 2".format(3))
    pylab.legend()
    pylab.xlabel("x")
    pylab.ylabel("f(x)")
    pylab.savefig("demopoly.pdf")
    sys.exit(0)
