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
class RationalApproximation(BaseEstimator, RegressorMixin):
    def __init__(self, X=None, Y=None, order=(2,1), fname=None, initDict=None, strategy=2, scale_min=-1, scale_max=1, pnames=None):
        """
        Multivariate rational approximation f(x)_mn =  g(x)_m/h(x)_n

        kwargs:
            fname --- to read in previously calculated Pade approximation

            X     --- anchor points
            Y     --- function values
            order --- tuple (m,n) m being the order of the numerator polynomial --- if omitted: auto
        """
        if initDict is not None:
            self.mkFromDict(initDict)
        elif fname is not None:
            self.mkFromJSON(fname)
        elif X is not None and Y is not None:
            self._m=order[0]
            self._n=order[1]
            self._scaler = apprentice.Scaler(np.atleast_2d(np.array(X, dtype=np.float64)), a=scale_min, b=scale_max, pnames=pnames)
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
    def N(self): return self._N
    @property
    def m(self): return self._m
    @property
    def n(self): return self._n

    def setStructures(self):
        self._struct_p = apprentice.monomialStructure(self.dim, self.m)
        self._struct_q = apprentice.monomialStructure(self.dim, self.n)
        from apprentice import tools
        self._M        = tools.numCoeffsPoly(self.dim, self.m)
        self._N        = tools.numCoeffsPoly(self.dim, self.n)
        self._K = 1 + self._m + self._n

    # @timeit
    def coeffSolve(self, VM, VN):
        """
        This does the solving for the numerator and denominator coefficients
        following Anthony's recipe.
        """
        Fmatrix=np.diag(self._Y)
        # rcond changes from 1.13 to 1.14
        rcond = -1 if np.version.version < "1.15" else None
        # Solve VM x = diag(Y)
        MM, res, rank, s  = np.linalg.lstsq(VM, Fmatrix, rcond=rcond)
        Zmatrix = MM.dot(VN)
        # Solve (VM Z - F VN)x = 0
        U, S, Vh = np.linalg.svd(VM.dot(Zmatrix) - Fmatrix.dot(VN))
        self._qcoeff = Vh[-1] # The column of (i.e. row of Vh) corresponding to the smallest singular value is the least squares solution
        self._pcoeff = Zmatrix.dot(self._qcoeff)

    # @timeit
    def coeffSolve2(self, VM, VN):
        """
        This does the solving for the numerator and denominator coefficients.
        F = p/q is reformulated as 0 = p - qF using the VanderMonde matrices.
        That defines the problem Ax = b and we solve for x in an SVD manner,
        exploiting A = U x S x V.T
        There is an additional manipulation exploiting on setting the constant
        coefficient in q to 1.
        """
        FQ = - (VN.T * self._Y).T # This is something like -F*q
        A = np.hstack([VM, FQ[:,1:]]) # Note that we leave the b0 terms out when defining A
        U, S, Vh = np.linalg.svd(A)
        # Given A = U Sigma VT, for A x = b, it follows, that: x = V Sigma^-1 UT b
        # b really is b0 * F but we explicitly choose b0 to be 1
        # The solution formula is taken from numerical recipes
        UTb = np.dot(U.T, self._Y.T)[:S.size]
        x = np.dot(Vh.T, 1./S * UTb)
        self._pcoeff = x[:self._M]
        self._qcoeff = np.concatenate([[1],x[self._M:]]) # b0 is set to 1 !!!

    # @timeit
    def coeffSolve3(self, VM, VN):
        """
        This does the solving for the numerator and denominator coefficients.
        F = p/q is reformulated as 0 = p - qF using the VanderMonde matrices.
        That defines the problem Ax = 0 and we solve for x in an SVD manner,
        exploiting A = U x S x V.T
        We get the solution as the last column in V (corresponds to the smallest singular value)
        """
        FQ = - (VN.T * self._Y).T
        A = np.hstack([VM, FQ])
        U, S, Vh = np.linalg.svd(A)
        self._pcoeff = Vh[-1][:self._M]
        self._qcoeff = Vh[-1][self._M:]

    def fit(self, **kwargs):
        """
        Do everything.
        """
        # Set M, N, K, polynomial structures
        from apprentice import tools
        n_required = tools.numCoeffsRapp(self.dim, (self.m, self.n))
        if n_required > self._Y.size:
            raise Exception("Not enough inputs: got %i but require %i to do m=%i n=%i"%(self._Y.size, n_required, self.m,self.n))

        self.setStructures()

        from apprentice import monomial
        VM = monomial.vandermonde(self._X, self._m)
        VN = monomial.vandermonde(self._X, self._n)
        strategy=kwargs["strategy"] if kwargs.get("strategy") is not None else 1
        if   strategy==1: self.coeffSolve( VM, VN)
        elif strategy==2: self.coeffSolve2(VM, VN)
        elif strategy==3: self.coeffSolve3(VM, VN)
        # NOTE, strat 1 is faster for smaller problems (Npoints < 250)
        else: raise Exception("fit() strategy %i not implemented"%strategy)

    def Q(self, X):
        """
        Evaluation of the denom poly at X.
        """
        rec_q = np.array(self.recurrence(X, self._struct_q))
        q = self._qcoeff.dot(rec_q)
        return q

    def denom(self, X):
        """
        Alias for Q, for compatibility
        """
        return self.Q(X)

    def P(self, X):
        """
        Evaluation of the numer poly at X.
        """
        rec_p = np.array(self.recurrence(X, self._struct_p))
        p = self._pcoeff.dot(rec_p)
        return p

    def predict(self, X):
        """
        Return the prediction of the RationalApproximation at X.
        """
        X=self._scaler.scale(np.array(X))
        return self.P(X)/self.Q(X)

    def __call__(self, X):
        """
        Operator version of predict.
        """
        return self.predict(X)

    def __repr__(self):
        """
        Print-friendly representation.
        """
        return "<RationalApproximation dim:{} m:{} n:{}>".format(self.dim, self.m, self.n)

    @property
    def asDict(self):
        """
        Store all info in dict as basic python objects suitable for JSON
        """
        d={}
        d["dim"]    = self.dim
        d["trainingsize"] = self.trainingsize
        d["m"]      = self.m
        d["n"]      = self.n
        d["pcoeff"] = list(self._pcoeff)
        d["qcoeff"] = list(self._qcoeff)
        d["scaler"] = self._scaler.asDict
        return d

    def save(self, fname):
        import json
        with open(fname, "w") as f:
            json.dump(self.asDict, f)

    def mkFromDict(self, pdict):
        self._pcoeff = np.array(pdict["pcoeff"])
        self._qcoeff = np.array(pdict["qcoeff"])
        self._m      = int(pdict["m"])
        self._n      = int(pdict["n"])
        self._dim    = int(pdict["dim"])
        self._scaler = apprentice.Scaler(pdict["scaler"])
        if self._dim==1: self.recurrence=apprentice.monomial.recurrence1D
        else           : self.recurrence=apprentice.monomial.recurrence
        try:
            self._trainingsize = int(pdict["trainingsize"])
        except:
            pass
        self.setStructures()

    def mkFromJSON(self, fname):
        import json
        d = json.load(open(fname))
        self.mkFromDict(d)

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
        for q in self._qcoeff:
            nrm+= abs(q)
        return nrm

    @property
    def coeff2Norm(self):
        nrm = 0
        for p in self._pcoeff:
            nrm+= p*p
        for q in self._qcoeff:
            nrm+= q*q
        return np.sqrt(nrm)

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

    import pylab
    pylab.plot(X, Y, marker="*", linestyle="none", label="Data")
    TX = sorted(X)
    for s in range(1,4):
        r=RationalApproximation(X,Y, order=(5,5), strategy=s)

        YW = [r(p) for p in TX]

        pylab.plot(TX, YW, label="Rational approx m={} n={} strategy {}".format(2,2,s))

    # Store the last and restore immediately, plot to see if all is good
    r.save("rapptest.json")
    r=RationalApproximation(fname="rapptest.json")
    YW = [r(p) for p in TX]
    pylab.plot(TX, YW, "m--", label="Restored m={} n={} strategy {}".format(2,2,s))
    pylab.legend()
    pylab.xlabel("x")
    pylab.ylabel("f(x)")
    pylab.savefig("demo.pdf")

    sys.exit(0)
