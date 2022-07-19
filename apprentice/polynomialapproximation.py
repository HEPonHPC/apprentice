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

class PolynomialApproximation():
    def __init__(self, X=None, Y=None, order=2, fname=None, initDict=None, strategy=2, scale_min=-1, scale_max=1, pnames=None, set_structures=True, computecov=False):
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
        self._vmin=None
        self._vmax=None
        self._xmin=None
        self._xmax=None
        if initDict is not None:
            self.mkFromDict(initDict, set_structures=set_structures)
        elif fname is not None:
            self.mkFromJSON(fname, set_structures=set_structures)
        elif X is not None and Y is not None:
            self._m=order
            self._scaler = apprentice.Scaler(np.atleast_2d(np.array(X, dtype=np.float64)), a=scale_min, b=scale_max, pnames=pnames)
            self._X   = self._scaler.scaledPoints
            self._dim = self._X[0].shape[0]
            self._Y   = np.array(Y, dtype=np.float64)
            self._trainingsize=len(X)
            if self._dim==1: self.recurrence=apprentice.monomial.recurrence1D
            else           : self.recurrence=apprentice.monomial.recurrence
            self.fit(strategy=strategy, computecov=computecov)
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
    @property
    def vmin(self): return self._vmin
    @property
    def vmax(self): return self._vmax
    @property
    def xmin(self): return self._xmin
    @property
    def xmax(self): return self._xmax

    def setStructures(self):
        self._struct_p = apprentice.monomialStructure(self.dim, self.m)
        from apprentice import tools
        self._M        = tools.numCoeffsPoly(self.dim, self.m)

        self._nnz = self._struct_p>0

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
        x, res, rank, s  = np.linalg.lstsq(VM, self._Y, rcond=rcond)
        self._pcoeff = x

    def fit(self, **kwargs):
        """
        Do everything
        """
        from apprentice import tools
        n_required = tools.numCoeffsPoly(self.dim, self.m)
        if n_required > self._Y.shape[0]:
            raise Exception("Not enough inputs: got %i but require %i to do m=%i"%(self._Y.shape[0], n_required, self.m))

        self.setStructures()

        from apprentice import monomial
        VM = monomial.vandermonde(self._X, self.m)
        strategy=kwargs["strategy"] if kwargs.get("strategy") is not None else 1
        if kwargs.get("computecov") is not False:
            self._cov = np.linalg.inv(2*VM.T@VM)
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
        return self._pcoeff.dot(rec_p)

    def predict2(self, X):
        """
        Evaluation of the numer poly at X.
        10% faster than predict --- exploit structure somewhat
        """
        X=self._scaler.scale(np.array(X))
        rec_p = apprentice.monomial.recurrence2(X, self._struct_p, self._nnz)
        return self._pcoeff.dot(rec_p)

    def predictArray(self, X):
        """
        Evaluation of the numer poly at many points X.
        """
        XS=self._scaler.scale(X)
        if self.dim > 1:
            zz=np.ones((len(XS), *self._struct_p.shape))
            np.power(XS, self._struct_p[:, np.newaxis], out=(zz), where=self._struct_p[:, np.newaxis]>0)
            rec_p = np.prod(zz, axis=2)
            # rec_p = np.prod(np.power(XS, self._struct_p[:, np.newaxis]), axis=2)
        else:
            rec_p = np.power(XS, self._struct_p[:, np.newaxis])
        return self._pcoeff.dot(rec_p)


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
        if self._vmin is not None: d["vmin"] = self._vmin
        if self._vmax is not None: d["vmax"] = self._vmax
        if self._xmin is not None: d["xmin"] = self._xmin
        if self._xmax is not None: d["xmax"] = self._xmax
        return d

    def save(self, fname):
        import json
        with open(fname, "w") as f:
            json.dump(self.asDict, f)

    def mkFromJSON(self, fname, set_structures=True):
        import json
        d = json.load(open(fname))
        self.mkFromDict(d, set_structures=set_structures)

    def mkFromDict(self, pdict, set_structures=True):
        self._pcoeff     = np.array(pdict["pcoeff"])
        self._m = int(pdict["m"])
        self._dim=int(pdict["dim"])
        self._scaler = apprentice.Scaler(pdict["scaler"])
        if self._dim==1: self.recurrence=apprentice.monomial.recurrence1D
        else           : self.recurrence=apprentice.monomial.recurrence
        if "vmin" in pdict: self._vmin = pdict["vmin"]
        if "vmax" in pdict: self._vmax = pdict["vmax"]
        if "xmin" in pdict: self._xmin = pdict["xmin"]
        if "xmax" in pdict: self._xmax = pdict["xmax"]
        try:
            self._trainingsize = int(pdict["trainingsize"])
        except:
            pass
        if set_structures:
            self.setStructures()

    def fmin(self, nsamples=1, nrestart=1, use_grad=False):
        return apprentice.tools.extreme(self, nsamples, nrestart, use_grad, mode="min")

    def fmax(self, nsamples=1, nrestart=1, use_grad=False):
        return apprentice.tools.extreme(self, nsamples, nrestart, use_grad, mode="max")

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


    def gradient(self, X):
        import numpy as np
        struct = np.array(self._struct_p, dtype=np.float)
        X = self._scaler.scale(np.array(X))

        if self.dim==1:
            struct[1:]=self._scaler.jacfac[0]*struct[1:]*np.power(X, struct[1:]-1)
            return np.dot(np.atleast_2d(struct),self._pcoeff)

        from apprentice.tools import gradientRecursion
        GREC = gradientRecursion(X, struct, self._scaler.jacfac)

        return np.sum(GREC * self._pcoeff, axis=1)

    def hessian(self, X):
        import numpy as np
        X = self._scaler.scale(np.array(X))
        S = self._struct_p

        HH = np.ones((self.dim, self.dim, len(S)), dtype=np.float)
        EE = np.full((self.dim, self.dim, len(S), self.dim), S, dtype=np.int32)

        for numx in range(self.dim):
            for numy in range(self.dim):
                if numx==numy:
                    HH[numx][numy] = S[:,numx] * (S[:,numx]-1)
                else:
                    HH[numx][numy] = S[:,numx] *  S[:,numy]
                EE[numx][numy][:,numx]-=1
                EE[numx][numy][:,numy]-=1

        NONZ = np.empty((self.dim, self.dim), dtype=tuple)
        for numx in range(self.dim):
            for numy in range(self.dim):
                NONZ[numx][numy]=np.where(HH[numx][numy]>0)

        JF = self._scaler.jacfac
        for numx in range(self.dim):
            for numy in range(self.dim):
                HH[numx][numy][NONZ[numx][numy]] *= (JF[numx] * JF[numy])

        HESS = np.empty((self.dim, self.dim), dtype=np.float)
        for numx in range(self.dim):
            for numy in range(self.dim):
                if numy>=numx:
                    HESS[numx][numy] = np.sum(HH[numx][numy][NONZ[numx][numy]] * np.prod(np.power(X, EE[numx][numy][NONZ[numx][numy]]), axis=1) * self._pcoeff[NONZ[numx][numy]])
                else:
                    HESS[numx][numy] = HESS[numy][numx]

        return HESS

    def wraps(self, v):
        dec=True
        if self.vmin is not None and self.vmax is not None:
            if self.vmin > v or self.vmax < v:dec=False
        return dec



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


    import autograd.numpy as np
    from autograd import hessian, grad

    def f(x):
        return 2*x**2 +1

    def fp(x):
        return 4*x

    X=np.linspace(0,10,11)
    # X=np.linspace(-1,1,11)
    Y=f(X)
    pp = PolynomialApproximation(X=[[x] for x in X], Y=Y, order=2, strategy=1)
    pylab.clf()
    pylab.plot(X, Y, marker="*", linestyle="none", label="Data")

    pylab.plot(X, [pp(x) for x in X], label="Polynomial approx m={} strategy 1".format(3))
    pylab.plot(X, pp.predictArray(X), label="Polynomial approx array  m={} strategy 1".format(3))

    myg = [pp.gradient(x) for x in X]

    g = grad(pp)
    G = [g(x) for x in X]
    FP = [fp(x) for x in X]

    pylab.plot(X, FP, marker="s", linestyle="none", label="analytic gradient")
    pylab.plot(X, G, label="auto gradient")
    pylab.plot(X, myg, label="manual gradient")
    pylab.legend()

    pylab.show()






    pylab.plot(TX, Y1, label="Polynomial approx m={} strategy 1".format(3))
    pylab.plot(TX, Y2, label="Polynomial approx m={} strategy 2".format(3))
    pylab.plot(TX, Y3, "m--", label="Restored polynomial approx m={} strategy 2".format(3))
    pylab.legend()
    pylab.xlabel("x")
    pylab.ylabel("f(x)")
    pylab.savefig("demopoly.pdf")
    sys.exit(0)
