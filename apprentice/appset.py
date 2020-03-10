import apprentice
import numpy as np

## legacy Prof2 code
def calcHistoCov(h, COV_P, result):
    """
    Propagate the parameter covariance onto the histogram covariance
    using the ipol gradients.
    """
    IBINS = h.bins
    from numpy import zeros
    COV_H = zeros((h.nbins, h.nbins))
    from numpy import array
    for i in range(len(IBINS)):
        GRD_i = array(IBINS[i].grad(result))
        for j in range(len(IBINS)):
            GRD_j = array(IBINS[j].grad(result))
            pc =GRD_i.dot(COV_P).dot(GRD_j)
            COV_H[i][j] = pc
    return COV_H


from numba import jit, njit
@jit(parallel=True, forceobj=True)
def startPoints(self, _PP):
    _CH = np.empty(len(_PP))
    for p in range(len(_PP)):
       _CH[p] = self.objective(_PP[p])
    return _PP[np.argmin(_CH)]

class AppSet(object):
    """
    Collection of Apprentice approximations with the same support.
    """
    def __init__(self, *args, **kwargs):
        self._debug = kwargs["debug"] if kwargs.get("debug") is not None else False
        if type(args[0]) == str:
            self.mkFromFile(*args, **kwargs)
        else:
            self.mkFromData(*args, **kwargs)

    @property
    def dim(self): return self._dim

    def mkFromFile(self, f_approx, **kwargs):
        self._binids, self._RA = apprentice.tools.readApprox(f_approx, set_structures=False)
        self.setAttributes(**kwargs)

    def mkFromData(self, RA, binids, **kwargs):
        self._RA = RA
        self._binids = binids
        self.setAttributes(**kwargs)

    def setAttributes(self, **kwargs):
        self._hnames = sorted(list(set([b.split("#")[0] for b in self._binids])))
        self._dim = self._RA[0].dim
        self._SCLR = self._RA[0]._scaler  # Here we quietly assume already that all scalers are identical
        self._bounds = self._SCLR.box
        self._debug = kwargs["debug"] if kwargs.get("debug") is not None else False
        if self.dim == 1: self.recurrence = apprentice.monomial.recurrence1D
        else:             self.recurrence = apprentice.monomial.recurrence
        self.setStructures()
        self.setCoefficients()

    def setStructures(self):
        omax_p=np.max([r.m                            for r in self._RA])
        omax_q=np.max([r.n if hasattr(r, "n") else 0  for r in self._RA])
        omax = max(omax_p, omax_q)

        self._structure = np.array(apprentice.monomialStructure(self.dim, omax), dtype=np.int32)
        S=self._structure
        # Gradient helpers
        self._NNZ  = [np.where(self._structure[:, coord] != 0) for coord in range(self.dim)]
        self._sred = np.array([self._structure[nz][:,num] for num, nz in enumerate(self._NNZ)], dtype=np.int32)
        # Hessian helpers
        self._HH = np.ones((self.dim, self.dim, len(S))             , dtype=np.float32) # Prefactors
        self._EE = np.full((self.dim, self.dim, len(S), self.dim), S, dtype=np.int32) # Initial structures

        for numx in range(self.dim):
            for numy in range(self.dim):
                if numx==numy:
                    self._HH[numx][numy] = S[:,numx] * (S[:,numx]-1)
                else:
                    self._HH[numx][numy] = S[:,numx] *  S[:,numy]
                self._EE[numx][numy][:,numx]-=1
                self._EE[numx][numy][:,numy]-=1

        self._HNONZ = np.empty((self.dim, self.dim), dtype=tuple)
        for numx in range(self.dim):
            for numy in range(self.dim):
                self._HNONZ[numx][numy]=np.where(self._HH[numx][numy]>0)

        # Jacobians for Hessian
        JF = self._SCLR.jacfac
        for numx in range(self.dim):
            for numy in range(self.dim):
                self._HH[numx][numy][self._HNONZ[numx][numy]] *= (JF[numx] * JF[numy])

    def setCoefficients(self):
        # Need maximum extends of coefficients
        lmax_p=np.max([r._pcoeff.shape[0]                           for r in self._RA])
        lmax_q=np.max([r._qcoeff.shape[0] if hasattr(r, "n") else 0 for r in self._RA])
        lmax = max(lmax_p, lmax_q)
        self._PC = np.zeros((len(self._RA), lmax), dtype=np.float32)
        for num, r in enumerate(self._RA): self._PC[num][:r._pcoeff.shape[0]] = r._pcoeff

        # Denominator
        if lmax_q > 0:
            self._hasRationals = True
            self._QC = np.zeros((len(self._RA), lmax), dtype=np.float32)
            for num, r in enumerate(self._RA):
                if hasattr(r, "n"):
                    self._QC[num][:r._qcoeff.shape[0]] = r._qcoeff
                else:
                    self._QC[num][0] = None
            self._mask = np.where(np.isfinite(self._QC[:, 0]))
        else:
            self._hasRationals = False

    def setRecurrence(self, x):
        xs = self._SCLR.scale(x)
        self._maxrec = self.recurrence(xs, self._structure)

    def vals(self, x, sel=slice(None, None, None), set_cache=True):
        if set_cache: self.setRecurrence(x)
        MM=self._maxrec * self._PC[sel]
        vals = np.sum(MM, axis=1)
        if self._hasRationals:
            den = np.sum(self._maxrec * self._QC[sel], axis=1)
            vals[self._mask[sel]] /= den[self._mask[sel]]
        return vals

    def grads(self, x, sel=slice(None, None, None), set_cache=True):
        if set_cache: self.setRecurrence(x)
        xs = self._SCLR.scale(x)
        JF = self._SCLR.jacfac
        GREC = apprentice.tools.gradientRecursionFast(xs, self._structure, self._SCLR.jacfac, self._NNZ, self._sred)

        Pprime = np.sum(self._PC[sel].reshape((self._PC[sel].shape[0], 1, self._PC[sel].shape[1])) * GREC, axis=2)

        if self._hasRationals:
            P = np.atleast_2d(np.sum(self._maxrec * self._PC[sel], axis=1))
            Q = np.atleast_2d(np.sum(self._maxrec * self._QC[sel], axis=1))
            Qprime = np.sum(self._QC[sel].reshape((self._QC[sel].shape[0], 1, self._QC[sel].shape[1])) * GREC, axis=2)
            return np.array(Pprime/Q.transpose() - (P/Q/Q).transpose()*Qprime, dtype=np.float32)

        return np.array(Pprime, dtype=np.float32)

    def hessians(self, x, sel=slice(None, None, None)):
        """
        To get the hessian matrix of bin number N, do
        H=hessians(pp)
        H[:,:,N]
        """
        if self._hasRationals:
            raise Exception("Hessians for rational functions not (yet) implemented")
        xs = self._SCLR.scale(x)

        NSEL = len(self._PC[sel])

        HESS = np.empty((self.dim, self.dim, NSEL), dtype=np.float32)
        for numx in range(self.dim):
            for numy in range(self.dim):
                rec = self._HH[numx][numy][self._HNONZ[numx][numy]] * np.prod(np.power(xs, self._EE[numx][numy][self._HNONZ[numx][numy]]), axis=1)
                if numy>=numx:
                    HESS[numx][numy] = np.sum((rec*self._PC[:,self._HNONZ[numx][numy][0]]), axis=1)
                else:
                    HESS[numx][numy] = HESS[numy][numx]

        return HESS

    def __len__(self): return len(self._RA)

    def rbox(self, ntrials):
        return np.random.uniform(low=self._SCLR._Xmin, high=self._SCLR._Xmax, size=(ntrials, self._SCLR.dim))

class TuningObjective2(object):
    def __init__(self, *args, **kwargs):
        self._debug = kwargs["debug"] if kwargs.get("debug") is not None else False
        if type(args[0]) == str: self.mkFromFiles(*args, **kwargs)
        else:                    self.mkFromData( *args, **kwargs)

    @property
    def dim(self): return self._AS.dim

    def rbox(self, ntrials):
        return self._AS.rbox(ntrials)

    def initWeights(self, fname, hnames, bnums):
        matchers = apprentice.weights.read_pointmatchers(fname)
        weights = []
        for hn, bnum in zip(hnames, bnums):
            pathmatch_matchers = [(m, wstr) for  m, wstr  in matchers.items()    if m.match_path(hn)]
            posmatch_matchers  = [(m, wstr) for (m, wstr) in pathmatch_matchers if m.match_pos(bnum)]
            w = float(posmatch_matchers[-1][1]) if posmatch_matchers else 0  # < NB. using last match
            weights.append(w)
        return np.array(weights)

    def setLimits(self, fname):
        lim, fix = apprentice.tools.read_limitsandfixed(fname)
        for num, pn in enumerate(self.pnames):
            if pn in lim:
                self._bounds[num] = lim[pn]

    def setAttributes(self, **kwargs):
        noiseexp = int(kwargs.get("noise_exponent")) if kwargs.get("noise_exponent") is not None else 2
        self._dim = self._AS.dim
        self._E2 = np.array([1. / e ** noiseexp for e in self._E], dtype=np.float32)
        self._SCLR = self._AS._SCLR
        self._bounds = self._SCLR.box
        if kwargs.get("limits") is not None: self.setLimits(kwargs["limits"])
        self._debug = kwargs["debug"] if kwargs.get("debug") is not None else False

    def mkFromFiles(self, f_weights, f_data, f_approx, **kwargs):
        AS = AppSet(f_approx)
        # hnames = sorted(list(set([b.split("#")[0] for b in AS._binids])))
        hnames = [b.split("#")[0] for b in AS._binids]
        bnums = [int(b.split("#")[1]) for b in AS._binids]
        weights = self.initWeights(f_weights, hnames, bnums)

        # Filter here to use only certain bins/histos
        dd = apprentice.tools.readExpData(f_data, [str(b) for b in AS._binids])
        Y = np.array([dd[b][0] for b in AS._binids], dtype=np.float32)
        E = np.array([dd[b][1] for b in AS._binids], dtype=np.float32)

        # Filter for wanted bins here and get rid of division by zero in case of 0 error which is undefined behaviour
        good = []
        for num, bid in enumerate(AS._binids):
            if weights[num] > 0 and E[num] > 0:
                if AS._RA[0]._scaler != AS._RA[num]._scaler:
                    if self._debug: print("Warning, dropping bin with id {} to guarantee caching works".format(bid))
                    continue
                good.append(num)
            else:
                if self._debug: print("Warning, dropping bin with id {} as its weight or error is 0. W = {}, E = {}".format(bid,weights[num],E[num]))

        # TODO This needs some re-engineering to allow fow multiple filterings
        RA = [AS._RA[g] for g in good]
        self._binids = [AS._binids[g] for g in good]
        self._AS = AppSet(RA, self._binids)
        self._E = E[good]
        self._Y = Y[good]
        self._W2 = np.array([w * w for w in np.array(weights)[good]], dtype=np.float32)
        self.setAttributes(**kwargs)

    def objective(self, x, sel=slice(None, None, None), unbiased=False):
        vals = self._AS.vals(x, sel=sel)
        if unbiased: return apprentice.tools.fast_chi(np.ones(len(vals)), self._Y[sel] - vals, self._E2[sel])
        else:        return apprentice.tools.fast_chi(self._W2[sel]     , self._Y[sel] - vals, self._E2[sel])

    def gradient(self, x, sel=slice(None, None, None)):
        vals  = self._AS.vals( x, sel = sel)
        grads = self._AS.grads(x, sel, set_cache=False)
        return apprentice.tools.fast_grad(self._W2[sel], self._Y[sel] - vals, self._E2[sel], grads)

    def hessian(self, x, sel=slice(None, None, None)):
        vals  = self._AS.vals( x, sel = sel)
        grads = self._AS.grads(x, sel, set_cache=False)
        hess  = self._AS.hessians(x, sel)

        spans = np.empty( (self.dim, self.dim, len(vals)) )
        for numx in range(self.dim):
            for numy in range(self.dim):
                if numy>=numx:
                    spans[numx][numy] = grads[:,numx] *  grads[:,numy]
                else:
                    spans[numx][numy] = spans[numy][numx]

        diff = vals - self._Y[sel]

        return 2 * np.sum( self._W2[sel]*self._E2[sel]*(spans+diff*hess), axis=2)

    def startPoint(self, ntrials):
        if ntrials == 0:
            if self._debug: print("StartPoint: {}".format(self._SCLR.center))
            return self._SCLR.center
        import numpy as np
        _PP = np.random.uniform(low=self._SCLR._Xmin, high=self._SCLR._Xmax, size=(ntrials, self._SCLR.dim))
        _CH = [self.objective(p) for p in _PP]
        if self._debug: print("StartPoint: {}".format(_PP[_CH.index(min(_CH))]))
        return _PP[_CH.index(min(_CH))]

    def startPointMPI(self, ntrials):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        XX = self.rbox(ntrials)
        rankWork = apprentice.tools.chunkIt(XX, comm.Get_size()) if rank == 0 else []
        rankWork = comm.scatter(rankWork, root=0)
        temp = [self.objective(x) for x in rankWork]
        ibest = np.argmin(temp)
        X = comm.gather(XX[ibest], root=0)
        FUN = comm.gather(temp[ibest], root=0)
        xbest = None
        if rank == 0:
            ibest = np.argmin(FUN)
            xbest = X[ibest]
        xbest = comm.bcast(xbest, root=0)
        return xbest

    def minimizeTrust(self, nstart=1, nrestart=1, sel=slice(None, None, None), use_mpi=False):
        from scipy import optimize
        minobj = np.Infinity
        finalres = None
        for t in range(nrestart):
            x0 = np.array(self.startPointMPI(nstart) if use_mpi else self.startPoint(nstart), dtype=np.float32)

            res = optimize.minimize(lambda x: self.objective(x, sel=sel), x0, jac=self.gradient, hess=self.hessian, method="trust-exact")
            if res["fun"] < minobj:
                minobj = res["fun"]
                finalres = res
        return finalres

    def minimize(self, nstart=1, nrestart=1, sel=slice(None, None, None), use_grad=True, tol=1e-4,  method="TNC", use_mpi=False):
        from scipy import optimize
        minobj = np.Infinity
        finalres = None
        for t in range(nrestart):
            x0 = np.array(self.startPointMPI(nstart) if use_mpi else self.startPoint(nstart), dtype=np.float32)

            if use_grad:
                if self._debug: print("using gradient")
                res = optimize.minimize(lambda x: self.objective(x, sel=sel), x0,
                        bounds=self._bounds, jac=self.gradient, method=method, tol=tol, options={'maxiter':1000, 'accuracy':tol})
            else:
                res = optimize.minimize(lambda x: self.objective(x, sel=sel), x0,
                        bounds=self._bounds, method=method, tol=tol, options={'maxiter':1000, 'accuracy':tol})
            if res["fun"] < minobj:
                minobj = res["fun"]
                finalres = res
        return finalres

    def __len__(self): return len(self._AS)
