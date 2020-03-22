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

    def mkFromFile(self, f_approx, binids=None, **kwargs):
        binids, RA = apprentice.tools.readApprox(f_approx, set_structures=False, usethese=binids)
        self._binids=np.array(binids)
        self._RA = np.array(RA)
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
        self._HH = np.ones((self.dim, self.dim, len(S))             , dtype=np.float64) # Prefactors
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
        self._PC = np.zeros((len(self._RA), lmax), dtype=np.float64)
        for num, r in enumerate(self._RA): self._PC[num][:r._pcoeff.shape[0]] = r._pcoeff

        # Denominator
        if lmax_q > 0:
            self._hasRationals = True
            self._QC = np.zeros((len(self._RA), lmax), dtype=np.float64)
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
            return np.array(Pprime/Q.transpose() - (P/Q/Q).transpose()*Qprime, dtype=np.float64)

        return np.array(Pprime, dtype=np.float64)

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

        HESS = np.empty((self.dim, self.dim, NSEL), dtype=np.float64)
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
        else:                    self.mkFromData( *args, **kwargs) # NOT impemented

    @property
    def dim(self): return self._AS.dim

    @property
    def pnames(self): return self._SCLR.pnames

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

    def setWeights(self, wdict):
        """
        Convenience function to update the bins weights.
        NOTE that hnames is in fact an array of strings repeating the histo name for each corresp bin
        """
        weights = []
        for hn in self._hnames[self._good]: weights.append(wdict[hn])
        self._W2 = np.array([w * w for w in np.array(weights)], dtype=np.float64)

    def setLimits(self, fname):
        lim, fix = apprentice.tools.read_limitsandfixed(fname)
        for num, pn in enumerate(self.pnames):
            if pn in lim:
                self._bounds[num] = lim[pn]

    def setAttributes(self, **kwargs):
        noiseexp = int(kwargs.get("noise_exponent")) if kwargs.get("noise_exponent") is not None else 2
        self._dim = self._AS.dim
        self._E2 = np.array([1. / e ** noiseexp for e in self._E], dtype=np.float64)
        self._SCLR = self._AS._SCLR
        self._bounds = self._SCLR.box
        if kwargs.get("limits") is not None: self.setLimits(kwargs["limits"])
        self._debug = kwargs["debug"] if kwargs.get("debug") is not None else False

    def mkFromFiles(self, f_weights, f_data, f_approx, f_errors=None, **kwargs):
        AS = AppSet(f_approx)
        hnames  = [    b.split("#")[0]  for b in AS._binids]
        bnums   = [int(b.split("#")[1]) for b in AS._binids]
        weights = self.initWeights(f_weights, hnames, bnums)
        nonzero = np.where(weights>0)


        # Filter here to use only certain bins/histos
        dd = apprentice.tools.readExpData(f_data, [str(b) for b in AS._binids[nonzero]])
        Y = np.array([dd[b][0] for b in AS._binids[nonzero]], dtype=np.float64)
        E = np.array([dd[b][1] for b in AS._binids[nonzero]], dtype=np.float64)

        # Filter for wanted bins here and get rid of division by zero in case of 0 error which is undefined behaviour
        good = []
        for num, bid in enumerate(AS._binids[nonzero]):
            if E[num] > 0:
                if AS._RA[0]._scaler != AS._RA[num]._scaler:
                    if self._debug: print("Warning, dropping bin with id {} to guarantee caching works".format(bid))
                    continue
                good.append(num)
            else:
                if self._debug: print("Warning, dropping bin with id {} as its weight or error is 0. W = {}, E = {}".format(bid,weights[nonzero][num],E[num]))

        self._good = good


        # TODO This needs some re-engineering to allow fow multiple filterings
        RA = [AS._RA[nonzero][g] for g in good]
        self._binids = [AS._binids[nonzero][g] for g in good]
        self._AS = AppSet(RA, self._binids)
        self._E = E[good]
        self._Y = Y[good]
        self._W2 = np.array([w  for w in np.array(weights[nonzero])[good]], dtype=np.float64)
        # self._W2 = np.array([w * w for w in np.array(weights[nonzero])[good]], dtype=np.float64)
        self._hnames = np.array([b.split("#")[0]  for b in AS._binids[nonzero]])
        # Add in error approximations
        if f_errors is not None:
            EAS = AppSet(f_errors)
            ERA = [EAS._RA[g] for g in good]
            self._EAS=AppSet(ERA, self._binids)
        else:
            self._EAS=None
        self.setAttributes(**kwargs)

    def objective(self, x, sel=slice(None, None, None), unbiased=False):
        vals = self._AS.vals(x, sel=sel)
        if self._EAS is not None:
            err2 = self._EAS.vals(x, sel=sel)**2
        else:
            err2=np.zeros_like(vals)
        if unbiased: return apprentice.tools.fast_chi(np.ones(len(vals)), self._Y[sel] - vals, 1./(err2 + 1./self._E2[sel]))
        else:        return apprentice.tools.fast_chi(self._W2[sel]     , self._Y[sel] - vals, 1./(err2 + 1./self._E2[sel]))# self._E2[sel])

    def gradient(self, x, sel=slice(None, None, None)):
        vals  = self._AS.vals( x, sel=sel)
        E2=1./self._E2[sel]
        grads = self._AS.grads(x, sel=sel, set_cache=False)
        if self._EAS is not None:
            err   = self._EAS.vals(  x, sel=sel, set_cache=False)
            egrads = self._EAS.grads( x, sel=sel, set_cache=False)
        else:
            err= np.zeros_like(vals)
            egrads = np.zeros_like(grads)
        return apprentice.tools.fast_grad2(self._W2[sel], self._Y[sel] - vals, E2, err,grads, egrads)

    def hessian(self, x, sel=slice(None, None, None)):
        vals  = self._AS.vals( x, sel = sel)
        grads = self._AS.grads(x, sel, set_cache=False)
        hess  = self._AS.hessians(x, sel)
        if self._EAS is not None:
            evals  = self._EAS.vals( x, sel = sel)
            egrads = self._EAS.grads(x, sel, set_cache=False)
            ehess  = self._EAS.hessians(x, sel)
        else:
            evals  = np.zeros_like(vals)
            egrads = np.zeros_like(grads)
            ehess  = np.zeros_like(hess)

        # Some useful definitions
        E2=1./self._E2[sel]
        lbd = E2 + evals*evals
        kap = vals - self._Y[sel]
        G1 = 2./lbd
        G2 = -4*kap*evals/lbd/lbd
        G3 =  2*kap/lbd

        H2 = -2 * kap*kap/lbd/lbd
        H3 = -2 * evals * kap /lbd * G2

        spans = np.zeros( (self.dim, self.dim, len(vals)) )
        # TODO explicitly exploit symmetry
        for numx in range(self.dim):
            for numy in range(self.dim):
                spans[numx][numy] +=        G1 *  grads[:,numx] *  grads[:,numy]
                spans[numx][numy] +=     2* G2 * egrads[:,numx] *  grads[:,numy]
                spans[numx][numy] += (H2 + H3) * egrads[:,numx] * egrads[:,numy]
        spans += G3*hess
        spans += H2*evals*ehess

        return np.sum( self._W2[sel]*(spans), axis=2)

    def startPoint(self, ntrials):
        if ntrials == 0:
            if self._debug: print("StartPoint: {}".format(self._SCLR.center))
            return self._SCLR.center
        import numpy as np
        _PP = np.random.uniform(low=self._bounds[:,0], high=self._bounds[:,1], size=(ntrials, self._SCLR.dim))
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

    # NOTE fast but tendency to end up outside box
    def minimizeTrust(self, nstart=1, nrestart=1, sel=slice(None, None, None), use_mpi=False):
        from scipy import optimize
        minobj = np.Infinity
        finalres = None
        import time
        t0=time.time()
        for t in range(nrestart):
            x0 = np.array(self.startPointMPI(nstart) if use_mpi else self.startPoint(nstart), dtype=np.float64)

            res = optimize.minimize(lambda x: self.objective(x, sel=sel), x0, jac=self.gradient, hess=self.hessian, method="trust-exact")
            if res["fun"] < minobj:
                minobj = res["fun"]
                finalres = res
        t1=time.time()
        if self._debug:
            print(t1-t0)
        return finalres

    def minimizeNCG(self, nstart=1, nrestart=1, sel=slice(None, None, None), use_mpi=False):
        from scipy import optimize
        minobj = np.Infinity
        finalres = None
        for t in range(nrestart):
            x0 = np.array(self.startPointMPI(nstart) if use_mpi else self.startPoint(nstart), dtype=np.float64)

            res = optimize.minimize(lambda x: self.objective(x, sel=sel), x0, jac=self.gradient, hess=self.hessian, method="Newton-CG")
            if res["fun"] < minobj:
                minobj = res["fun"]
                finalres = res
        return finalres

    def minimize(self, nstart=1, nrestart=1, sel=slice(None, None, None), use_grad=True, tol=1e-4,  method="TNC", use_mpi=False):
        from scipy import optimize
        minobj = np.Infinity
        finalres = None
        import time
        t0=time.time()
        for t in range(nrestart):
            x0 = np.array(self.startPointMPI(nstart) if use_mpi else self.startPoint(nstart), dtype=np.float64)

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
        t1=time.time()
        if self._debug:
            print(t1-t0)
        return finalres

    def minimizeLBFGSB(self, nstart=1, nrestart=1, sel=slice(None, None, None), use_grad=True, tol=1e-4,  method="L-BFGS-B", use_mpi=False):
        from scipy import optimize
        minobj = np.Infinity
        finalres = None
        import time
        t0=time.time()
        for t in range(nrestart):
            x0 = np.array(self.startPointMPI(nstart) if use_mpi else self.startPoint(nstart), dtype=np.float64)

            if use_grad:
                if self._debug: print("using gradient")
                res = optimize.minimize(lambda x: self.objective(x, sel=sel), x0,
                        bounds=self._bounds, jac=self.gradient, method=method, tol=tol)
            else:
                res = optimize.minimize(lambda x: self.objective(x, sel=sel), x0,
                        bounds=self._bounds, method=method, tol=tol)
            if res["fun"] < minobj:
                minobj = res["fun"]
                finalres = res
        t1=time.time()
        if self._debug:
            print(t1-t0)
        return finalres

    def writeParams(self, x, fname):
        with open(fname, "w") as f:
            for pn, val in zip(self.pnames, x):
                f.write("{}\t{}\n".format(pn, val))

    def printParams(self, x):
        s= ""
        for num, (pn, val) in enumerate(zip(self.pnames, x)):
            s+= ("{}\t{}  [{} ... {}]\n".format(pn, val, self._SCLR.box[num][0], self._SCLR.box[num][1]))
        return s

    def lineScan(self, x0, dim, npoints=100, bounds=None):
        if bounds is None:
            xmin, xmax = self._SCLR.box[dim]
        else:
            xmin, xmax = bounds

        xcoords = list(np.linspace(xmin, xmax, npoints))
        xcoords.append(x0[dim])
        xcoords.sort()

        X = np.tile(x0, (len(xcoords),1))
        for num, x in enumerate(X):
            x[dim] = xcoords[num]
        return X



    def __len__(self): return len(self._AS)
