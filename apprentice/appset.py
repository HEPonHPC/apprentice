import apprentice
import numpy as np
from apprentice.mpi4py_ import MPI_
import pandas as pd

# https://stackoverflow.com/questions/32808383/formatting-numbers-so-they-align-on-decimal-point
def dot_aligned(seq):
    snums = [str(n) for n in seq]
    dots = [s.find('.') for s in snums]
    m = max(dots)
    return [' '*(m - d) + s for s, d in zip(snums, dots)]

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


# from numba import jit, njit
# @jit(parallel=True, forceobj=True)
def startPoints(self, _PP):
    _CH = np.empty(len(_PP))
    for p in range(len(_PP)):
       _CH[p] = self.objective(_PP[p])
    return _PP[np.argmin(_CH)]

# @jit(forceobj=True)#, parallel=True)
def prime(GREC, COEFF, dim, NNZ):
    ret = np.empty((len(COEFF), dim))
    for i in range(dim):
        ret[:,i] = np.sum(COEFF[:,NNZ[i]] * GREC[i, NNZ[i]], axis=2).flatten()
    return ret

# @jit(forceobj=True)#, parallel=True)
def doubleprime(dim, xs, NSEL, HH, HNONZ, EE, COEFF):
    ret = np.empty((dim, dim, NSEL), dtype=np.float64)
    for numx in range(dim):
        for numy in range(dim):
            rec = HH[numx][numy][HNONZ[numx][numy]] * np.prod(np.power(xs, EE[numx][numy][HNONZ[numx][numy]]), axis=1)
            if numy>=numx:
                ret[numx][numy] = np.sum((rec*COEFF[:,HNONZ[numx][numy][0]]), axis=1)
            else:
                ret[numx][numy] = ret[numy][numx]

    return ret

# @jit
def jitprime(GREC, COEFF, dim):
    ret = np.empty((len(COEFF), dim))
    for i in range(dim):
        for j in range(len(COEFF)):
            ret[j,i] = np.sum(COEFF[j] * GREC[i])
    return ret



# @njit(parallel=True)
def calcSpans(spans1, DIM, G1, G2, H2, H3, grads, egrads):
    for numx in range(DIM):
        for numy in range(DIM):
            if numy<=numx:
                spans1[numx][numy] +=        G1 *  grads[:,numx] *  grads[:,numy]
                spans1[numx][numy] +=        G2 * (egrads[:,numx] *  grads[:,numy] + egrads[:,numy] *  grads[:,numx])
                spans1[numx][numy] += (H2 + H3) * egrads[:,numx] * egrads[:,numy]
    for numx in range(DIM):
        for numy in range(DIM):
            if numy>numx:
                spans1[numx][numy] = spans1[numy][numx]
    return spans1


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
        binids, RA = apprentice.io.readApprox(f_approx, set_structures=False, usethese=binids)
        self._binids=np.array(binids)
        self._RA = np.array(RA)
        self.setAttributes(**kwargs)

    def mkFromData(self, RA, binids, **kwargs):
        self._RA = RA
        self._binids = binids
        self.setAttributes(**kwargs)

    def mkReduced(self, keep, **kwargs):
        return AppSet(np.array(self._RA)[keep], np.array(self._binids)[keep], **kwargs)

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

    def vals(self, x, sel=slice(None, None, None), set_cache=True, maxorder=None):
        if set_cache: self.setRecurrence(x)
        if maxorder is None:
            MM=self._maxrec * self._PC[sel]
        else:
            nc = apprentice.tools.numCoeffsPoly(self.dim, 2)
            MM=self._maxrec[:nc] * self._PC[sel][:,:nc]
        vals = np.sum(MM, axis=1)
        if self._hasRationals:
            den = np.sum(self._maxrec * self._QC[sel], axis=1)
            vals/=den
            # FIXME this logic with the mask is not working
            # The code will divide by zero in case we hav mixed bits here
            # Note that this will go away come federations
            # vals[self._mask[sel]] /= den[self._mask[sel]]
        return vals

    def grads(self, x, sel=slice(None, None, None), set_cache=True):
        if set_cache: self.setRecurrence(x)
        xs = self._SCLR.scale(x)
        JF = self._SCLR.jacfac
        GREC = apprentice.tools.gradientRecursionFast(xs, self._structure, self._SCLR.jacfac, self._NNZ, self._sred)

        # NOTE this is expensive -- pybind11??
        # Pprime = np.sum(self._PC[sel].reshape((self._PC[sel].shape[0], 1, self._PC[sel].shape[1])) * GREC, axis=2)
        Pprime = prime(GREC, self._PC[sel], self.dim, self._NNZ)

        if self._hasRationals:
            P = np.atleast_2d(np.sum(self._maxrec * self._PC[sel], axis=1))
            Q = np.atleast_2d(np.sum(self._maxrec * self._QC[sel], axis=1))
            Qprime = prime(GREC, self._QC[sel], self.dim, self._NNZ)
            return np.array(Pprime/Q.transpose() - (P/Q/Q).transpose()*Qprime, dtype=np.float64)

        return np.array(Pprime, dtype=np.float64)

    # @jit(forceobj=True)#, parallel=True)
    def hessians(self, x, sel=slice(None, None, None)):
        """
        To get the hessian matrix of bin number N, do
        H=hessians(pp)
        H[:,:,N]
        """
        xs = self._SCLR.scale(x)

        NSEL = len(self._PC[sel])

        Phess = doubleprime(self.dim, xs, NSEL, self._HH, self._HNONZ, self._EE, self._PC[sel])

        #TODO check against autograd?
        if self._hasRationals:
            JF = self._SCLR.jacfac
            GREC = apprentice.tools.gradientRecursionFast(xs, self._structure, self._SCLR.jacfac, self._NNZ, self._sred)
            P = np.atleast_2d(np.sum(self._maxrec * self._PC[sel], axis=1))
            Q = np.atleast_2d(np.sum(self._maxrec * self._QC[sel], axis=1))
            Pprime = np.atleast_2d(prime(GREC, self._PC[sel], self.dim, self._NNZ))
            Qprime = np.atleast_2d(prime(GREC, self._QC[sel], self.dim, self._NNZ))
            Qhess = doubleprime(self.dim, xs, NSEL, self._HH, self._HNONZ, self._EE, self._QC[sel])

            w = Phess/Q
            for numx in range(self.dim):
                for numy in range(self.dim):
                    w[numx][numy] -= 2*(Pprime[:,numx]*Qprime[:,numy]/Q/Q).flatten()
                    w[numx][numy] += 2*(Qprime[:,numx]*Qprime[:,numy]*P/Q/Q/Q).flatten()

            w -= Qhess*(P/Q/Q)
            return w

        return Phess


    def __len__(self): return len(self._RA)

    def rbox(self, ntrials):
        return np.random.uniform(low=self._SCLR._Xmin, high=self._SCLR._Xmax, size=(ntrials, self._SCLR.dim))

class TuningObjective2(object):
    def __init__(self, *args, **kwargs):
        self._manual_sp=None;
        self._debug = kwargs["debug"] if kwargs.get("debug") is not None else False
        if type(args[0]) == str: self.mkFromFiles(*args, **kwargs)
        else:                    self.mkFromData( *args, **kwargs) # NOT implemented --- also add a mkReduced for small scale tests

    @property
    def dim(self): return self._AS.dim

    @property
    def pnames(self): return self._SCLR.pnames

    def setManualStartPoint(self, p0):
        self._manual_sp = p0

    def unsetManualStartPoint(self):
        self._manual_sp = None

    def rbox(self, ntrials):
        return self._AS.rbox(ntrials)

    def initWeights(self, fname, hnames, bnums, blows, bups):
        matchers = apprentice.weights.read_pointmatchers(fname)
        weights = []
        for hn, bnum, blow, bup in zip(hnames, bnums, blows, bups):
            pathmatch_matchers = [(m, wstr) for  m, wstr  in matchers.items()    if m.match_path(hn)]
            posmatch_matchers  = [(m, wstr) for (m, wstr) in pathmatch_matchers if m.match_pos(bnum, blow, bup)]
            w = float(posmatch_matchers[-1][1]) if posmatch_matchers else 0  # < NB. using last match
            weights.append(w)
        return np.array(weights)

    def setWeights(self, wdict):
        """
        Convenience function to update the bins weights.
        NOTE that hnames is in fact an array of strings repeating the histo name for each corresp bin
        """
        weights = []
        for hn in self._hnames: weights.append(wdict[hn])
        self._W2 = np.array([w * w for w in np.array(weights)], dtype=np.float64)

    def setLimitsAndFixed(self, fname):
        lim, fix = apprentice.io.read_limitsandfixed(fname)

        i_fix, v_fix, i_free =[], [], []
        for num, pn in enumerate(self.pnames):
            if pn in lim:
                self._bounds[num] = lim[pn]
            if pn in fix:
                i_fix.append(num)
                v_fix.append(fix[pn])
            else:
                i_free.append(num)

        self._fixIdx = (i_fix, )
        self._fixVal = v_fix
        self._freeIdx = (i_free, )


    def setAttributes(self, **kwargs):
        noiseexp = int(kwargs.get("noise_exponent")) if kwargs.get("noise_exponent") is not None else 2
        self._dim = self._AS.dim
        self._E2 = np.array([1. / e ** noiseexp for e in self._E], dtype=np.float64)
        self._SCLR = self._AS._SCLR
        self._bounds = self._SCLR.box
        self._freeIdx = ([i for i in range(self._dim)],)
        self._fixIdx = ([],)
        self._fixVal = []
        if kwargs.get("limits") is not None: self.setLimits(kwargs["limits"])
        self._debug = kwargs["debug"] if kwargs.get("debug") is not None else False

    def envelope(self):
        if hasattr(self._RA[0], 'vmin') and hasattr(self._RA[0], "vmax"):
            if self._RA[0].vmin is None or self._RA[0].vmax is None:
                return np.where(self._Y)  # use everything

            VMIN = np.array([r.vmin for r in self._RA])
            VMAX = np.array([r.vmax for r in self._RA])
            return np.where(np.logical_and(VMAX > self._Y, VMIN < self._Y))
        else:
            return np.where(self._Y)  # use everything

    def mkFromFiles(self, f_weights, f_data, f_approx, f_errors=None, **kwargs):
        AS = AppSet(f_approx)
        hnames  = [    b.split("#")[0]  for b in AS._binids]
        bnums   = [int(b.split("#")[1]) for b in AS._binids]
        blow    = [float(ra.xmin) if ra.xmin is not None else None for ra in AS._RA]
        bup     = [float(ra.xmax) if ra.xmax is not None else None for ra in AS._RA]
        weights = self.initWeights(f_weights, hnames, bnums, blow, bup)
        if sum(weights)==0:
            raise Exception("No observables selected. Check weight file and if it is compatible with experimental data supplied.")
        nonzero = np.where(weights>0)

        # Filter here to use only certain bins/histos
        dd = apprentice.io.readExpData(f_data, [str(b) for b in AS._binids[nonzero]])
        Y = np.array([dd[b][0] for b in AS._binids[nonzero]], dtype=np.float64)
        E = np.array([dd[b][1] for b in AS._binids[nonzero]], dtype=np.float64)

        # Filter for wanted bins here and get rid of division by zero in case of 0 error which is undefined behaviour
        good = []
        for num, bid in enumerate(AS._binids[nonzero]):
            if E[num] > 0:
                _num = np.where(AS._binids==bid)[0][0]
                if AS._RA[0]._scaler != AS._RA[_num]._scaler:
                    if self._debug: print("Warning, dropping bin with id {} to guarantee caching works".format(bid))
                    continue
                if not AS._RA[_num].wraps(Y[num]):
                    if self._debug: print("Warning, dropping bin with id {} as it is not wrapping the data".format(bid))
                    continue
                else:
                    pass#print("check passed")
                # check for Enveloped data
                good.append(num)
            else:
                if self._debug: print("Warning, dropping bin with id {} as its weight or error is 0. W = {}, E = {}".format(bid,weights[nonzero][num],E[num]))
        self._good = good

        if len(good)==0:
            raise Exception("No bins left after filtering.")

        # TODO This needs some re-engineering to allow fow multiple filterings
        RA =           [AS._RA[nonzero][g]     for g in good]
        self._binids = [AS._binids[nonzero][g] for g in good]
        self._AS = AppSet(RA, self._binids)
        self._E = E[good]
        self._Y = Y[good]
        self._W2 = np.array([w * w for w in np.array(weights[nonzero])[good]], dtype=np.float64)
        self._hnames = np.array([b.split("#")[0]  for b in self._binids])
        # Add in error approximations
        if f_errors is not None:
            EAS = AppSet(f_errors)
            ERA = [EAS._RA[g] for g in good]
            self._EAS=AppSet(ERA, self._binids)
        else:
            self._EAS=None
        self.setAttributes(**kwargs)

    def mkFromData(cls, AS, EAS, Y, E, W2, **kwargs):
        cls._AS = AS
        cls._EAS = EAS
        cls._Y = Y
        cls._E = E
        cls._W2 =W2
        cls._binids = AS._binids
        cls._hnames = np.array([b.split("#")[0]  for b in AS._binids])
        cls.setAttributes(**kwargs)

    def mkReduced(self, keep, **kwargs):
        AS = self._AS.mkReduced(keep, **kwargs)
        if self._EAS is not None:  EAS = self._EAS.mkReduced(keep, **kwargs)
        else:                      EAS = None
        Y = self._Y[keep]
        E = self._E[keep]
        W2 = self._W2[keep]
        return TuningObjective2(AS, EAS, Y, E, W2, **kwargs)

    def setReduced(self, keep, **kwargs):
        self._AS = self._AS.mkReduced(keep, **kwargs)
        if self._EAS is not None:  self._EAS = self._EAS.mkReduced(keep, **kwargs)
        else:                      self._EAS = None
        self._Y = self._Y[keep]
        self._E = self._E[keep]
        self._W2 = self._W2[keep]

    def mkPoint(self, _x):
        x=np.empty(self._dim, dtype=np.float64)
        x[self._fixIdx] = self._fixVal
        x[self._freeIdx] = _x
        return x

    def objective_without_surrograte_values(self,
                                            surrogate_alternate_df:pd.DataFrame,
                                            unbiased=False):
        """

                         MC                      DMC
        bin1.P        [[1,2],[3,4],[6,3]]     [[1,2],[3,4],[6,3]]
        bin1.V        [19, 18, 17]               [99, 98, 97]
        bin2.P        [[1,2],[3,4],[6,3]]     [[1,2],[3,4],[6,3]]
        bin2.V        [29, 28, 27]              [89, 88, 87]
        """
        columnnames = list(surrogate_alternate_df.index)
        rownames = list(surrogate_alternate_df.columns.values)
        obj_val = 0.
        for cnum in range(0, len(columnnames),2):
            val = surrogate_alternate_df[rownames[0]]['{}'.format(columnnames[cnum+1])]
            if self._EAS is not None:
                err = surrogate_alternate_df[rownames[1]]['{}'.format(columnnames[cnum+1])]
            else:
                err = [0.]
            term_name = columnnames[cnum].split('.')[0]
            if '#' not in term_name:
                term_name += "#1"
            if term_name in self._binids and len(val) > 0:
                ionum = self._binids.index(term_name)
                w2 = 1. if unbiased else self._W2[ionum]
                obj_val += w2 * (
                        (val[0] - self._Y[ionum]) ** 2 / (err[0] ** 2 + self._E[ionum] ** 2))
            else:
                continue
        return obj_val

    def objective(self, _x, sel=slice(None, None, None), unbiased=False):
        x=self.mkPoint(_x)
        vals = self._AS.vals(x, sel=sel)
        if self._EAS is not None:
            err2 = self._EAS.vals(x, sel=sel)**2
        else:
            err2=np.zeros_like(vals)
        if unbiased: return apprentice.tools.fast_chi(np.ones(len(vals)), self._Y[sel] - vals, 1./(err2 + 1./self._E2[sel]))
        else:        return apprentice.tools.fast_chi(self._W2[sel]     , self._Y[sel] - vals, 1./(err2 + 1./self._E2[sel]))# self._E2[sel])

    def gradient(self, _x, sel=slice(None, None, None),set_cache=False):
        x=self.mkPoint(_x)
        vals  = self._AS.vals( x, sel=sel)
        E2=1./self._E2[sel]
        grads = self._AS.grads(x, sel=sel, set_cache=set_cache)
        if self._EAS is not None:
            err   = self._EAS.vals(  x, sel=sel, set_cache=set_cache)
            egrads = self._EAS.grads( x, sel=sel, set_cache=set_cache)
        else:
            err= np.zeros_like(vals)
            egrads = np.zeros_like(grads)
        return apprentice.tools.fast_grad2(self._W2[sel], self._Y[sel] - vals, E2, err, grads, egrads)[self._freeIdx]

    def hessian(self, _x, sel=slice(None, None, None),set_cache=False):
        x=self.mkPoint(_x)
        vals  = self._AS.vals( x, sel = sel)
        grads = self._AS.grads(x, sel, set_cache=set_cache)[:,self._freeIdx].reshape(len(vals), len(_x))
        hess  = self._AS.hessians(x, sel)[:,self._freeIdx][self._freeIdx,:].reshape(len(_x),len(_x),len(vals))
        if self._EAS is not None:
            evals  = self._EAS.vals( x, sel = sel)
            egrads = self._EAS.grads(x, sel, set_cache=set_cache)[:,self._freeIdx].reshape(len(vals), len(_x))
            ehess  = self._EAS.hessians(x, sel)[:,self._freeIdx][self._freeIdx,:].reshape(len(_x),len(_x),len(vals))
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

        spans = calcSpans(np.zeros( (len(_x), len(_x), len(vals)) ), len(_x), G1,G2,H2,H3,grads,egrads)

        spans += G3*hess
        spans += H2*evals*ehess

        return np.sum( self._W2[sel]*(spans), axis=2)

    def startPoint(self, ntrials, sel=slice(None, None, None), method="lhs"):
        if self._manual_sp is not None:
            if self._debug: print("Manual start point: {}".format(self._manual_sp))
            return self._manual_sp
        if ntrials == 0:
            if self._debug: print("StartPoint: {}".format(self._SCLR.center))
            x0 =self._bounds[:,0] + 0.5*(self._bounds[:,1]-self._bounds[:,0])
            return x0[self._freeIdx]
        import numpy as np
        import time
        t0=time.time()
        if   method == "uniform":
            _PP = np.random.uniform(low=self._bounds[self._freeIdx][:,0], high=self._bounds[self._freeIdx][:,1], size=(ntrials, len(self._freeIdx[0])))
        elif method == "lhs":
            import pyDOE2
            a = self._bounds[self._freeIdx][:,0]
            b = self._bounds[self._freeIdx][:,1]
            _PP = a + (b-a) * pyDOE2.lhs(len(self._freeIdx[0]), samples=max(ntrials,2), criterion="maximin")
        else:
            raise Exception("Startpoint sampling method {} not known, exiting".format(method))

        _CH = [self.objective(p, sel=sel) for p in _PP]
        t1=time.time()
        if self._debug: print("StartPoint: {}, evaluation took {} seconds".format(_PP[_CH.index(min(_CH))], t1-t0))
        return _PP[_CH.index(min(_CH))]

    def startPointMPI(self, ntrials, sel=slice(None, None, None)):
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        XX = self.rbox(ntrials)
        rankWork = apprentice.tools.chunkIt(XX, comm.Get_size()) if rank == 0 else []
        rankWork = comm.scatter(rankWork, root=0)
        temp = [self.objective(x, sel=sel) for x in rankWork]
        ibest = np.argmin(temp)
        X = comm.gather(XX[ibest], root=0)
        FUN = comm.gather(temp[ibest], root=0)
        xbest = None
        if rank == 0:
            ibest = np.argmin(FUN)
            xbest = X[ibest]
        xbest = comm.bcast(xbest, root=0)
        return xbest

    def minimizeMPI(self,nstart=1, nrestart=1, sel=slice(None, None, None), method="tnc", tol=1e-6, saddle_point_check=True,comm = MPI_.COMM_WORLD,minimize=True):
        # comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        _res = np.zeros(nrestart, dtype=object)
        _F = np.zeros(nrestart)
        allWork = apprentice.tools.chunkIt([i for i in range(nrestart)], size)
        rankWork = comm.scatter(allWork, root=0)
        import time
        import sys
        import datetime
        t0 = time.time()
        for ii in rankWork:
            res = self.minimize(nstart=nstart,nrestart=1,sel=sel,method=method,tol=tol,
                                saddle_point_check=saddle_point_check,use_MPI_for_x0=False,minimize=minimize)
            _res[ii] = res
            _F[ii] = res["fun"]

            if rank == 0 and self._debug:
                print("[{}] {}/{}".format(rank, ii, len(rankWork)))
                now = time.time()
                tel = now - t0
                ttg = tel * (len(rankWork) - ii) / (ii + 1)
                eta = now + ttg
                eta = datetime.datetime.fromtimestamp(now + ttg)
                sys.stdout.write(
                    "[{}] {}/{} (elapsed: {:.1f}s, to go: {:.1f}s, ETA: {})\r".format(
                        rank, ii + 1, len(rankWork), tel, ttg, eta.strftime('%Y-%m-%d %H:%M:%S')))
                sys.stdout.flush()
        a = comm.gather(_res[rankWork])
        b = comm.gather(_F[rankWork])
        myreturnvalue = None
        if rank == 0:
            allWork = apprentice.tools.chunkIt([i for i in range(nrestart)], size)
            for r in range(size): _res[allWork[r]] = a[r]
            for r in range(size): _F[allWork[r]] = b[r]
            myreturnvalue = _res[np.argmin(_F)]
        myreturnvalue = comm.bcast(myreturnvalue, root=0)

        return myreturnvalue

    def minimize(self, nstart=1, nrestart=1, sel=slice(None, None, None), method="tnc", tol=1e-6,
                 minimize=True,saddle_point_check=True, use_mpi=False,use_MPI_for_x0 = False, comm=None):
        from scipy import optimize
        if not minimize: raise Exception("not implemented")
        if use_mpi: return self.minimizeMPI(nstart=nstart,nrestart=nrestart,sel=sel,method=method,tol=tol,
                                            minimize=minimize,saddle_point_check=saddle_point_check,comm=comm)
        minobj = np.Infinity
        finalres = None
        import time
        t0=time.time()
        for t in range(nrestart):
            isSaddle = True
            maxtries=10
            while (isSaddle):
                x0 = np.array(self.startPointMPI(nstart, sel=sel), dtype=np.float64) if use_MPI_for_x0 else np.array(
                            self.startPoint(nstart, sel=sel), dtype=np.float64)

                if   method=="tnc":    res = self.minimizeTNC(   x0, sel, tol=tol)
                elif method=="ncg":    res = self.minimizeNCG(   x0, sel, tol=tol)
                elif method=="trust":  res = self.minimizeTrust( x0, sel, tol=tol)
                elif method=="lbfgsb": res = self.minimizeLBFGSB(x0, sel, tol=tol)
                else: raise Exception("Unknown minimiser {}".format(method))


                isSaddle = False if not saddle_point_check else self.isSaddle(res.x)
                if isSaddle and maxtries>0:
                    if self._debug: print("Minimisation ended up in saddle point, retrying, {} tries left".format(maxtries))
                    maxtries -= 1
                elif isSaddle and maxtries==0:
                    if self._debug: print("Minimisation ended up in saddle point")
                    break

            if res["fun"] < minobj:
                minobj = res["fun"]
                finalres = res
        t1=time.time()
        if self._debug:
            print(t1-t0)
        return finalres

    def minimizeAPOSMM(self):
        def sim_f(H, persis_info, sim_specs, _):
            import time
            batch = len(H['x'])
            H_o = np.zeros(batch, dtype=sim_specs['out'])

            for i, x in enumerate(H['x']):
                H_o['f'][i] = self.objective(x)

                if 'grad' in H_o.dtype.names:
                    H_o['grad'][i] = self.gradient(x)

                if 'user' in sim_specs and 'pause_time' in sim_specs['user']:
                    time.sleep(sim_specs['user']['pause_time'])

            return H_o, persis_info

        def run_aposmm(sim_max):
            sim_specs = {'sim_f': sim_f,
                         'in': ['x'],
                         'out': [('f', float), ('grad', float, ndim)]}

            gen_out = [('x', float, ndim), ('x_on_cube', float, ndim), ('sim_id', int),
                       ('local_min', bool), ('local_pt', bool)]

            gen_specs = {'gen_f': gen_f,
                         'in': [],
                         'out': gen_out,
                         'user': {'initial_sample_size': 100,
                                  'localopt_method': 'LD_MMA',
                                  # 'opt_return_codes': [0],
                                  # 'nu': 1e-6,
                                  # 'mu': 1e-6,
                                  'xtol_rel': 1e-6,
                                  'ftol_rel': 1e-6,
                                  # 'run_max_eval':10000,
                                  # 'dist_to_bound_multiple': 0.5,
                                  'max_active_runs': 6,
                                  'lb': self._bounds[:, 0],
                                  # This is only for sampling. TAO_NM doesn't honor constraints.
                                  'ub': self._bounds[:, 1]}
                         }
            alloc_specs = {'alloc_f': alloc_f, 'out': [('given_back', bool)], 'user': {}}

            persis_info = add_unique_random_streams({}, nworkers + 1)

            exit_criteria = {'sim_max': sim_max}

            # Perform the run
            # H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
            #                             alloc_specs, libE_specs)
            return libE(sim_specs, gen_specs, exit_criteria, persis_info,
                                        alloc_specs, libE_specs)

        from libensemble.libE import libE

        import libensemble.gen_funcs
        libensemble.gen_funcs.rc.aposmm_optimizers = 'nlopt'  # scipy'#petsc'nlopt'
        from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f

        from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f
        from libensemble.tools import parse_args, add_unique_random_streams
        from time import time
        import sys, os

        nworkers, is_master, libE_specs, _ = parse_args()
        if is_master:
            start_time = time()

        if nworkers < 2:
            sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

        ndim = self.dim
        simmax = 2000
        H, persis_info, flag = run_aposmm(sim_max=simmax)
        if is_master:
            # print('[Manager]:', H[np.where(H['local_min'])]['x'])
            # print('[Manager]: Time taken =', time() - start_time, flush=True)
            # print('[Manager]:', H[np.where(H['local_min'])]['x'])
            # optimal = [[j, self.objective(j)] for j in H[np.where(H['local_min'])]['x']]
            # print('[Manager]:', optimal)
            optimalObj = []
            optimalParams = []
            for j in H[np.where(H['local_min'])]['x']:
                optimalParams.append(j)
                optimalObj.append(self.objective(j))
            minindex = int(np.argmin(optimalObj))
            ret = {
                    'x': optimalParams[minindex],
                   'fun': optimalObj[minindex],
                    'log': {'time': time() - start_time}
            }
            return ret

    def minimizeTrust(self, x0, sel=slice(None, None, None), tol=1e-6):
        from scipy import optimize
        res = optimize.minimize(
                lambda x: self.objective(x, sel=sel),
                x0,
                jac=lambda x:self.gradient(x, sel=sel),
                hess=lambda x:self.hessian(x, sel=sel),
                method="trust-exact")
        return res

    def minimizeNCG(self, x0, sel=slice(None, None, None), tol=1e-6):
        from scipy import optimize
        res = optimize.minimize(
                lambda x: self.objective(x, sel=sel),
                x0,
                jac=lambda x:self.gradient(x, sel=sel),
                hess=lambda x:self.hessian(x, sel=sel),
                method="Newton-CG")
        return res

    def minimizeTNC(self, x0, sel=slice(None, None, None), tol=1e-6):
        from scipy import optimize
        res = optimize.minimize(
                lambda x: self.objective(x, sel=sel),
                x0,
                bounds=self._bounds[self._freeIdx],
                jac=lambda x:self.gradient(x, sel=sel),
                method="TNC", tol=tol, options={'maxiter':1000, 'accuracy':tol})
        return res

    def minimizeLBFGSB(self, x0, sel=slice(None, None, None), tol=1e-6):
        from scipy import optimize
        res = optimize.minimize(
                lambda x: self.objective(x, sel=sel),
                x0,
                bounds=self._bounds[self._freeIdx],
                jac=lambda x:self.gradient(x, sel=sel),
                method="L-BFGS-B", tol=tol)
        return res

    def writeParams(self, x, fname):
        with open(fname, "w") as f:
            for pn, val in zip(self.pnames, x):
                f.write("{}\t{}\n".format(pn, val))

    def writeResult(self, x, fname, meta=None):
        with open(fname, "w") as f:
            if meta is not None:
                f.write("{}".format(meta))
            f.write("{}".format(self.printParams(x)))


    def printParams(self, x_):
        x=self.mkPoint(x_)
        slen = max((max([len(p) for p in self.pnames]), 6))
        x_aligned = dot_aligned(x)
        plen = max((max([len(p) for p in x_aligned]), 6))

        b_dn = dot_aligned(self._SCLR.box[:,0])
        b_up = dot_aligned(self._SCLR.box[:,1])
        dnlen = max((max([len(p) for p in b_dn]), 5))
        uplen = max((max([len(p) for p in b_up]), 6))

        islowbound = x==self._bounds[:,0]
        isupbound  = x==self._bounds[:,1]
        isbound = islowbound + isupbound

        isbelow = x < self._SCLR.box[:,0]
        isabove = x > self._SCLR.box[:,1]
        isoutside = isbelow + isabove

        isfixed = [i in self._fixIdx[0] for i in range(self.dim)]

        s= ""
        s+= ("#\n#{:<{slen}}\t{:<{plen}} #    COMMENT    [ {:<{dnlen}}  ...  {:<{uplen}} ]\n#\n".format(" PNAME", " PVALUE", " PLOW", " PHIGH", slen=slen, plen=plen, uplen=uplen, dnlen=dnlen))
        for pn, val, bdn, bup, isf, isb, iso in zip(self.pnames, x_aligned, b_dn, b_up, isfixed, isbound, isoutside):

            if isb and isf:
                comment = "FIX & ONBOUND"
            elif isb and not isf:
                comment="ONBOUND"
            elif not isb and isf:
                comment="FIX"
            elif iso and not isf:
                comment = "OUTSIDE"
            elif iso and isf:
                comment = "FIX & OUTSIDE"
            else:
                comment = ""

            s+= ("{:<{slen}}\t{:<{plen}} # {:<13} [ {:<{dnlen}}  ...  {:<{uplen}} ]\n".format(pn, val, comment, bdn, bup, slen=slen, plen=plen, uplen=uplen, dnlen=dnlen))
        return s

    def lineScan(self, x0, dim, npoints=100, bounds=None):
        if bounds is None:
            xmin, xmax = self._bounds[self._freeIdx][dim]
        else:
            xmin, xmax = bounds

        xcoords = list(np.linspace(xmin, xmax, npoints))
        xcoords.append(x0[dim])
        xcoords.sort()

        X = np.tile(x0, (len(xcoords),1))
        for num, x in enumerate(X):
            x[dim] = xcoords[num]
        return X

    def isSaddle(self, x):
    # if   any(x==GOF._bounds[:,0]): print("WARNING: Minimisation ended up at lower boundary")
    # elif any(x==GOF._bounds[:,1]): print("WARNING: Minimisation ended up at upper boundary")
        H=self.hessian(x)
        # Test for negative eigenvalue
        return np.sum(np.sign(np.linalg.eigvals(H))) != len(H)

    @property
    def ndf(self): return len(self) - self.dim - len(self._fixIdx[0])


    def __len__(self): return len(self._AS)
