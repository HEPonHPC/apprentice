import apprentice
import numpy as np
from numba import jit

@jit
def gradientRecurrence(X, struct, jacfac, NNZ, sred):
    """
    X ... scaled point
    struct ... polynomial structure
    jacfac ... jacobian factor
    NNZ  ... list of np.where results
    sred ... reduced structure
    returns array suitable for multiplication with coefficient vector
    """
    dim = len(X)
    REC = np.zeros((dim, len(struct)))
    _RR = intpower(X, struct)
    nelem = len(sred[0])

    for coord, nz in enumerate(NNZ):
        RR = _RR[nz]
        RR[:, coord] = jacfac[coord] * sred[coord] *_RR[:nelem, coord]
        # reduction
        temp = np.ones(nelem)
        for k in range(nelem):
            for l in range(dim):
                temp[k] *= RR[k][l]
        REC[coord][nz] = temp

    return REC

def gradientRecurrenceMulti(X, struct, jacfac, NNZ, sred):
    """
    X ... np array  of scaled points
    struct ... polynomial structure
    jacfac ... jacobian factor
    NNZ  ... list of np.where results
    sred ... reduced structure
    returns array suitable for multiplication with coefficient vector
    """

    nelem = len(sred[0])

    m_RR = np.power(X[:,np.newaxis], struct)
    WW = m_RR[:,NNZ].reshape(*X.shape, nelem, X.shape[1])

    RREC = np.zeros((*X.shape, len(struct)))

    for coord, nz in enumerate(NNZ):
        a = jacfac[coord] * sred[coord] * m_RR[:,:nelem,coord]
        RR = WW[:,coord]
        RR[:,:,coord] = a
        RREC[:,coord][:,nz] = np.prod(RR,axis=2)#.reshape((X.shape[0], 1, nelem))

    return RREC



# This is the explicit triple loop version
@jit
def prime(GREC, COEFF, dim, NNZ):
    ret = np.zeros((len(COEFF), dim))
    for i in range(dim):
        for j in range(len(COEFF)):
            for k in NNZ[i]:
                ret[j][i] += COEFF[j][k] * GREC[i][k]
    return ret

@jit
def doubleprime(dim, xs, NSEL, HH, HNONZ, EE, COEFF):
    ret = np.zeros((dim, dim, NSEL), dtype=np.float64)
    for numx in range(dim):
        for numy in range(dim):
            rec = HH[numx][numy][HNONZ[numx][numy]] * hreduction(xs, EE[numx][numy][HNONZ[numx][numy]])
            if numy>=numx:
                ret[numx][numy] = np.sum((rec*COEFF[:,HNONZ[numx][numy]]), axis=1)
            else:
                ret[numx][numy] = ret[numy][numx]

    return ret

@jit(nopython=True)
def hreduction(xs, ee):
    dim =len(xs)
    nel = len(ee)
    ret = np.ones(nel)
    for n in range(nel):
        for d in range(dim):
            if ee[n][d] == 0: continue
            if ee[n][d] == 1:
                ret[n] *= xs[d]
            else:
                ret[n] *= pow(xs[d], ee[n][d])
    return ret

@jit
def intpower(xs, ee):
    dim =len(xs)
    nel = len(ee)
    ret = np.ones((nel,dim))
    for n in range(nel):
        for d in range(dim):
            if ee[n][d] == 0: continue
            if ee[n][d] == 1:
                ret[n][d] = xs[d]
            else:
                ret[n][d] = pow(xs[d], ee[n][d])
    return ret


from numba import jit
@jit
def jval(rec, pc):
    nitems = len(pc)
    nterms = len(rec)
    ret = np.zeros(nitems)
    for i in range(nitems):
        for j in range(nterms):
            ret[i] += rec[j] * pc[i][j]
    return ret

class FunctionalApprox(object):
    def __init__(self, ndim, pcoeff=None, qcoeff=None, m=0, n=0):
        self.scaler_ = None
        self.dim_ = ndim
        self.orderp_ = m
        self.orderq_ = n
        self.currec_ = None

        if self.dim_ == 1: self.recurrence = apprentice.monomial.recurrence1D
        else:              self.recurrence = apprentice.monomial.recurrence

        self.setCoefficients(pcoeff, qcoeff)
        self.setStructure()

    @property
    def dim(self): return self.dim_

    def setCoefficients(self, pcoeff, qcoeff=None):
        """
        Obvious
        """
        self.pcoeff_ = pcoeff
        self.qcoeff_ = qcoeff

    def setStructure(self):
        """
        Monomial structures for evaluation of values and gradients
        """
        self.structure_      = np.array(apprentice.monomialStructure(self.dim_, max(self.orderp_, self.orderq_)), dtype=np.int32)

        nnn = len(np.where(self.structure_[:, 0])[0])
        self.nonzerostruct_  = np.empty((self.dim, nnn), dtype=int)
        for d in range(self.dim):
            self.nonzerostruct_[d] = np.where(self.structure_[:, d])[0]

        self.reducedstruct_  = np.array([self.structure_[nz][:,num] for num, nz in enumerate(self.nonzerostruct_)], dtype=np.int32)


    def setStructureForHessians(self):
        """
        Monomial structures for evaluation of hessians
        """

        # Hessian helpers
        self.HH_ = np.ones((self.dim, self.dim, len(self.structure_))                           , dtype=np.float64) # Prefactors
        self.EE_ = np.full((self.dim, self.dim, len(self.structure_), self.dim), self.structure_, dtype=np.int32) # Initial structures

        # This is the differentiation by coordinate
        for numx in range(self.dim):
            for numy in range(self.dim):
                if numx==numy:
                    self.HH_[numx][numy] = self.structure_[:,numx] * (self.structure_[:,numx]-1)
                else:
                    self.HH_[numx][numy] = self.structure_[:,numx] *  self.structure_[:,numy]
                self.EE_[numx][numy][:,numx]-=1
                self.EE_[numx][numy][:,numy]-=1

        # Observe that there is always the same number of no-zeros per dim
        nnn = len(np.where(self.HH_[0][0]>0)[0])
        self.HNONZ_  = np.empty((self.dim, self.dim, nnn), dtype=int)
        for numx in range(self.dim):
            for numy in range(self.dim):
                self.HNONZ_[numx][numy] = np.where(self.HH_[numx][numy]>0)[0]

        # Jacobians for Hessian
        JF = self.scaler_.jacfac
        for numx in range(self.dim):
            for numy in range(self.dim):
                self.HH_[numx][numy][self.HNONZ_[numx][numy]] *= (JF[numx] * JF[numy])


    def setScaler(self, sdict):
        self.scaler_ = apprentice.Scaler(sdict)

    def setRecurrence(self, x):
        if self.scaler_ is not None:
            xs = np.array(self.scaler_.scale(x))
        else:
            xs = np.array(x)
        self.currec_ = hreduction(xs, self.structure_)

    def setRecurrences(self, x):
        if self.scaler_ is not None:
            xs = np.array(self.scaler_.scale(x))
        else:
            xs = np.array(x)
        self.currec_ = np.prod(np.power(xs, self.structure_[:, np.newaxis]), axis=2).T

    def val(self, x, sel=slice(None, None, None), set_recurrence=True):
        """
        Evaluation of the numerator and denominator polynomials at one  points x.
        """

        if set_recurrence:
            self.setRecurrence(x)

        vals = jval(self.currec_, self.pcoeff_[sel])

        if self.qcoeff_ is not None:
            qvals = jval(self.currec_, self.qcoeff_[sel])
            vals/=qvals

        return vals

    def vals(self, x, sel=slice(None, None, None), set_recurrence=True):
        """
        Evaluation of the numerator and denominator polynomials at many points x.
        """

        if set_recurrence:
            self.setRecurrences(x)

        PV = self.currec_ * self.pcoeff_[sel][:, np.newaxis]
        vals = np.sum(PV, axis=2)

        if self.qcoeff_ is not None:
            QV = self.currec_ * self.qcoeff_[sel][:, np.newaxis]
            qvals = np.sum(QV, axis=2)
            vals/=qvals

        return vals

    def grad(self, x, sel=slice(None, None, None), set_recurrence=True):
        """
        If x is single point -> array of gradients for all bins
        If x is collection of points -> all sorts of gradients etc
        """
        x=np.array(x)
        if set_recurrence: self.setRecurrence(x)
        xs = self.scaler_.scale(x)
        GREC = gradientRecurrence(xs, self.structure_, self.scaler_.jacfac, self.nonzerostruct_, self.reducedstruct_)

        Pprime = prime(GREC, self.pcoeff_[sel], self.dim_, self.nonzerostruct_)

        # if self._hasRationals:
            # P = np.atleast_2d(np.sum(self._maxrec * self._PC[sel], axis=1))
            # Q = np.atleast_2d(np.sum(self._maxrec * self._QC[sel], axis=1))
            # Qprime = prime(GREC, self._QC[sel], self.dim, self._NNZ)
            # return np.array(Pprime/Q.transpose() - (P/Q/Q).transpose()*Qprime, dtype=np.float64)

        return Pprime

    def hess(self, x,sel=slice(None, None, None)):
        """
        If x is single point -> array of hessians for all bins
        If x is collection of points -> all sorts of hessians etc
        """
        """
        To get the hessian matrix of bin number N, do
        H=hessians(pp)
        H[:,:,N]
        """
        xs = self.scaler_.scale(x)

        NSEL = len(self.pcoeff_[sel])

        Phess = doubleprime(self.dim, xs, NSEL, self.HH_, self.HNONZ_, self.EE_, self.pcoeff_[sel])

        return Phess



def readFunctionalApprox(fname):
    binids, RA = apprentice.io.readApprox(fname, set_structures=False)
    dim = RA[0].dim
    m = RA[0].m
    n=0 if not hasattr(RA[0], "n") else RA[0].n

    sdict = RA[0]._scaler.asDict
    p_nc = apprentice.tools.numCoeffsPoly(dim, m)
    q_nc = 0 if n==0 else apprentice.tools.numCoeffsPoly(dim, n)

    pcoeff = np.zeros((len(RA), max(p_nc, q_nc)), dtype=np.float64)
    for num, r in enumerate(RA): pcoeff[num][:p_nc] = r._pcoeff
    # Denominator
    if n > 0:
        qcoeff = np.zeros((len(RA), max(p_nc, q_nc)), dtype=np.float64)
        for num, r in enumerate(RA):
            self._QC[num][:q_nc] = r._qcoeff
    else:
        qcoeff = None

    ft = apprentice.FunctionalApprox(dim, m=m, n=n)
    ft.setCoefficients(pcoeff, qcoeff)
    ft.setScaler(sdict)
    ft.ids_ = binids
    return ft
