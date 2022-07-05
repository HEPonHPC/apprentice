import apprentice
from apprentice.util import Util
import numpy as np

class PolySet(object):
    def __init__(self, fnspace, pcoeff, qcoeff=None, **kwargs):
        self.fnspace_ = fnspace
        self.PC_ = pcoeff
        self.QC_ = qcoeff
        self.recurrence = recurrence=apprentice.monomial.recurrence1D if self.dim==1 else apprentice.monomial.recurrence
        # TODO, there is also the recurrence2 function which is faster in some circumstances

        if self.has_rationals: self.mask_ = np.where(np.isfinite(self.QC_[:, 0]))
        # infer orders from structure of pcoeff, qcoeff to get omax
        omax = max(
                Util.num_coeffs_to_order(self.dim, pcoeff.shape[1]),
                Util.num_coeffs_to_order(self.dim, qcoeff.shape[1]) if qcoeff is not None else 0 )

        self.structure_ = np.array(apprentice.monomialStructure(self.dim, omax), dtype=np.int32)
        # Gradient helpers
        # Observe that there is always the same number of non-zero terms per dim
        self.nonzero_gradient_indices_  = np.empty( (self.dim, len(np.where(self.structure_[:, 0])[0])), dtype=int)
        for d in range(self.dim):         self.nonzero_gradient_indices_[d] = np.where(self.structure_[:, d])[0]

        self.reduced_structure_ = np.array([self.structure_[nz][:,num] for num, nz in enumerate(self.nonzero_gradient_indices_)], dtype=np.int32)
        # Hessian helpers
        self.HH_ = np.ones((self.dim, self.dim, len(self.structure_))                           , dtype=np.float64) # Prefactors
        self.EE_ = np.full((self.dim, self.dim, len(self.structure_), self.dim), self.structure_, dtype=np.int32)   # Initial structures

        for numx in range(self.dim):
            for numy in range(self.dim):
                if numx==numy:    self.HH_[numx][numy] = self.structure_[:,numx] * (self.structure_[:,numx] - 1)
                else:             self.HH_[numx][numy] = self.structure_[:,numx] *  self.structure_[:,numy]
                self.EE_[numx][numy][:,numx]-=1
                self.EE_[numx][numy][:,numy]-=1

        # Observe that there is always the same number of non-zeros per dim
        self.nonzero_hessian_indices_ = np.empty((self.dim, self.dim, len(np.where(self.HH_[0][0]>0)[0])), dtype=int)
        for numx in range(self.dim):
            for numy in range(self.dim):
                self.nonzero_hessian_indices_[numx][numy] = np.where(self.HH_[numx][numy]>0)[0]

        # Jacobians for Hessian
        JF = self.fnspace_.jacfac
        for numx in range(self.dim):
            for numy in range(self.dim):
                self.HH_[numx][numy][self.nonzero_hessian_indices_[numx][numy]] *= (JF[numx] * JF[numy])

    @property
    def has_rationals(self):
        return self.QC_ is not None

    @property
    def dim(self): return self.fnspace_.dim

    @classmethod
    def from_surrogates(cls, surr):
        """
        Create fast calculator from list of approximations. They must be of
        polynomial or rational type. Orders can be different but no the dimension.
        If hessians are to be used, the function_spaces must be the same
        """

        omax_p = -1
        omax_q = -1
        dim = surr[0].dim
        fnspace = surr[0].function_space
        # Initial checks/survey for:
        # everyone has the same dim, function_space
        # TODO here we want to check that the surrogates are either polynomials or rationals
        # NOTE it is imperative that all polynomials have the same function_space when computing the
        # hessian due to the jacobian and the way we use it
        for r in surr:
            if r.order_numerator > omax_p: omax_p = r.order_numerator
            if hasattr(r, "n") and r.order_denominator > omax_q: omax_q = r.order_denominator
            if r.function_space != fnspace:
                raise Exception("Scalers must be identical") # Implicitly checks dimension

        # Set coefficient arrays
        # Need maximum extends of coefficients
        lmax = Util.num_coeffs_poly(dim, max(omax_p, omax_q))
        PC = np.zeros((len(surr), lmax), dtype=np.float64)
        for num, r in enumerate(surr): PC[num][:r.coeff_numerator.shape[0]] = r.coeff_numerator

        # Denominator
        if omax_q > 0:
            QC = np.zeros((len(surr), lmax), dtype=np.float64)
            for num, r in enumerate(surr):
                if hasattr(r, "n"):
                    QC[num][:r.coeff_denominator.shape[0]] = r.coeff_denominator
                else:
                    QC[num][0] = None

        else:
            QC = None

        return cls(fnspace, PC, QC)

    def compute_recurrence(self, x):
        xs = self.fnspace_.scale(x)
        self.recvec_ = self.recurrence(xs, self.structure_)

    def vals(self, x, sel=slice(None, None, None), set_cache=True, maxorder=None):
        """
        Maxorder: truncate computation
        """
        if set_cache: self.compute_recurrence(x)
        if maxorder is None:
            MM = self.recvec_ * self.PC_[sel]
        else:
            nc = Util.num_coeffs_poly(self.dim, maxorder)
            MM=self.rec_vec[:nc] * self.PC[sel][:,:nc]
        vals = np.sum(MM, axis=1)
        if self.has_rationals:
            den = np.sum(self.recvec_ * self.QC_[sel], axis=1)
            vals/=den
            # FIXME this logic with the mask is not working
            # The code will divide by zero in case we hav mixed bits here
            # Note that this will go away come federations
            # vals[self._mask[sel]] /= den[self._mask[sel]]
            # Really? should that no be divi by one then?
        return vals

    def grads(self, x, sel=slice(None, None, None), set_cache=True):
        if set_cache: self.compute_recurrence(x)
        xs = self.fnspace_.scale(x)
        GREC = Util.gradient_recurrence(xs, self.structure_, self.fnspace_.jacfac, self.nonzero_gradient_indices_, self.reduced_structure_)

        Pprime = Util.prime(GREC, self.PC_[sel], self.dim, self.nonzero_gradient_indices_) # Move this to utils

        if self.has_rationals:
            P = np.atleast_2d(np.sum(self._maxrec * self._PC[sel], axis=1))
            Q = np.atleast_2d(np.sum(self._maxrec * self._QC[sel], axis=1))
            Qprime = Util.prime(GREC, self._QC[sel], self.dim, self._NNZ)
            return np.array(Pprime/Q.transpose() - (P/Q/Q).transpose()*Qprime, dtype=np.float64)

        return np.array(Pprime, dtype=np.float64)

    # @jit(forceobj=True)#, parallel=True)
    def hessians(self, x, sel=slice(None, None, None)):
        """
        To get the hessian matrix of bin number N, do
        H=hessians(pp)
        H[:,:,N]
        """
        xs = self.fnspace_.scale(x)

        NSEL = len(self.PC_[sel])

        Phess = Util.doubleprime(self.dim, xs, NSEL, self.HH_, self.nonzero_hessian_indices_, self.EE_, self.PC_[sel])

        #TODO check against autograd?
        if self.has_rationals:
            JF = self._SCLR.jacfac
            GREC = Util.gradient_recurrence(xs, self.structure_, self.fnspace_.jacfac, self.nonzero_gradient_indices_, self.reduced_structure_)
            P = np.atleast_2d(np.sum(self.recvec_ * self.PC_[sel], axis=1))
            Q = np.atleast_2d(np.sum(self.recvec_ * self.QC_[sel], axis=1))
            Pprime = np.atleast_2d(Util.prime(GREC, self.PC_[sel], self.dim, self._NNZ))
            Qprime = np.atleast_2d(Util.prime(GREC, self.QC_[sel], self.dim, self._NNZ))
            Qhess = Util.doubleprime(self.dim, xs, NSEL, self.HH_, self.nonzero_hessian_indices_, self.EE_, self.QC_[sel])

            w = Phess/Q
            for numx in range(self.dim):
                for numy in range(self.dim):
                    w[numx][numy] -= 2*(Pprime[:,numx]*Qprime[:,numy]/Q/Q).flatten()
                    w[numx][numy] += 2*(Qprime[:,numx]*Qprime[:,numy]*P/Q/Q/Q).flatten()

            w -= Qhess*(P/Q/Q)
            return w

        return Phess

