from apprentice.util import Util
from apprentice.function import Function
from apprentice.polyset import PolySet
import numpy as np

class LeastSquares(Function):
    """
    sum a^2*(f(x) - d)^2 / (s^2 + e(x)^2)
    """
    def __init__(self, dim, fnspace, data, s_val, errors, prefactors, e_val=None, **kwargs):
        super(LeastSquares, self).__init__(dim, fnspace, **kwargs) # bounds and fixed go in kwargs

        self.data_         = data
        self.err2_         = np.array(errors)**2
        self.prf2_         = np.array(prefactors)**2
        if hasattr(s_val, "vals"):
            self.vals     = s_val.vals
            self.grads    = s_val.grads
            self.hessians = s_val.hessians
            self.hessians_need_transpose=False
        else:
            self.vals     = lambda x: [v.f_x(x)      for v in s_val]
            self.grads    = lambda x: [v.gradient(x) for v in s_val]
            self.hessians = lambda x: [v.hessian(x)  for v in s_val]
            self.hessians_need_transpose=True

        if e_val is not None:
            if hasattr(e_val, "vals"):
                self.evals     = e_val.vals
                self.egrads    = e_val.grads
                self.ehessians = e_val.hessians
            else:
                self.evals     = lambda x: [v.f_x(x)      for v in e_val]
                self.egrads    = lambda x: [v.gradient(x) for v in e_val]
                self.ehessians = lambda x: [v.hessian(x)  for v in e_val]
        else:
            self.evals     = None

    def objective(self, x):
        nom   = self.prf2_ * ( np.array( self.vals(x) ) - self.data_ )**2
        denom = 1*self.err2_
        if self.evals is not None:  denom += np.array( self.evals(x) )**2
        return np.sum(nom/denom)


    def gradient(self, x):
        self.currpoint_[self.free_indices_] = x # This is hidden when using the __call__ operator of the base class for objective
        vals  = self.vals(self.currpoint_)
        grads = self.grads(self.currpoint_)
        if self.evals is not None:
            err    = np.array(self.evals(self.currpoint_))
            egrads = self.egrads(self.currpoint_)
            errterm = 1./(self.err2_ + err**2)
        else:
            errterm = 1./(self.err2_)

        d  = self.data_ - vals
        v1 = -2 * self.prf2_ * d     * errterm

        gr = np.sum(grads * v1[:,None], axis=0)

        if self.evals is not None:
            v2 = +2 * self.prf2_ * d * d * errterm*errterm * err   # NOTE massive thx to JT for the -
            gr -= np.sum(egrads * v2[:,None], axis=0)

        return gr[self.free_indices_]

    def hessian(self, x):
        self.currpoint_[self.free_indices_] = x # This is hidden when using the __call__ operator of the base class for objective
        vals  = np.array(self.vals(self.currpoint_))
        grads = np.array(self.grads(self.currpoint_))[:,self.free_indices_].reshape(len(vals), len(x))
        if self.hessians_need_transpose: protohess = np.transpose(np.array(self.hessians(self.currpoint_)), (1,2,0))
        else                           : protohess =                       self.hessians(self.currpoint_)
        hess  = protohess[:,self.free_indices_][self.free_indices_,:].reshape(len(x), len(x),len(vals))

        if self.evals is not None:
            err    = np.array(self.evals(self.currpoint_))
            egrads = np.array(self.egrads(self.currpoint_))[:,self.free_indices_].reshape(len(vals), len(x))

            if self.hessians_need_transpose: protoehess = np.transpose(np.array(self.ehessians(self.currpoint_)), (1,2,0))
            else                           : protoehess =                       self.ehessians(self.currpoint_)
            ehess  = protoehess[:,self.free_indices_][self.free_indices_,:].reshape(len(x), len(x),len(vals))
        else:
            err    = np.zeros_like(vals)
            egrads = np.zeros_like(grads)
            ehess  = np.zeros_like(hess)

        # Some useful definitions
        E2=1./self.err2_
        lbd = E2 + err*err
        kap = vals - self.data_
        G1 = 2./lbd
        G2 = -4*kap*err/lbd/lbd
        G3 =  2*kap/lbd

        H2 = -2 * kap*kap/lbd/lbd
        H3 = -2 * err * kap /lbd * G2

        spans = Util.calcSpans(np.zeros( (len(x), len(x), len(vals)) ), len(x), G1, G2, H2, H3, grads, egrads)

        spans += G3*hess
        spans += H2*err*ehess

        return np.sum( self.prf2_ *(spans), axis=2)
