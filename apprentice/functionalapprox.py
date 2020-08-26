import apprentice
import numpy as np

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

    def setCoefficients(self, pcoeff, qcoeff=None):
        """
        Obvious
        """
        self.pcoeff_ = pcoeff
        self.qcoeff_ = qcoeff

    def setStructure(self):
        """
        Set P and Q structures
        """
        self.structure_ = np.array(apprentice.monomialStructure(self.dim_, max(self.orderp_, self.orderq_)), dtype=np.int32)

    def setScaler(self, sdict):
        self.scaler_ = apprentice.Scaler(sdict)

    def setRecurrence(self, x):
        if self.scaler_ is not None:
            xs = self.scaler_.scale(x)
        else:
            xs = x
        self.currec_ = np.prod(np.power(xs, self.structure_[:, np.newaxis]), axis=2)

    def vals(self, x, sel=slice(None, None, None), set_recurrence=True):

        """
        Evaluation of the numer poly at many points X.
        """

        if set_recurrence:
            self.setRecurrence(x)

        TT = self.currec_.T * self.pcoeff_[sel][:, np.newaxis]
        vals = np.sum(TT, axis=2)


        # if self._hasRationals:
            # den = np.sum(self._maxrec * self._QC[sel], axis=1)
            # vals[self._mask[sel]] /= den[self._mask[sel]]
        return vals

    def val(X, sel=slice(None, None, None)):
        """
        If x is single point -> array of values for all bins
        If x is collection of points ->
        """
        pass

    def grad(X, sel=slice(None, None, None)):
        """
        If x is single point -> array of gradients for all bins
        If x is collection of points -> all sorts of gradients etc
        """
        pass

    def hess(X, sel=slice(None, None, None)):
        """
        If x is single point -> array of hessians for all bins
        If x is collection of points -> all sorts of hessians etc
        """
        pass
