class FunctionalApprox(object):
    def __init__():
        self.scaler_ = ...
        self.dim_ = ..
        self.orderp_
        self.orderq_
        pass

    def setCoefficients():
        """
        Obvious
        """
        pass

    def setStructures():
        """
        Set P and Q structures
        """
        pass

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
