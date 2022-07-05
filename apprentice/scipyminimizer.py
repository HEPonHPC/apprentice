import numpy as np
from scipy import optimize
from apprentice.minimizer import Minimizer

class ScipyMinimizer(Minimizer):
    def __init__(self, function, **kwargs):
        super(ScipyMinimizer, self).__init__(function, **kwargs)

        # self.bounds_ = function.bounds
        # self.constraints_ = ()

        # linear constraints 4 slsqp
        # maximise flag -> multiplication with +1 or -1

        # set bounds method in baseclass

    def minimize(self, x0=None, nrestart=1, method="tnc", tol=1e-6):
        """
        """
        minobj = np.Infinity
        finalres = None
        import time
        t0=time.time()

        if x0 is None: startpoints = self.sample(nrestart).tolist()
        else:
            x0 = np.array(x0)
            if x0.ndim==2: startpoints = x0.tolist()
            else:          startpoints = [x0]

        print(startpoints)

        # MPI this
        for sp in startpoints:

            if   method=="tnc":    res = self.minimiseTNC(   sp, tol=tol)
            elif method=="lbfgsb": res = self.minimiseLBFGSB(sp, tol=tol)
            else: raise Exception("Unknown minimizer {}".format(method))

            if res["fun"] < minobj:
                minobj = res["fun"]
                finalres = res
        t1=time.time()
        print("Minimisation took {} seconds".format(t1-t0))
        return finalres


    def minimiseTNC(self, x0, tol=1e-6):
        from scipy import optimize
        res = optimize.minimize(
                lambda x: self.function_(x),
                x0,
                bounds=self.bounds_,
                jac=None if self.gradient_ is None else lambda x:self.gradient_(x),
                method="TNC", tol=tol, options={'maxiter':1000, 'accuracy':tol})
        return res

    def minimiseLBFGSB(self, x0, tol=1e-6):
        from scipy import optimize
        res = optimize.minimize(
                lambda x: self.function_(x),
                x0,
                bounds=self.bounds_,
                jac=None if self.gradient_ is None else lambda x:self.gradient_(x),
                method="L-BFGS-B", tol=tol)
        return res

    def minimiseSLSQP(self, x0, tol=1e-6):
        from scipy import optimize
        res = optimize.minimize(
                lambda x: self.function_(x),
                x0,
                bounds=self.bounds_,
                jac=None if self.gradient_ is None else lambda x:self.gradient_(x),
                method="SLSQP", tol=tol, constraints=self.constraints_, options={'maxiter':1000, 'accuracy':tol})
        return res

    def minimizeNCG(self, x0, sel=slice(None, None, None), tol=1e-6):
        from scipy import optimize
        res = optimize.minimize(
                lambda x: self.function_(x),
                x0,
                jac=None if self.gradient_ is  None else lambda x:self.gradient_(x, sel=sel),
                hess=None if self.hessian_ is  None else lambda x:self.hessian_(x, sel=sel),
                method="Newton-CG")
        return res

