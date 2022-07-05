import apprentice
import numpy as np

class TuningObjective(apprentice.ObjectiveBase):
    def __init__(self, *args, **kwargs):
        super(TuningObjective, self).__init__(*args, **kwargs)

        # self.predictors_ = kwargs["predictors"] if kwargs.get("predictors") is not None else None
        # if self.predictors_ is None:
            # raise Exception("No predictors given!")

        self.debug_ = False
        # self.nbins_ = len(self.predictors_)
        self.mkFromFiles(*args, **kwargs)


    def mkFromFiles(self, f_weights, f_data, f_approx, f_errors=None, **kwargs):
        """
        No actual filtering happening here
        """

        # This loader first reads everything, then discards things with 0 weight
        FT = apprentice.functionalapprox.readFunctionalApprox(f_approx)
        hnames  = [    b.split("#")[0]  for b in FT.ids_]
        bnums   = [int(b.split("#")[1]) for b in FT.ids_]
        weights = self.initWeights(f_weights, hnames, bnums)
        if sum(weights)==0:
            raise Exception("No observables selected. Check weight file and if it is compatible with experimental data supplied.")
        self.nonzero_ = np.where(weights>0)
        FT.pcoeff_ = FT.pcoeff_[self.nonzero_]
        if FT.qcoeff_ is not None: FT.qcoeff_ = FT.qcoeff_[self.nonzero_]
        FT.ids_ = list(np.array(FT.ids_)[self.nonzero_])
        weights = weights[self.nonzero_]

        # from IPython import embed
        # embed()

        dd = apprentice.io.readExpData(f_data, [str(b) for b in FT.ids_])
        Y = np.array([dd[b][0] for b in FT.ids_], dtype=np.float64)
        E = np.array([dd[b][1] for b in FT.ids_], dtype=np.float64)

        # Filter for wanted bins here and get rid of division by zero in case of 0 error which is undefined behaviour
        good = []
        for num, bid in enumerate(FT.ids_):
            if E[num] > 0: good.append(num)
            else:
                if self.debug_: print("Warning, dropping bin with id {} as its weight or error is 0. W = {}, E = {}".format(bid,weights[num],E[num]))
        self.good_ = good

        if len(good)==0:
            raise Exception("No bins left after filtering.")

        self.ids_ = FT.ids_
        self.FT_ = FT
        self.E_ = E
        self.Y_ = Y
        self.W2_ = np.array([w * w for w in weights], dtype=np.float64)
        self.hnames_ = np.array([b.split("#")[0]  for b in FT.ids_])
        # Add in error approximations
        if f_errors is not None:
            # TODO make the reduction thing (nonzero) a function
            FTerr = apprentice.functionalapprox.readFunctionalApprox(f_errors)
            self.FTerr_ = FTerr
            # TODO maybe sanity check on binids
        else:
            self.FTerr = None
        self.setAttributes(**kwargs)

    def setAttributes(self, **kwargs):
        self.dim_ = self.FT_.dim
        self.E2_ = np.array([1. / e ** 2 if num in self.good_ else 0 for num, e in enumerate(self.E_) ], dtype=np.float64)
        self.bounds_ = self.FT_.scaler_.box
        self._freeIdx = ([i for i in range(self.dim_)],)
        self._fixIdx = ([],)
        self._fixVal = []
        if kwargs.get("limits") is not None: self.setLimits(kwargs["limits"])
        self._debug = kwargs["debug"] if kwargs.get("debug") is not None else False

    def mkPoint(self, _x):
        x=np.zeros(self.dim_, dtype=np.float64)
        x[self._fixIdx] = self._fixVal
        x[self._freeIdx] = _x
        return x

    def objective(self, _x, sel=slice(None, None, None), unbiased=False):
        x=self.mkPoint(_x)
        vals = self.FT_.val(x, sel=sel)
        return apprentice.tools.fast_chi(self.W2_[sel]     , self.Y_[sel] - vals,self.E2_[sel])
        # from IPython import embed
        # embed()
        # if self._EAS is not None:
            # err2 = self._EAS.vals(x, sel=sel)**2
        # else:
            # err2=np.zeros_like(vals)
        # if unbiased: return apprentice.tools.fast_chi(np.ones(len(vals)), self._Y[sel] - vals, 1./(err2 + 1./self._E2[sel]))
        # else:        return apprentice.tools.fast_chi(self.W2_[sel]     , self.Y_[sel] - vals, 1./(err2 + 1./self._E2[sel]))# self._E2[sel])


    def __call__(self, x):
        """
        This is the most general version. We are not exploiting any of the common computations here.
        """
        return [p.predict(x) for p in self.predictors_]

