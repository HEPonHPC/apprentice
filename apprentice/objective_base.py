import apprentice
import numpy as np

class ObjectiveBase(object):
    """
    Base class for objective functions
    """
    def __init__(self, *args, **kwargs):
        self.name_ =  "Objective Base class"
        pass

    def minimize(self, *args, **kwargs):
        pass

    def objective(self, *args, **kwargs):
        pass

    @property
    def ndf(self):
        pass

    def __str__(self): return self.name_

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

    def initWeights(self, fname, hnames, bnums):
        matchers = apprentice.weights.read_pointmatchers(fname)
        weights = []
        for hn, bnum in zip(hnames, bnums):
            pathmatch_matchers = [(m, wstr) for  m, wstr  in matchers.items()    if m.match_path(hn)]
            posmatch_matchers  = [(m, wstr) for (m, wstr) in pathmatch_matchers if m.match_pos(bnum)]
            w = float(posmatch_matchers[-1][1]) if posmatch_matchers else 0  # < NB. using last match
            weights.append(w)
        return np.array(weights)
