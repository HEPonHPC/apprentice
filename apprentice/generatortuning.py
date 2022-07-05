from apprentice.leastsquares import LeastSquares

import numpy as np

class GeneratorTuning(LeastSquares):

    def __init__(self, dim, fnspace, data, errors, s_val, e_val, weights, binids, bindn, binup, **kwargs):

        super(GeneratorTuning, self).__init__(dim, fnspace, data, s_val, errors, weights, e_val, **kwargs) # bounds and fixed go in kwargs

        self.binids_  = binids
        self.hnames_  = np.array([b.split("#")[0]  for b in self.binids_])
        self.bindn_   = bindn
        self.binup_   = binup


    @classmethod
    def from_bla(cls):
        pass

    def set_weights(self):
        pass

    # hnames


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

    def mkReduced(self, keep, **kwargs):
        AS = self._AS.mkReduced(keep, **kwargs)
        if self._EAS is not None:  EAS = self._EAS.mkReduced(keep, **kwargs)
        else:                      EAS = None
        Y = self._Y[keep]
        E = self._E[keep]
        W2 = self._W2[keep]
        return TuningObjective2(AS, EAS, Y, E, W2, **kwargs)


