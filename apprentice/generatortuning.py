from apprentice.leastsquares import LeastSquares

import numpy as np

class GeneratorTuning(LeastSquares):
    """

    Generator Tuning function

    """

    def __init__(self, dim, fnspace, data, errors, s_val, e_val, weights, binids, bindn, binup, **kwargs):
        """

        :param dim: parameter dimension
        :type dim: int
        :param fnspace: function space object
        :type fnspace: apprentice.space.Space
        :param data: data values
        :type data: np.array
        :param s_val: surrogate polyset or list of SurrogateModel
        :type s_val: apprentice.polyset.PolySet or list
        :param errors: data errors
        :type errors: np.array
        :param weights: weights
        :type weights: np.array
        :param e_val: surrogate polyset or list of SurrogateModel
        :type e_val: apprentice.polyset.PolySet or list
        :param binids: list of binids
        :type binids: list
        :param bindn: bin down
        :type bindn: list
        :param binup: bin up
        :type binup: list

        """
        super(GeneratorTuning, self).__init__(dim, fnspace, data, s_val, errors, weights, e_val, **kwargs) # bounds and fixed go in kwargs

        self.binids_  = binids
        self.hnames_  = np.array([b.split("#")[0]  for b in self.binids_])
        self.bindn_   = bindn
        self.binup_   = binup




    def initWeights(self, fname, hnames, bnums, blows, bups):
        """

        Initialize weights

        :param fname: read weight file
        :type fname: str
        :param hnames: observable names
        :type hnames: list
        :param bnums: bin ids
        :type bnums: list
        :param blows: lower end of bins
        :type blows: np.array
        :param bups: upper end of bins
        :type bups: lower end of bins
        :return: list of weights
        :rtype: np.array

        """
        import apprentice
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

        :param wdict: weight dictionary
        :type wdict: dict

        """
        weights = []
        #for hn in self._hnames: weights.append(wdict[hn])
        for hn in self.binids_: weights.append(wdict[hn])
        self.prf2_ = np.array([w * w for w in np.array(weights)], dtype=np.float64)

    def mkReduced(self, keep, **kwargs):
        """

        Make reduced function

        :param keep: terms to keep, true if to keep and false if not to keep
        :type keep: list
        :return: reduced tuning objective object
        :rtype: apprentice.appset.TuningObjective2
        """
        AS = self._AS.mkReduced(keep, **kwargs)
        if self._EAS is not None:  EAS = self._EAS.mkReduced(keep, **kwargs)
        else:                      EAS = None
        Y = self._Y[keep]
        E = self._E[keep]
        W2 = self._W2[keep]
        from apprentice.appset import TuningObjective2
        return TuningObjective2(AS, EAS, Y, E, W2, **kwargs)


