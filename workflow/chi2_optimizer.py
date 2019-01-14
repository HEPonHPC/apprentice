#!/usr/bin/env python

import json
import apprentice
import numpy as np

def readApprox(fname):
    with open(fname) as f: rd = json.load(f)
    binids = sorted(rd.keys())
    RA = [apprentice.RationalApproximation(initDict=rd[b]) for b in binids]
    return binids, RA

def readExpData(fname, binids):
    with open(fname) as f: dd = json.load(f)
    Y = np.array([dd[b][0] for b in binids])
    E = np.array([dd[b][1] for b in binids])
    return Y, E

def mkCov(yerrs):
    return np.atleast_2d(yerrs).T * np.atleast_2d(yerrs) * np.eye(yerrs.shape[0])

if __name__ == "__main__":
    import sys

    # At the moment, the object selection for the chi2 minimisation is
    # simply everything that is in the approximation file
    binids, RA = readApprox(sys.argv[1])
    Y, E = readExpData(sys.argv[2], [str(b) for b in  binids])
    E2 = [e**2 for e in E]

    S = apprentice.Scaler("{}.scaler".format(sys.argv[1]))

    # NOTE: all evaluations are happening in the scaled world

    def chi2(x):
        return sum([ (Y[i] - RA[i](x))**2/E2[i] for i in range(len(binids))])


    from scipy import optimize
    res = optimize.minimize(chi2, S.center, bounds=S.sbox)
    print("Minimum found at {}".format(S(res["x"], unscale=True)))


    # Now do some more universes
    NSAMPLES = 1000
    COV = mkCov(E)
    import scipy.stats as st

    # Here we draw samples using the Covariance matrix above
    mn = st.multivariate_normal(Y, COV)
    sampledyvals = mn.rvs(size=NSAMPLES)

    def chi2_smeared(x, V):
        return sum([ (V[i] - RA[i](x))**2/(2*E2[i]) for i in range(len(binids))])

    res_smeared = []
    for num, v in enumerate(sampledyvals):
        _r = optimize.minimize( lambda x:chi2_smeared(x, v), S.center, bounds=S.sbox)
        res_smeared.append(_r)
        if (num+1)%10 == 0: print("Done with {}/{}".format(num+1, NSAMPLES))

    P = [list(S(x["x"], unscale=True))   for x in res_smeared]
    P.append(list(S(res["x"], unscale=True)))
    F = [x["fun"] for x in res_smeared]
    F.append(res["fun"])

    with open("{}.minimization".format(sys.argv[1]), "w") as f:
        json.dump({ "x": P, "fun" : F}, f)
