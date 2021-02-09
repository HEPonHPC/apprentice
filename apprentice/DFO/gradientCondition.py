import argparse
import apprentice
import json
from apprentice.tools import TuningObjective
class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass

def readApprox(fname):
    import json
    import apprentice
    with open(fname) as f: rd = json.load(f)
    rd = json.load(open(fname))
    binids = sorted(rd.keys())
    if "__xmin" in binids:
        binids.remove("__xmin")
    if "__xmax" in binids:
        binids.remove("__xmax")
    for b in binids:
        if "qcoeff" not in rd[b]:
            rd[b]["qcoeff"] = [1.0]
            rd[b]["n"] = 0

    RA = [apprentice.RationalApproximation(initDict=rd[b]) for b in binids]
    return binids, RA

def readExpData(fname, binids):
    import json
    import numpy as np
    with open(fname) as f: dd = json.load(f)
    Y = np.array([dd[b][0] for b in binids])
    E = np.array([dd[b][1] for b in binids])
    return Y, E

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate sample points',
                                     formatter_class=SaneFormatter)
    parser.add_argument("-a", dest="ALGOPARAMS", type=str, default=None,
                        help="Algorithm Parameters JSON")
    parser.add_argument("-m", dest="MODEL", type=str, default=None,
                        help="Model File")
    parser.add_argument("-d", dest="EXPDATA", type=str, default=None,
                        help="experimental data file")

    args = parser.parse_args()
    with open(args.ALGOPARAMS,'r') as f:
        ds = json.load(f)

    tr_center = ds['tr']['center']
    tr_radius = ds['tr']['radius']
    sigma = ds['tr']['sigma_temp']
    binids, RA = readApprox(args.MODEL)
    Y,E = readExpData(args.EXPDATA,[str(b) for b in  binids])
    import numpy as np

    good = np.where(E > 0)

    Y = Y[good]
    E = E[good]
    binids = [binids[g] for g in good[0]]
    RA = [RA[g] for g in good[0]]

    SCLR = RA[0]._scaler
    IO = TuningObjective(RA, Y, E, np.ones(len(binids)), binids, cache_recursions=True)

    grad = IO.gradient(tr_center)
    if np.linalg.norm(grad) <= sigma * tr_radius:
        print("the green condition says YES")
    else: print("the green condition says NO")




