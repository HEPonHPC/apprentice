#!/usr/bin/python3

import apprentice as app
import numpy as np


def getParams(fname, pidx, field="params"):
    import h5py
    with h5py.File(fname, "r") as f: return f.get(field)[pidx]

def getY(fname, bidx, pidx, field="values"):
    import h5py
    with h5py.File(fname, "r") as f: return f.get(field)[bidx,pidx]


if __name__ == "__main__":
    import sys
    import time

    import optparse, os, sys
    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-v", "--debug", dest="DEBUG", action="store_true", default=False, help="Turn on some debug messages")
    op.add_option("-w", dest="WEIGHTS", default=None, help="Weight file to choose observables to predict (default: %default)")
    op.add_option("-o", "--output", dest="OUTPUT", default="comparison", help="Output directory for plots (default: %default)")
    op.add_option("-p", "--paramindex", dest="PIDX", type=int, default=0, help="Parameter point index (default: %default)")
    op.add_option("--log", dest="ISLOG", action='store_true', default=False, help="input data is logarithmic --- affects how we filter (default: %default)")
    opts, args = op.parse_args()

    # Objects in approximation
    binids, RA = app.tools.readApprox(args[0])
    hnames = sorted(list(set([b.split("#")[0] for b in binids])))
    observables = [x for x in list(set(app.tools.readObs(opts.WEIGHTS))) if x in hnames] if opts.WEIGHTS is not None else hnames

    P = getParams(args[1], opts.PIDX)

    _h5ids = app.tools.readIndex(args[1])
    h5ids = np.array([_h5ids.index(x) for x in binids])
    im = app.tools.indexMap(args[1], observables)

    if not os.path.exists(opts.OUTPUT): os.makedirs(opts.OUTPUT)
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    plt.style.use('ggplot')
    for obs in observables:
        truth = getY(args[1], im[obs], opts.PIDX)
        if opts.ISLOG: truth = np.log10(truth)

        bids = [x for x in binids if x.startswith(obs)]
        bids = sorted(bids, key=lambda x: int(x.split("#")[-1]))
        pred = [RA[binids.index(x)].predict(P) if app.tools.pInBox(P, RA[binids.index(x)]._scaler.box) else None for x in bids]

        plt.title("{} param idx {}".format(obs, opts.PIDX))
        plt.scatter(range(len(pred)), pred, label="prediction")
        plt.scatter(range(len(pred)),  truth,s=10, label="truth")
        plt.legend()
        plt.savefig(os.path.join(opts.OUTPUT, "{}_pidx{}.pdf".format(obs.replace("/","_"), opts.PIDX)))
        plt.clf()
