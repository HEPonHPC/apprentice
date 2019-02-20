#!/usr/bin/env python

import apprentice
import numpy as np

def mkPlotNorm(data, f_out, norm=2):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    xi, yi, ci = [],[],[]

    for k, v in data.items():
        xi.append(k[0])
        yi.append(k[1])
        ci.append(v)

    i_winner = ci.index(min(ci))
    winner = (xi[i_winner], yi[i_winner])

    cmapname   = 'viridis'
    plt.clf()
    from matplotlib.ticker import MaxNLocator
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    mpl.rc('text', usetex = True)
    mpl.rc('font', family = 'serif', size=12)
    mpl.style.use("ggplot")

    plt.scatter(winner[0], winner[1], marker = '*', c = "magenta",s=400, alpha = 0.9)
    plt.scatter(xi, yi, marker = 'o', c = np.log10(ci), cmap = cmapname, alpha = 0.8)
    plt.xlabel("$m$")
    plt.ylabel("$n$")
    plt.xlim((min(xi)-0.5,max(xi)+0.5))
    plt.ylim((min(yi)-0.5,max(yi)+0.5))
    b=plt.colorbar()
    b.set_label("$\log_{{10}}$ L{}".format(norm))

    plt.savefig(f_out)
    plt.close('all')



def raNorm(ra, X, Y, norm=2):
    nrm = 0
    for num, x in enumerate(X):
        nrm+= abs(ra.predict(x) - Y[num])**norm
    return nrm

def raNormInf(ra, X, Y):
    nrm = 0
    for num, x in enumerate(X):
        nrm = max(nrm,abs(ra.predict(x) - Y[num]))
    return nrm

def mkBestRASIP(X, Y, pnames=None, split=0.5, norm=2, m_max=None, n_max=None, f_plot=None, seed=1234, ts = "1x"):
    """
    """
    np.random.seed(seed)
    _N, _dim = X.shape

    i_train = [i for i in range(_N)]
    i_test = [i for i in range(_N)]

    # i_train = sorted(list(np.random.choice(range(_N), int(np.ceil(split*_N)))))
    # i_test = [i for i in range(_N) if not i in i_train]

    # N_train = len(i_train)
    # N_test  = len(i_test)

    # orders = apprentice.tools.possibleOrders(N_train, _dim, mirror=True)
    # if n_max is not None: orders = [ o for o in orders if o[1] <= n_max]
    # if m_max is not None: orders = [ o for o in orders if o[0] <= m_max]


    FS = ["filter", "scipy"]
    # RS = ["ms", "baron"]
    # FS = ['filter']
    # FS = ["scipy"]
    RS = ["ms"]

    """
    Mohan: trying with unscaled dims
    """
    Xtrain = X[i_train]
    Ytrain = Y[i_train]
    scalemin = []
    scalemax = []
    for d in range(Xtrain[0].shape[0]):
        scalemin.append(-1)
        scalemax.append(1)
        # scalemin.append(min(Xtrain[:,d]))
        # scalemax.append(max(Xtrain[:,d]))

    import json
    for fs in FS:
        for rs in RS:
            rrr = apprentice.RationalApproximationSIP(Xtrain, Ytrain,
                    m=3, n=3, pnames=pnames, fitstrategy=fs, trainingscale=ts,
                    roboptstrategy=rs,scalemin=scalemin,scalemax=scalemax)
            print("Test error FS {} RS {}: 1N:{} 2N:{} InfN:{}".format(fs, rs,
                            raNorm(rrr, X[i_test], Y[i_test],1),
                            np.sqrt(raNorm(rrr, X[i_test], Y[i_test],2)),
                            raNormInf(rrr, X[i_test], Y[i_test])))
            print("Total Approximation time {}\n".format(rrr.fittime))

            with open("test2D_{}_{}.json".format(fs,rs), "w") as f: json.dump(rrr.asDict, f, indent=4)


if __name__ == "__main__":

    import os, sys
    if len(sys.argv)!=4:
        print("Usage: {} input.hf output.json".format(sys.argv[0]))
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print("Input file '{}' not found.".format(sys.argv[1]))

    # Prevent overwriting of input data
    assert(sys.argv[2]!=sys.argv[1])


    # TODO rethink data read in --- this is obviously a bit stupid
    # if rank==0:
    # This reads the data for all bins
    try:
        X,Y = apprentice.tools.readData(sys.argv[1])
    except:
        DATA = apprentice.tools.readH5(sys.argv[1], [0])
        X, Y= DATA[0]


    mkBestRASIP(X, Y, m_max=4, n_max=2, seed=int(sys.argv[2]), ts=sys.argv[3])
