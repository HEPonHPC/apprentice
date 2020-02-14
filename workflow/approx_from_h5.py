#!/usr/bin/env python

import h5py
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

def mkBestRA(X,Y, pnames, split=0.7, norm=2, m_max=5, n_max=None, f_plot=None):
    """
    """
    _N, _dim = X.shape

    i_train = sorted(list(np.random.choice(range(_N), int(np.ceil(split*_N)))))
    i_test = [i for i in range(_N) if not i in i_train]

    N_train = len(i_train)
    N_test  = len(i_test)

    orders = apprentice.tools.possibleOrders(N_train, _dim, mirror=True)
    if n_max is not None: orders = [ o for o in orders if o[1] <= n_max and o[0]<=m_max]


    # d_RA   = { o : apprentice.RationalApproximation(X[i_train], Y[i_train], order=o, pnames=pnames) for o in orders }
    d_RA   = {}
    for o in orders:
        if o[1] ==0:
            d_RA[o] = apprentice.PolynomialApproximation(X[i_train], Y[i_train], order=o[0], pnames=pnames)
        else:
            d_RA[o] = apprentice.RationalApproximation(X[i_train], Y[i_train], order=o, pnames=pnames)
    d_norm = { o : raNorm(d_RA[o], X[i_test], Y[i_test]) for o in orders }
    import operator
    sorted_norm = sorted(d_norm.items(), key=operator.itemgetter(1))
    if f_plot is not None: mkPlotNorm(d_norm, f_plot, norm)
    winner = sorted_norm[0]
    print("Winner: m={} n={} with L2={}".format(*winner[0], winner[1]))
    return apprentice.RationalApproximation(X, Y, order=winner[0], pnames=pnames)

if __name__ == "__main__":

    import os, sys
    if len(sys.argv)!=3:
        print("Usage: {} input.hf output.json".format(sys.argv[0]))
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print("Input file '{}' not found.".format(sys.argv[1]))

    # Prevent overwriting of input data
    assert(sys.argv[2]!=sys.argv[1])

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # TODO rethink data read in --- this is obviously a bit stupid
    # if rank==0:
    # This reads the data for all bins
    import time
    t1=time.time()
    DATA = apprentice.tools.readH5(sys.argv[1], [])
    pnames = apprentice.tools.readPnamesH5(sys.argv[1], xfield="params")
    idx = [i for i in range(len(DATA))]
    # else:
        # DATA=None
        # pnames=None
        # idx=None
    # DATA   = comm.bcast(DATA, root=0)
    # pnames = comm.bcast(DATA, root=0)
    # idx = comm.bcast(DATA, root=0)

    # # This reads the data for a selection of bins
    # idx = [0,1,2,5,7,8,9,14,20,44]
    # DATA = apprentice.tools.readH5(sys.argv[1], idx)

    # Note: the idx is needed to make a connection to experimental data
    with h5py.File(sys.argv[1], "r") as f:  binids = [s.decode() for s in f.get("index")[idx]]
    t2=time.time()
    print("Data preparation took {} seconds".format(t2-t1))


    ras = []
    scl = []
    t1=time.time()
    for num, (X, Y) in  enumerate(DATA):
        # t11=time.time()
        # ras.append(mkBestRA(X,Y, pnames, n_max=3))#, f_plot="{}.pdf".format(binids[num].replace("/","_"))))
        # t22=time.time()
        # print("Approximation {}/{} took {} seconds".format(num+1,len(binids), t22-t11))
        try:
            ras.append(apprentice.RationalApproximation(X, Y, order=(2,0), pnames=pnames))
        except Exception as e:
            print("Problem with {}".format(binids[num]))
            pass

    t2=time.time()
    print("Approximation took {} seconds".format(t2-t1))
    # This reads the unique identifiers of the bins

    # jsonify # The decode deals with the conversion of byte string atributes to utf-8
    JD = { x : y.asDict for x, y in zip(binids, ras) }

    import json
    with open(sys.argv[2], "w") as f: json.dump(JD, f, indent=4)

    print("Done --- approximation of {} objects written to {}".format(len(idx), sys.argv[2]))
