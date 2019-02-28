#!/usr/bin/env python

import apprentice
import numpy as np


# Very slow for many datapoints.  Fastest for many costs, most readable
# https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
def is_pareto_efficient_dumb(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
    return is_efficient

def mkPlotParetoSquare(data, f_out):
    """
    Awesome
    """

    # Data preparation
    pareto_orders = []

    mMax = np.max(data[:,0])
    nMax = np.max(data[:,1])
    vMax = np.max(data[:,2])

    pdists = []
    p3D = is_pareto_efficient_dumb(data)

    vmin=1e99
    i_winner=-1
    for num, (m,n,v,a,b) in enumerate(data):
        if p3D[num]:
            # This is the old approach of using the area
            nc = m+n
            test = v*(m+n)
            if test < vmin:
                vmin=test
                i_winner=num

            # This is the euclidian distance which does not work well
            # if v<vmin:
                # vmin=v
                # i_winner=num
            # pareto_orders.append((a,b))
            # pdists.append(np.sqrt(m*m/mMax/mMax + n*n/nMax/nMax + v*v/vMax/vMax))
            # pdists.append(np.sqrt(m*m + n*n + v*v))

    # i_winner = pdists.index(min(pdists))
    # winner = pareto_orders[i_winner]
    # winner = pareto_orders[i_winner]
    winner = (data[i_winner][3], data[i_winner][4])
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rc('text', usetex = True)
    mpl.rc('font', family = 'serif', size=12)
    mpl.style.use("ggplot")
    # plt.clf()
    cmapname   = 'viridis'
    from matplotlib.ticker import MaxNLocator
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.scatter(winner[0], winner[1], marker = '*', c = "magenta",s=400, alpha = 1)

    d_pareto = data[p3D]
    d_other = data[np.logical_not(p3D)]
    plt.scatter(d_pareto[:,3], d_pareto[:,4], marker = '*', s=200, c = np.log10(d_pareto[:,2]), cmap = cmapname, alpha = 1.0)
    plt.scatter(d_other[:,3],   d_other[:,4], marker = 's', c = np.log10(d_other[:,2]), cmap = cmapname, alpha = 1.0)
    plt.xlabel("$m$")
    plt.ylabel("$n$")
    plt.xlim((min(data[:,3])-0.5,max(data[:,3])+0.5))
    plt.ylim((min(data[:,4])-0.5,max(data[:,4])+0.5))
    b=plt.colorbar()
    b.set_label("$\log_{{10}}\\frac{{L_2^\\mathrm{{test}}}}{{N_\mathrm{{non-zero}}}}$")

    plt.savefig(f_out)
    plt.close('all')

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

def mkPlotScatter(data, f_out, orders=None,lx="$x$", ly="$y$", logy=True, logx=True):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    plt.clf()
    mpl.rc('text', usetex = True)
    mpl.rc('font', family = 'serif', size=12)
    mpl.style.use("ggplot")

    plt.xlabel(lx)
    plt.ylabel(ly)
    if logx: plt.xscale("log")
    if logy: plt.yscale("log")

    txt = []

    if orders is not None:
        c   = []
        mrk = []
        for num, (m,n) in enumerate(orders):
            if n==0:
                c.append("b")
            else:
                c.append("r")
            # mrk.append(marker[clus[num]])
            txt.append("({},{})".format(m,n))
    else:
        c = ["r" for _ in len(data)]
        txt = []

    data=np.array(data)
    for num, d in enumerate(data):
        plt.scatter(d[0], d[1],c=c[num])#, marker=mrk[num])
    # plt.scatter(data[:,0], data[:,1],c=c, marker=mrk)

    if orders is not None:
        for num, t in enumerate(txt):
            plt.annotate(t, data[num])


    plt.savefig(f_out)
    plt.close('all')

def mkPlotCompromise(data, f_out, orders=None,lx="$x$", ly="$y$", logy=True, logx=True, normalize_data=True):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    plt.clf()
    mpl.rc('text', usetex = True)
    mpl.rc('font', family = 'serif', size=12)
    mpl.style.use("ggplot")


    plt.xlabel(lx)
    plt.ylabel(ly)
    if logx: plt.xscale("log")
    if logy: plt.yscale("log")

    # CMP = [a*b for a,b in data]
    # CMP = [np.sqrt(a*a + b*b) for a,b in data]

    # i_cmp = CMP.index(min(CMP))
    # i_2 = CMP.index(sorted(CMP)[1])
    # i_3 = CMP.index(sorted(CMP)[2])
    # plt.scatter(data[i_cmp][0], data[i_cmp][1], marker = '*', c = "gold"  ,s=400, alpha = 1.0)
    # plt.scatter(data[i_2][0]  , data[i_2][1]  , marker = '*', c = "silver",s=400, alpha = 1.0)
    # plt.scatter(data[i_3][0]  , data[i_3][1]  , marker = '*', c = "peru"  ,s=400, alpha = 1.0)


    data=np.array(data)
    print(orders)
    print(data)
    if normalize_data: data = data / data.max(axis=0)

    b_pareto = is_pareto_efficient_dumb(data)
    _pareto = data[b_pareto] # Pareto points, sorted in x

    pareto = _pareto[_pareto[:,0].argsort()]
    print("===========")
    print(pareto)
    print("===========")

    slopes = []
    for num, (x0,y0) in enumerate(pareto[:-1]):
        x1, y1 = pareto[num+1]
        # slopes.append( abs( (y0 - y1)/(x1 - x0)) )
        slopes.append( (y0 - y1)/(x1 - x0))

    d_slopes = np.array([])
    for num, s in enumerate(slopes[:-1]):
        d_slopes = np.append(d_slopes, slopes[num+1]/s)

    print(d_slopes)



    eps=0.2
    winner=0
    for num, s in enumerate(slopes):
        print(num, s)
        if s<eps:
            break
        else:
            winner = num
    print("{}: {}".format(winner, slopes[winner]))
    i_win = winner + 1


    # OR
    print("--------------------")
    for num, (x0,y0) in enumerate(pareto[:-1]):
        print("%.2f \t %.4f"%(x0,y0))
        print ("upon")
        x1, y1 = pareto[num+1]
        print("%.2f \t %.4f"%(x1,y1))
        print("s = %.4f "%(slopes[num]))

    print("--------------------")




    winner = np.argmax(d_slopes)
    print("{}: {}".format(winner, d_slopes[winner]))
    i_win = winner + 1









    plt.scatter(pareto[:,0]  , pareto[:,1]  , marker = 'o', c = "silver"  ,s=100, alpha = 1.0)
    plt.scatter(pareto[i_win,0]  , pareto[i_win,1]  , marker = '*', c = "gold"  ,s=444, alpha = 1.0)

    c, txt   = [], []
    for num, (m,n) in enumerate(orders):
        if n==0:
            c.append("b")
        else:
            c.append("r")
        if b_pareto[num]:
            txt.append("({},{})".format(m,n))
        else:
            txt.append("")

    print("Plotting")
    # from IPython import embed
    # embed()
    # import sys
    # sys.exit(1)

    for num, d in enumerate(data): plt.scatter(d[0], d[1],c=c[num])
    for num, t in enumerate(txt): plt.annotate(t, (data[num][0], data[num][1]))

    print("==================")
    sorted_ds = np.argsort(-d_slopes)
    for num in sorted_ds:
    # for num, s in enumerate(slopes[:-1]):
        o1, o2, o3 = "","",""
        for ind, t in enumerate(txt):
            if np.all(pareto[num] == data[ind]):
                o1 = t
            if np.all(pareto[num+1] == data[ind]):
                o2 = t
            if np.all(pareto[num+2] == data[ind]):
                o3 = t
        print("s = {:.10f} between {} and {}".format(slopes[num+1],o2,o3))
        print ("upon")
        print("s = {:.10f} between {} and {}".format(slopes[num],o1,o2))
        print("r = %.4f "%(d_slopes[num]))


        print("\n")
    print("==================")

    # plt.axes().set_aspect('equal', 'datalim')
    plt.savefig(f_out)
    plt.close('all')

def mkPlotParetoVariance(data, n_test, f_out, orders=None,lx="$x$", ly="$y$", logy=True, logx=True):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    plt.clf()
    mpl.rc('text', usetex = True)
    mpl.rc('font', family = 'serif', size=12)
    mpl.style.use("ggplot")


    plt.xlabel(lx)
    plt.ylabel(ly)
    if logx: plt.xscale("log")
    if logy: plt.yscale("log")

    PV = [a*a / (n_test-b -1) for a, b in data]

    NC = []
    for m,n in orders:
        if n==0:
            NC.append(apprentice.tools.numCoeffsPoly(m))
        else:
            NC.append(apprentice.tools.numCoeffsRapp((m,n)))


    # CMP = [a*np.sqrt(b) for a,b in data]

    # i_cmp = CMP.index(min(CMP))
    # i_2 = CMP.index(sorted(CMP)[1])
    # i_3 = CMP.index(sorted(CMP)[2])
    # plt.scatter(data[i_cmp][0], data[i_cmp][1], marker = '*', c = "gold"  ,s=400, alpha = 1.0)
    # plt.scatter(data[i_2][0]  , data[i_2][1]  , marker = '*', c = "silver",s=400, alpha = 1.0)
    # plt.scatter(data[i_3][0]  , data[i_3][1]  , marker = '*', c = "peru"  ,s=400, alpha = 1.0)


    c, txt   = [], []
    for num, (m,n) in enumerate(orders):
        if n==0:
            c.append("b")
        else:
            c.append("r")
        txt.append("({},{})".format(m,n))

    for num, d in enumerate(data): plt.scatter(d[0], d[1],c=c[num])
    for num, t in enumerate(txt): plt.annotate(t, data[num])

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


def plotoptimaldegree(folder,testfile, desc,bottom_or_all):
    import glob
    import json
    import re
    filelistRA = np.array(glob.glob(folder+"/out/*.json"))
    filelistRA = np.sort(filelistRA)

    filelistPA = np.array(glob.glob(folder+"/outpa/*.json"))
    filelistPA = np.sort(filelistPA)

    if not os.path.exists(folder+"/plots"):
        os.mkdir(folder+'/plots')


    maxpap = 0
    dim = 0
    ts = ""

    orders = []
    APP = []

    for file in filelistRA:
        if file:
			with open(file, 'r') as fn:
				datastore = json.load(fn)
        m = datastore['m']
        n = datastore['n']
        ts = datastore['trainingscale']
        if(n!=0):
            orders.append((m,n))
            APP.append(apprentice.RationalApproximationSIP(datastore))
        dim = datastore['dim']

    for file in filelistPA:
        if file:
			with open(file, 'r') as fn:
				datastore = json.load(fn)
        m = datastore['m']
        if((m,0) in orders):
            continue
        orders.append((m,0))
        APP.append(apprentice.PolynomialApproximation(initDict=datastore))
        if m > maxpap:
            maxpap = m

    try:
        X, Y = apprentice.tools.readData(testfile)
    except:
        DATA = apprentice.tools.readH5(testfile, [0])
        X, Y= DATA[0]

    if(bottom_or_all == "bottom"):
        trainingsize = apprentice.tools.numCoeffsPoly(dim,maxpap)
        if(ts == ".5x" or ts == "0.5x"):
            trainingsize = 0.5 * trainingsize
        elif(ts == "1x"):
            trainingsize = trainingsize
        elif(ts == "2x"):
            trainingsize = 2 * trainingsize
        elif(ts == "Cp"):
             trainingsize = len(X)
        else: raise Exception("Training scale %s unknown"%(ts))
        testset = [i for i in range(trainingsize,len(X))]
        X_test = X[testset]
        Y_test = Y[testset]
    elif(bottom_or_all == "all"):
        X_test = X
        Y_test = Y
    else:
        raise Exception("bottom or all? Option ambiguous. Check spelling and/or usage")
    if(len(X_test)<=1): raise Exception("Not enough testing data")

    # print(orders)
    # print(len(X_test))


    L2      = [np.sqrt(raNorm(app, X_test, Y_test, 2))               for app in APP]
    Linf    = [raNormInf(app, X_test, Y_test)                        for app in APP]
    NNZ     = [apprentice.tools.numNonZeroCoeff(app, 1e-6) for app in APP]
    VAR     = [l/m for l, m in zip(L2, NNZ)]

    ncN, ncM = [], []

    NC = []
    for m,n in orders:
        ncM.append(apprentice.tools.numCoeffsPoly(dim, m))
        ncN.append(apprentice.tools.numCoeffsPoly(dim, n))
        if n==0:
            NC.append(apprentice.tools.numCoeffsPoly(dim, m))
        else:
            NC.append(apprentice.tools.numCoeffsRapp(dim, (m,n)))

    D3D = np.array([(m,n,v,o[0], o[1]) for m,n,v, o in zip(ncM,ncN, VAR, orders)])
    outfileparetomn = "%s/plots/Poptdeg_%s_paretomn.png"%(folder, desc)
    mkPlotParetoSquare(D3D, outfileparetomn)
    print("paretomn written to %s"%(outfileparetomn))

    # import matplotlib as mpl
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # plt.clf()
    # mpl.rc('text', usetex = True)
    # mpl.rc('font', family = 'serif', size=12)
    # mpl.style.use("ggplot")
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # p3D = is_pareto_efficient_dumb(D3D)
    # for num, (m,n,v) in enumerate(zip(ncM,ncN, VAR)):
    #     if p3D[num]:
    #         ax.scatter(m, n, np.log10(v), c="gold")
    #     else:
    #         ax.scatter(m, n, np.log10(v), c="r")
    # ax.set_xlabel('nc m')
    # ax.set_ylabel('nc n')
    # ax.set_zlabel('log v')
    # plt.show()

    NNC = []
    for num , n in enumerate(NC):
        NNC.append(n-NNZ[num])

    CMP = [a*b for a,b in zip(NNZ, L2)]

    outfilepareton = "%s/plots/Poptdeg_%s_pareton.png"%(folder, desc)
    mkPlotCompromise([(a,b) for a, b in zip(NC, VAR)],  outfilepareton,  orders, ly="$\\frac{L_2^\\mathrm{test}}{N_\mathrm{non-zero}}$", lx="$N_\\mathrm{coeff}$", logy=True, logx=True, normalize_data=False)
    print("pareton written to %s"%(outfilepareton))









if __name__ == "__main__":

    import os, sys, h5py
    if len(sys.argv)!=5:
        print("Usage: {} infolder testfile  fndesc  bottom_or_all".format(sys.argv[0]))
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print("Input folder '{}' not found.".format(sys.argv[1]))

    if not os.path.exists(sys.argv[1]+"/out"):
        print("Input folder '{}' not found.".format(sys.argv[1]+"/out"))

    if not os.path.exists(sys.argv[1]+"/outpa"):
        print("Input folder '{}' not found.".format(sys.argv[1]+"/outpa"))

    if not os.path.exists(sys.argv[2]):
        print("Test file '{}' not found.".format(sys.argv[2]))

    plotoptimaldegree(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

    exit(0)
