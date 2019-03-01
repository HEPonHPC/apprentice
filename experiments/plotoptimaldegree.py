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

"""
Calculates the corner as the one with largest slope ratio
For curve:
D - B
    |
    A - C
If slope_left > slope_right, winner is the mid point of L (s(BA)>s(AC) => corner is A)
If slope_right > slope_left, winner is the tail end of L (s(BA)>s(DB) => corner is A)

Ignores pareto points before largest slope to improve monotonicity of the pareto front
which is required for this method to work corrctly. Hence, this technique is not
guaranteed to work since there could be further big drops in the pareto front
after the largest slope which could result in getting an incorrect corner

Use the triangulation method instead as that is more robust and consistent
"""
def findCornerSlopesRatios(pareto):
    slopes = []
    for num, (x0,y0) in enumerate(pareto[:-1]):
        x1, y1 = pareto[num+1]
        slopes.append( (np.log10(y0) - np.log10(y1))/( np.log10(x1) - np.log10(x0)))

    d_slopes_2 = []
    kslope=np.argmax(slopes)-1
    for k in range(kslope,len(pareto)-1):
        pk = pareto[k]
        pk_p1 = pareto[k+1]
        pk_m1 = pareto[k-1]
        sk    = (np.log10(pk_p1[1]) - np.log10(pk[1]   ))/( np.log10(pk[0])    - np.log10(pk_p1[0]))
        sk_m1 = (np.log10(pk[1]) - np.log10(pk_m1[1]))/( np.log10(pk_m1[0]) - np.log10(pk[0]))
        if sk>sk_m1:
            dk = sk/sk_m1
        else:
            dk = sk_m1/sk
        d_slopes_2.append(dk)
    # print("--------------------")
    # for num, (x0,y0) in enumerate(pareto[:-1]):
    #     print("%.2f \t %.4f"%(x0,y0))
    #     print ("upon")
    #     x1, y1 = pareto[num+1]
    #     print("%.2f \t %.4f"%(x1,y1))
    #     print("s = %.4f "%(slopes[num]))
    #
    # print("--------------------")


    kofint  = np.argmax(d_slopes_2) + kslope-1
    s_left  = slopes[kofint]
    s_right = slopes[kofint+1]

    if s_right > s_left: i_win = kofint + 2
    else: i_win = kofint + 1
    return i_win

"""
Code Holger and Mohan came up with on 20190228.
Cleaned up on 20190301
Cleaned up code from findCornerTriangulation2():
"""
def findCornerTriangulation(pareto):
    def angle(B, A, C):
        """
        Find angle at A
        On the L, B is north of A and C is east of A
        B
        | \
        A - C
        """

        ba = [A[0]-B[0], A[1]-B[1]]
        ac = [C[0]-A[0], C[1]-A[1]]

        l1=np.linalg.norm(ba)
        l2=np.linalg.norm(ac)
        import math
        return math.acos(np.dot(ba,ac)/l1/l2)

    def area(B, A, C):
        """
        A,B,C ---
        On the L, B is north of A and C is east of A
        B
        | \     (area is -ve)
        A - C
        """
        return 0.5 *( ( C[0] - A[0] )*(A[1] - B[1]) -  (A[0] - B[0])*(C[1] - A[1]) )

    cte=np.cos(7./8*np.pi)
    cosmax=-2
    corner=len(pareto)-1

    lpareto=np.log10(pareto)

    C = lpareto[corner]
    for k in range(0,len(lpareto)-2):
        B = lpareto[k]
        for j in range(k,len(lpareto)-2):
            A = lpareto[j+1]
            _area = area(B,A,C)
            _angle = angle(B,A,C)

            if _angle > cte and _angle > cosmax and _area < 0:
                corner = j + 1
                cosmax = _angle
                print(_area,_angle)
    print("In findCornerTriangulation, I got this {}".format(pareto[corner]))
    return corner

"""
Code Holger and Mohan came up with on 20190228.
Cleaned up code in findCornerTriangulation.
Keeping this around for reference
"""
def findCornerTriangulation2(pareto):
    def angle(A,B,C):
        """
        Find angle at A
        """
        ca = [A[0]-C[0], A[1]-C[1]]
        ab = [B[0]-A[0], B[1]-A[1]]

        l1=np.linalg.norm(ca)
        l2=np.linalg.norm(ab)
        import math
        top= np.dot(ca,ab)
        return math.acos(top/l1/l2)

    def area(A,B,C):
        """
        A,B,C ---
        """
        return 0.5 *( ( B[0] - A[0] )*(A[1] - C[1]) -  (A[0] - C[0])*(B[1] - A[1]) )

    cte=np.cos(7./8*np.pi)
    cosmax=-2
    corner=len(pareto)-1

    lpareto=np.log10(pareto)

    C = lpareto[corner]
    for k in range(0,len(lpareto)-2):
        B = lpareto[k]
        for j in range(k,len(lpareto)-2):
            A=lpareto[j+1]
            _a = area(A,C,B)
            _t = angle(A,C,B)
            if _t>cte and _t> cosmax and _a<0:
                corner = j+1
                cosmax = _t
                print(_a,_t)
    print("In findCornerTriangulation2, I got this {}".format(pareto[corner]))
    return corner


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

    data=np.array(data)
    # if normalize_data: data = data / data.max(axis=0)

    b_pareto = is_pareto_efficient_dumb(data)
    _pareto = data[b_pareto] # Pareto points

    pareto = _pareto[_pareto[:,0].argsort()]# Pareto points, sorted in x
    print("===========")
    print(pareto)
    print("===========")

    cornerT = findCornerTriangulation(pareto)
    cornerT2 = findCornerTriangulation2(pareto)
    cornerdSl = findCornerSlopesRatios(pareto)




    plt.scatter(pareto[:,0]  , pareto[:,1]  , marker = 'o', c = "silver"  ,s=100, alpha = 1.0)
    plt.scatter(pareto[cornerdSl,0]  , pareto[cornerdSl,1]  , marker = '+', c = "gold"  ,s=777, alpha = 1.0)
    plt.scatter(pareto[cornerT2,0]  , pareto[cornerT2,1]  , marker = 'x', c = "cyan"  ,s=444, alpha = 1.0)
    plt.scatter(pareto[cornerT,0]  , pareto[cornerT,1]  , marker = '*', c = "peru"  ,s=250, alpha = 1.0)


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

    for num, d in enumerate(data): plt.scatter(d[0], d[1],c=c[num])
    for num, t in enumerate(txt): plt.annotate(t, (data[num][0], data[num][1]))

    # plt.plot([pareto[0][0], pareto[-1][0]], [pareto[0][1], pareto[-1][1]], "k-")
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
        sys.exit(1)

    if not os.path.exists(sys.argv[1]+"/out"):
        print("Input folder '{}' not found.".format(sys.argv[1]+"/out"))
        sys.exit(1)

    if not os.path.exists(sys.argv[1]+"/outpa"):
        print("Input folder '{}' not found.".format(sys.argv[1]+"/outpa"))
        sys.exit(1)

    if not os.path.exists(sys.argv[2]):
        print("Test file '{}' not found.".format(sys.argv[2]))
        sys.exit(1)

    plotoptimaldegree(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

    exit(0)
