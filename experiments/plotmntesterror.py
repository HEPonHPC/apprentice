import numpy as np
from apprentice import RationalApproximationSIP
from sklearn.model_selection import KFold
from apprentice import tools, readData
import os

def plotmntesterr(folder,testfile, desc,bottom_or_all, measure):
    import glob
    import json
    import re
    filelist = np.array(glob.glob(folder+"/out/*.json"))
    filelist = np.sort(filelist)

    try:
        X, Y = readData(testfile)
    except:
        DATA = tools.readH5(testfile, [0])
        X, Y= DATA[0]
    minp = np.inf
    minq = np.inf
    maxp = 0
    maxq = 0
    dim = 0

    stats = {}

    for file in filelist:
        if file:
			with open(file, 'r') as fn:
				datastore = json.load(fn)
        dim = datastore['dim']
        m = datastore['m']
        n = datastore['n']
        if(m<minp):
            minp=m
        if(n<minq):
            minq=n
        if m > maxp:
            maxp = m
        if n > maxq:
            maxq = n

    if(bottom_or_all == "bottom"):
        trainingsize = tools.numCoeffsPoly(dim,maxp) + tools.numCoeffsPoly(dim,maxq)
        testset = [i for i in range(trainingsize,len(X))]
        X_test = X[testset]
        Y_test = Y[testset]
    elif(bottom_or_all == "all"):
        X_test = X
        Y_test = Y
    else:
        raise Exception("bottom or all? Option ambiguous. Check spelling and/or usage")
    if(len(X_test)<=1): raise Exception("Not enough testing data")

    if not os.path.exists(folder+"/plots"):
        os.mkdir(folder+'/plots')
    outfilepng = "%s/plots/Pmn_%s_%s_from_plotmntesterr.png"%(folder, desc, measure)
    outfilestats = "%s/plots/J%s_stats_from_plotmntesterr.json"%(folder,desc)
    error = np.empty(shape = (maxp-minp+1,maxq-minq+1))
    for i in range(maxp-minp+1):
        for j in range(maxq-minq+1):
            error[i][j] = None

    nnzthreshold = 1e-6
    for file in filelist:
        if file:
			with open(file, 'r') as fn:
				datastore = json.load(fn)
        m = datastore['m']
        n = datastore['n']
        ts = datastore['trainingscale']

        for i, p in enumerate(datastore['pcoeff']):
            if(abs(p)<nnzthreshold):
                datastore['pcoeff'][i] = 0.
        if('qcoeff' in datastore):
            for i, q in enumerate(datastore['qcoeff']):
                if(abs(q)<nnzthreshold):
                    datastore['qcoeff'][i] = 0.

        rappsip = RationalApproximationSIP(datastore)
        Y_pred = rappsip.predictOverArray(X_test)

        key = "p%d_q%d_ts%s"%(m,n,ts)
        l1 = np.sum(np.absolute(Y_pred-Y_test))
        l2 = np.sqrt(np.sum((Y_pred-Y_test)**2))
        linf = np.max(np.absolute(Y_pred-Y_test))
        nnz = tools.numNonZeroCoeff(rappsip,nnzthreshold)
        l2divnnz = l2/float(nnz)
        stats[key] = {}
        stats[key]['nnz'] =nnz
        stats[key]['l2divnnz'] = l2divnnz
        stats[key]['l1'] =l1
        stats[key]['l2'] =l2
        stats[key]['linf'] =linf

        if(measure == 'l1'):
            error[m-minp][n-minq] = l1
        elif(measure == 'l2'):
            error[m-minp][n-minq] = l2
        elif(measure == 'linf'):
            error[m-minp][n-minq] = linf
        elif(measure == 'l2divnnz'):
            error[m-minp][n-minq] = l2divnnz
        else:
            raise Exception("measure not found. Check spelling and/or usage")



    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import re

    mpl.rc('text', usetex = True)
    mpl.rc('font', family = 'serif', size=12)
    mpl.style.use("ggplot")
    cmapname   = 'viridis'
    plt.clf()

    markersize = 200
    vmin = -4
    vmax = 2
    X,Y = np.meshgrid(range(minq,maxq+1),range(minp,maxp+1))
    # plt.scatter(X,Y , marker = 's', s=markersize, c = np.ma.log10(error), cmap = cmapname, vmin=vmin, vmax=vmax, alpha = 1)
    plt.scatter(X,Y , marker = 's', s=markersize, c = error, cmap = cmapname, alpha = 1)
    plt.xlabel("$n$")
    plt.ylabel("$m$")
    plt.xlim((minq-1,maxq+1))
    plt.ylim((minp-1,maxp+1))
    b=plt.colorbar()
	# b.set_label("$\log_{10}\\left|\\left|f - \\frac{p^m}{q^n}\\right|\\right|_2$")
	# b.set_label("$\\left|\\left|f - \\frac{p^m}{q^n}\\right|\\right|_2$")
    if(measure == 'l1'):
        b.set_label("L1 error norm")
    elif(measure == 'l2'):
        b.set_label("L2 error norm")
    elif(measure == 'linf'):
        b.set_label("Linf error norm")
    elif(measure == 'l2divnnz'):
        b.set_label("L2/nnz error norm")



    keys = []
    l1arr = np.array([])
    l2arr = np.array([])
    linfarr = np.array([])
    nnzarr = np.array([])
    l2divnnzarr= np.array([])
    for key in stats:
        keys.append(key)
        l1arr = np.append(l1arr,stats[key]['l1'])
        l2arr = np.append(l2arr,stats[key]['l2'])
        linfarr = np.append(linfarr,stats[key]['linf'])
        nnzarr = np.append(nnzarr,stats[key]['nnz'])
        l2divnnzarr = np.append(l2divnnzarr,stats[key]['l2divnnz'])

    minstats = {}
    minstats["l1"] = {}
    minstats["l1"]["val"] = np.min(l1arr)
    minstats["l1"]["loc"] = keys[np.argmin(l1arr)]

    minstats["l2"] = {}
    minstats["l2"]["val"] = np.min(l2arr)
    minstats["l2"]["loc"] = keys[np.argmin(l2arr)]

    minstats["linf"] = {}
    minstats["linf"]["val"] = np.min(linfarr)
    minstats["linf"]["loc"] = keys[np.argmin(linfarr)]

    minstats["nnz"] = {}
    minstats["nnz"]["val"] = np.min(nnzarr)
    minstats["nnz"]["loc"] = keys[np.argmin(nnzarr)]

    minstats["l2divnnz"] = {}
    minstats["l2divnnz"]["val"] = np.min(l2divnnzarr)
    minstats["l2divnnz"]["loc"] = keys[np.argmin(l2divnnzarr)]

    if(measure == 'l1'):
        minkey = keys[np.argmin(l1arr)]
        minval = np.min(l1arr)
    elif(measure == 'l2'):
        minkey = keys[np.argmin(l2arr)]
        minval = np.min(l2arr)
    elif(measure == 'linf'):
        minkey = keys[np.argmin(linfarr)]
        minval = np.min(linfarr)
    elif(measure == 'l2divnnz'):
        minkey = keys[np.argmin(l2divnnzarr)]
        minval = np.min(l2divnnzarr)

    digits = [int(s) for s in re.findall(r'-?\d+\.?\d*', minkey)]
    winner = (digits[0], digits[1])

    stats["minstats"] = minstats

    plt.scatter(winner[1], winner[0], marker = '*', c = "magenta",s=markersize, alpha = 0.9)

    plt.title("%s. Winner is p%d q%d with val = %f"%(desc,winner[0], winner[1],minval))
    plt.savefig(outfilepng)

    import json
    with open(outfilestats, "w") as f:
        json.dump(stats, f,indent=4, sort_keys=True)












# python plotmntesterror.py f21_2x ../benchmarkdata/f21_test.txt f21 all l2

if __name__ == "__main__":
    import os, sys
    if len(sys.argv)!=6:
        print("Usage: {} infolder testfile  fndesc  bottom_or_all l1_or_l2_or_linf_or_l2divnnz".format(sys.argv[0]))
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print("Input folder '{}' not found.".format(sys.argv[1]))

    if not os.path.exists(sys.argv[2]):
        print("Test file '{}' not found.".format(sys.argv[2]))

    plotmntesterr(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],  sys.argv[5])
###########
