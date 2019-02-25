import numpy as np
from apprentice import RationalApproximationSIP
from sklearn.model_selection import KFold
from apprentice import tools, readData
import os

def plotmntesterr(folder,testfile, desc,bottom_or_all):
    import glob
    import json
    import re
    filelist = np.array(glob.glob(folder+"/out/*.json"))
    filelist = np.sort(filelist)

    X, Y = readData(testfile)
    minp = np.inf
    minq = np.inf
    maxp = 0
    maxq = 0

    stats = {}

    for file in filelist:
        if file:
			with open(file, 'r') as fn:
				datastore = json.load(fn)
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

    if not os.path.exists(folder+"/plots"):
        os.mkdir(folder+'/plots')
    outfilepng = "%s/plots/P%s_from_plotmntesterr.png"%(folder,desc)
    outfilestats = "%s/plots/J%s_stats_from_plotmntesterr.json"%(folder,desc)
    error = np.zeros(shape = (maxp-minp+1,maxq-minq+1))

    for file in filelist:
        if file:
			with open(file, 'r') as fn:
				datastore = json.load(fn)
        m = datastore['m']
        n = datastore['n']
        ts = datastore['trainingscale']


        if(bottom_or_all == "bottom"):
            trainingsize = datastore['trainingsize']
            testset = [i for i in range(trainingsize,len(X_test))]
            X_test = X[testset]
            Y_test = Y[testset]
        else:
            X_test = X
            Y_test = Y

        if(len(X_test)<=1): raise Exception("Not enough testing data")
        rappsip = RationalApproximationSIP(datastore)
        Y_pred = rappsip.predictOverArray(X_test)

        key = "p%d_q%d_ts%s"%(m,n,ts)
        l1 = np.sum(np.absolute(Y_pred-Y_test))
        l2 = np.sqrt(np.sum((Y_pred-Y_test)**2))
        linf = np.max(np.absolute(Y_pred-Y_test))
        stats[key] = {}
        stats[key]['l1'] =l1
        stats[key]['l2'] =l2
        stats[key]['linf'] =linf

        error[m-minp][n-minq] = l2

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rc('text', usetex = True)
    mpl.rc('font', family = 'serif', size=12)
    mpl.style.use("ggplot")
    cmapname   = 'viridis'
    plt.clf()

    markersize = 500
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
    b.set_label("L2 error norm")
    plt.title(desc)
    plt.savefig(outfilepng)

    keys = []
    l1arr = np.array([])
    l2arr = np.array([])
    linfarr = np.array([])
    for key in stats:
        keys.append(key)
        l1arr = np.append(l1arr,stats[key]['l1'])
        l2arr = np.append(l2arr,stats[key]['l2'])
        linfarr = np.append(linfarr,stats[key]['linf'])

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
    
    stats["minstats"] = minstats

    import json
    with open(outfilestats, "w") as f:
        json.dump(stats, f,indent=4, sort_keys=True)










# python plotmntesterror.py f21_2x ../benchmarkdata/f21_test.txt f21 all

if __name__ == "__main__":
    import os, sys
    if len(sys.argv)!=5:
        print("Usage: {} infolder testfile  fndesc  bottom_or_all".format(sys.argv[0]))
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print("Input folder '{}' not found.".format(sys.argv[1]))

    if not os.path.exists(sys.argv[2]):
        print("Test file '{}' not found.".format(sys.argv[2]))

    plotmntesterr(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
###########
