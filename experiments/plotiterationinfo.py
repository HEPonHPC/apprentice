import numpy as np
from apprentice import RationalApproximationSIP
from sklearn.model_selection import KFold
from apprentice import tools, readData
import os

def plotiterationinfo(fname,noise, m,n,ts):
    # import glob
    import json
    # import re
    noisestr = ""

    if(noise!="0"):
        noisestr = "_noisepct"+noise
    folder = "%s%s_%s"%(fname,noisestr,ts)

    optjsonfile = folder+"/plots/Joptdeg_"+fname+noisestr+"_jsdump_opt6.json"

    if not os.path.exists(optjsonfile):
        print("optjsonfile: " + optjsonfile+ " not found")
        exit(1)

    if optjsonfile:
        with open(optjsonfile, 'r') as fn:
            optjsondatastore = json.load(fn)

    optm = optjsondatastore['optdeg']['m']
    optn = optjsondatastore['optdeg']['n']
    print(optm,optn)
    rappsipfile = "%s/out/%s%s_%s_p%s_q%s_ts%s.json"%(folder,fname,noisestr,ts,m,n,ts)

    if rappsipfile:
        with open(rappsipfile, 'r') as fn:
            datastore = json.load(fn)
    ts = datastore['trainingscale']
    trainingsize =datastore['trainingsize']
    totaltime = datastore['log']['fittime']

    noofmultistarts = np.array([])
    mstime = np.array([])
    robobj = np.array([])
    fittime = np.array([])
    lsqobj = np.array([])

    interationinfo = datastore['iterationinfo']
    for num,iter in enumerate(interationinfo):
        roboinfo = iter["robOptInfo"]["info"]
        noofmultistarts = np.append(noofmultistarts,roboinfo[len(roboinfo)-1]["log"]["noRestarts"])
        mstime = np.append(mstime,roboinfo[len(roboinfo)-1]["log"]["time"])
        robobj = np.append(robobj,roboinfo[len(roboinfo)-1]["robustObj"])
        fittime = np.append(fittime,iter["log"]["time"])
        lsqobj = np.append(lsqobj,iter["leastSqObj"])
        print(str(num))
        print(roboinfo[0]["robustArg"])

    Xvals = range(1,len(interationinfo)+1)
    import matplotlib.pyplot as plt
    # f, axes = plt.subplots(4, sharex=True,figsize=(12,12))
    f, axes = plt.subplots(2, sharex=True, figsize=(12,12))
    # p0, = axes[0].plot(Xvals,np.ma.log10(noofmultistarts),'g')
    tmp = axes[0]
    axes[0] = axes[1]
    axes[1] = tmp
    msax = axes[0].twinx()
    p01, = axes[0].plot(Xvals,np.ma.log10(mstime),'r--',label="multistart time")
    p02, = msax.plot(Xvals,robobj,'b-')

    firerrorax = axes[1].twinx()
    p11, = axes[1].plot(Xvals,np.ma.log10(fittime),'r--')
    p12, = firerrorax.plot(Xvals,np.ma.log10(lsqobj),'b')

    axes[1].set_xlabel("no. of iterations")
    # axes[0].set_ylabel("log$_{10}$(no. of multistarts)")
    axes[0].set_ylabel("log$_{10}$(multistart time in sec)")
    msax.set_ylabel("$min$ $q(x)$")
    axes[1].set_ylabel("log$_{10}$(fit time in sec)")
    firerrorax.set_ylabel("$log_{10}\\left(min\ \\left|\\left|f-\\frac{p}{q}\\right|\\right||_2^2\\right)$")

    for ax in axes.flat:
        ax.label_outer()
    axes[0].yaxis.label.set_color(p01.get_color())
    msax.yaxis.label.set_color(p02.get_color())
    axes[1].yaxis.label.set_color(p11.get_color())
    firerrorax.yaxis.label.set_color(p12.get_color())

    tkw = dict(size=4, width=1.5)
    axes[0].tick_params(axis='y', colors=p01.get_color(), **tkw)
    msax.tick_params(axis='y', colors=p02.get_color(), **tkw)
    # axes[1].tick_params(axis='y', colors=p1.get_color(), **tkw)
    # axes[2].tick_params(axis='y', colors=p2.get_color(), **tkw)
    axes[1].tick_params(axis='y', colors=p11.get_color(), **tkw)
    firerrorax.tick_params(axis='y', colors=p12.get_color(), **tkw)

    # f.suptitle("%s. m = %d, n = %d, ts = %d (%s). \n Total CPU time = %.4f, Total # of iterations = %d.\nl1 = %.4f, l2 = %.4f, linf = %.4f, nnz = %d, l2/nnz = %f"%(desc,m,n,trainingsize,ts,totaltime,len(interationinfo),l1,l2,linf,nnz,l2/nnz), size=15)
    outfile = "%s/plots/Piterinfo_%s%s_p%s_q%s_ts%s.pdf"%(folder, fname,noisestr,m,n,ts )
    plt.savefig(outfile)
    print("open %s;"%(outfile))
    plt.clf()

# python plottopniterationinfo.py f20_2x ../benchmarkdata/f20_test.txt f20 10 all

if __name__ == "__main__":
    import os, sys
    if len(sys.argv)!=6:
        print("Usage: {} fname noise m n ts".format(sys.argv[0]))
        sys.exit(1)

    plotiterationinfo(sys.argv[1], sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
###########
