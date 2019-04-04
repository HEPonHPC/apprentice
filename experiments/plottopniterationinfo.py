import numpy as np
from apprentice import RationalApproximationSIP
from sklearn.model_selection import KFold
from apprentice import tools, readData
import os

def plottopniterationinfo(folder,testfile, desc,topn, bottom_or_all):
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

    fileArr = np.array([])
    iterationNoArr = np.array([])

    for file in filelist:
        if file:
            with open(file, 'r') as fn:
                datastore = json.load(fn)
        fileArr = np.append(fileArr,file)
        iterationNoArr = np.append(iterationNoArr,len(datastore['iterationinfo']))

    print(np.c_[fileArr,iterationNoArr])

    if(topn > len(fileArr)):
        raise Exception("n (%d) > available runs (%d)"%(topn,len(fileArr)))

    if not os.path.exists(folder+"/plots"):
        os.mkdir(folder+'/plots')


    topnindex = iterationNoArr.argsort()[-topn:][::-1]

    for i in range(len(topnindex)):
        index = topnindex[i]
        outfile = "%s/plots/Ptii_%s_topiterinfo_%d.pdf"%(folder, desc, i+1)
        file = fileArr[index]
        if file:
            with open(file, 'r') as fn:
                datastore = json.load(fn)
        m = datastore['m']
        n = datastore['n']
        ts = datastore['trainingscale']
        trainingsize =datastore['trainingsize']
        totaltime = datastore['log']['fittime']

        if(bottom_or_all == "bottom"):
            trainingsize = datastore['trainingsize']
            testset = [i for i in range(trainingsize,len(X))]
            X_test = X[testset]
            Y_test = Y[testset]
        elif(bottom_or_all == "all"):
            X_test = X
            Y_test = Y
        else:
            raise Exception("bottom or all? Option ambiguous. Check spelling and/or usage")

        if(len(X_test)<=1): raise Exception("Not enough testing data")

        noofmultistarts = np.array([])
        mstime = np.array([])
        robobj = np.array([])
        fittime = np.array([])
        lsqobj = np.array([])

        interationinfo = datastore['iterationinfo']
        for iter in interationinfo:
            roboinfo = iter["robOptInfo"]["info"]
            noofmultistarts = np.append(noofmultistarts,roboinfo[len(roboinfo)-1]["log"]["noRestarts"])
            mstime = np.append(mstime,roboinfo[len(roboinfo)-1]["log"]["time"])
            robobj = np.append(robobj,roboinfo[len(roboinfo)-1]["robustObj"])
            fittime = np.append(fittime,iter["log"]["time"])
            lsqobj = np.append(lsqobj,iter["leastSqObj"])

        Xvals = range(1,int(iterationNoArr[index])+1)

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

        nnzthreshold = 1e-6
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


        # f.suptitle("%s. m = %d, n = %d, ts = %d (%s). \n Total CPU time = %.4f, Total # of iterations = %d.\nl1 = %.4f, l2 = %.4f, linf = %.4f, nnz = %d, l2/nnz = %f"%(desc,m,n,trainingsize,ts,totaltime,len(interationinfo),l1,l2,linf,nnz,l2/nnz), size=15)
        plt.savefig(outfile)
        plt.clf()

# python plottopniterationinfo.py f20_2x ../benchmarkdata/f20_test.txt f20 10 all

if __name__ == "__main__":
    import os, sys
    if len(sys.argv)!=6:
        print("Usage: {} infolder testfile  fndesc n bottom_or_all".format(sys.argv[0]))
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print("Input folder '{}' not found.".format(sys.argv[1]))

    if not os.path.exists(sys.argv[2]):
        print("Test file '{}' not found.".format(sys.argv[2]))

    plottopniterationinfo(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]),  sys.argv[5])
###########
