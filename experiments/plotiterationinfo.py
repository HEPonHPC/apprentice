import numpy as np
from apprentice import RationalApproximationSIP
from sklearn.model_selection import KFold
from apprentice import tools, readData
import os

def plotiterationinfo(fname,noise, m,n,ts,plot="no"):
    # import glob
    import json
    # import re
    noisestr = ""

    if(noise!="0"):
        noisestr = "_noisepct"+noise
    folder = "%s%s_%s"%(fname,noisestr,ts)
    if not os.path.exists(folder+"/plots"):
        os.mkdir(folder+'/plots')

    # optjsonfile = folder+"/plots/Joptdeg_"+fname+noisestr+"_jsdump_opt6.json"
    #
    # if not os.path.exists(optjsonfile):
    #     print("optjsonfile: " + optjsonfile+ " not found")
    #     exit(1)
    #
    # if optjsonfile:
    #     with open(optjsonfile, 'r') as fn:
    #         optjsondatastore = json.load(fn)
    #
    # optm = optjsondatastore['optdeg']['m']
    # optn = optjsondatastore['optdeg']['n']
    # print(optm,optn)
    rappsipfile = "%s/out/%s%s_%s_p%s_q%s_ts%s.json"%(folder,fname,noisestr,ts,m,n,ts)

    if rappsipfile:
        with open(rappsipfile, 'r') as fn:
            datastore = json.load(fn)
    ts = datastore['trainingscale']
    trainingsize =datastore['trainingsize']
    totaltime = datastore['log']['fittime']
    iterationinfono = len(datastore['iterationinfo'])

    noofmultistarts = np.array([])
    mstime = np.array([])
    robobj = np.array([])
    fittime = np.array([])
    lsqobj = np.array([])

    interationinfo = datastore['iterationinfo']
    robargdata = {0:[],1:[],2:[]}



    for num,iter in enumerate(interationinfo):
        roboinfo = iter["robOptInfo"]["info"]
        noofmultistarts = np.append(noofmultistarts,roboinfo[len(roboinfo)-1]["log"]["noRestarts"])
        mstime = np.append(mstime,roboinfo[len(roboinfo)-1]["log"]["time"])
        robobj = np.append(robobj,roboinfo[len(roboinfo)-1]["robustObj"])
        fittime = np.append(fittime,iter["log"]["time"])
        lsqobj = np.append(lsqobj,iter["leastSqObj"])
        print(str(num))
        print(roboinfo[0]["robustArg"])
        if(plot == "yes_3" or plot == "yes_2"):
            for i in range(3):
                robargdata[i].append(roboinfo[0]["robustArg"][i])

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

    if(plot=="yes_3"):

        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(15,10))

        ax = fig.add_subplot(1, 1, 1, projection='3d')
        for num in range(len(robargdata[0])):
            if(abs(robargdata[0][num]) == 1 and abs(robargdata[1][num]) == 1):
                ax.scatter(robargdata[0][num],robargdata[1][num], robargdata[2][num], marker='o',c='blue',s=1000,alpha = 1.0)
            else:
                ax.scatter(robargdata[0][num],robargdata[1][num],robargdata[2][num], marker='x',c='red',s=200,alpha = 1.0)
        # ax.view_init(azim=135, elev=90)

        # ax.scatter(robargdata[0],robargdata[1],robargdata[2])
        ax.set_xlabel('$x1$', fontsize = 12)
        ax.set_ylabel('$x2$', fontsize = 12)
        ax.set_zlabel('$x3$', fontsize = 12)
        outfile = "%s/plots/Piterinfo_robarg_%s%s_p%s_q%s_ts%s.pdf"%(folder, fname,noisestr,m,n,ts )
        ZZ = np.arange(-1, 1, 0.01)
        for l1 in [-1,1]:
            for l2 in [-1,1]:
                XX = l1*np.ones(len(ZZ))
                YY = l2*np.ones(len(ZZ))
                ax.plot(XX,YY,ZZ,c='orange')
        # plt.savefig(outfile)
        # print("open %s;"%(outfile))
        plt.show()
    elif(plot == "yes_2"):
        props = dict(boxstyle='square', facecolor='wheat', alpha=0.5)
        from math import ceil,sqrt
        no = int(ceil(sqrt(iterationinfono)))
        rows = no
        cols = no
        fig, axarr = plt.subplots(rows,cols,figsize=(15,15),sharex=True)
        index = 0
        for index in range(iterationinfono):
            r = int(index / no)
            c = index % no
            if rappsipfile:
                with open(rappsipfile, 'r') as fn:
                    datastore = json.load(fn)
            datastore['pcoeff'] = datastore['iterationinfo'][index]['pcoeff']
            datastore['qcoeff'] = datastore['iterationinfo'][index]['qcoeff']
            rappsip = RationalApproximationSIP(datastore)
            lbbb=-0.95
            ubbb=0.95
            ZZ = np.arange(lbbb, ubbb, 0.01)
            for l1 in [lbbb,ubbb]:
                for l2 in [lbbb,ubbb]:
                    XX = l1*np.ones(len(ZZ))
                    YY = l2*np.ones(len(ZZ))
                    qx = []
                    for num in range(len(ZZ)):
                        X=rappsip._scaler.scale(np.array([XX[num],YY[num],ZZ[num]]))
                        qx.append(rappsip.denom(X))
                    axarr[r][c].plot(ZZ,qx, label="x1 = %.2f, x2 = %.2f"%(l1,l2), linewidth=1)
                    if(abs(robargdata[0][index]) == 1 and abs(robargdata[1][index]) == 1):
                        x1 = np.array([robargdata[0][index],robargdata[1][index],robargdata[2][index]])
                        qx1 = rappsip.denom(x1)
                        ux = rappsip._scaler.unscale(x1)
                        axarr[r][c].scatter(ux[2],qx1, marker='o',c='black',s=100,alpha = 1.0)

            # axarr[r][c].set_xlabel('$x3$', fontsize = 12)
            # axarr[r][c].set_ylabel('$q(x)$', fontsize = 12)
            # axarr[r][c].legend()
            axarr[r][c].set_title("Iteration: %d"%(index+1))
            index+=1

        l= ("x1 = %.2f, x2 = %.2f"%(lbbb,lbbb),
            "x1 = %.2f, x2 = %.2f"%(lbbb,ubbb),
            "x1 = %.2f, x2 = %.2f"%(ubbb,lbbb),
            "x1 = %.2f, x2 = %.2f"%(ubbb,ubbb)
        )
        fig.legend(l,loc='upper center', ncol=4,bbox_to_anchor=(0.5, 0.93), borderaxespad=0.,shadow=False)




        # plt.scatter(robargdata[0],robargdata[1])

        # outfile = "%s/plots/Piterinfo_robarg_%s%s_p%s_q%s_ts%s.pdf"%(folder, fname,noisestr,m,n,ts )
        # plt.savefig(outfile)
        # print("open %s;"%(outfile))
        # plt.show()
        plt.savefig("/Users/mkrishnamoorthy/Desktop/f23-2.pdf")


# python plottopniterationinfo.py f20_2x ../benchmarkdata/f20_test.txt f20 10 all

if __name__ == "__main__":
    import os, sys
    if len(sys.argv)!=6 and len(sys.argv)!=7:
        print("Usage: {} fname noise m n ts [plot(yes_2 or yes_3)]".format(sys.argv[0]))
        sys.exit(1)
    if(len(sys.argv)==7):
        plot = sys.argv[6]
    else:
        plot = "no"

    plotiterationinfo(sys.argv[1], sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5], plot)
###########
