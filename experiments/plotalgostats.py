import numpy as np
from apprentice import RationalApproximationSIP
from sklearn.model_selection import KFold
from apprentice import tools, readData
from mpl_toolkits.mplot3d import Axes3D
import os

def plotalgostat(folder, desc, plotdim):
    import glob
    import json
    import re
    filelist = np.array(glob.glob(folder+"/out/*.json"))
    filelist = np.sort(filelist)

    fileArr = np.array([])
    iterationNoArr = np.array([])

    outfilepng = "%s/plots/Palg_%s_%s_from_plotalgostat.png"%(folder, desc,plotdim)

    totaltimearr = np.array([])
    qptimearr  = np.array([])
    mstimearr = np.array([])

    noofitersarr = np.array([])
    noofitersperparr = np.array([])

    marr = np.array([])
    narr = np.array([])
    Marr = np.array([])
    Narr = np.array([])


    for file in filelist:
        if file:
			with open(file, 'r') as fn:
				datastore = json.load(fn)

        totaltime = datastore['log']['fittime']
        totaltimearr = np.append(totaltimearr,totaltime)
        qptime = 0
        mstime = 0
        for iter in datastore['iterationinfo']:
            qptime += iter['log']['time']
            roboinfo = iter["robOptInfo"]["info"]
            mstime += roboinfo[len(roboinfo)-1]["log"]["time"]
        qptimearr = np.append(qptimearr,qptime)
        mstimearr = np.append(mstimearr,mstime)

        noofiters = len(datastore['iterationinfo'])
        noofitersarr = np.append(noofitersarr,noofiters)

        m = datastore['m']
        n = datastore['n']
        M = datastore['M']
        N = datastore['N']

        marr = np.append(marr,m)
        narr = np.append(narr,n)
        Marr = np.append(Marr,M)
        Narr = np.append(Narr,N)

    maxn = int(np.max(narr))
    maxm = int(np.max(marr))


    if(plotdim=="2d"):
        import matplotlib.pyplot as plt
        f, axes = plt.subplots(4,2, sharex=True, sharey='row',figsize=(15,20))
        axes[0][0].scatter(narr,np.ma.log10(totaltimearr),marker = 'o', c = "blue",s=100, alpha = 0.7)
        axes[0][0].set_ylabel('log$_{10}$(CPU time in sec)', fontsize = 12)

        maxtotaltimeperq = np.zeros(maxn+1, dtype=np.float64)
        for i in range(len(narr)):
            n = int(narr[i])
            if(totaltimearr[i] > maxtotaltimeperq[n]):
                maxtotaltimeperq[n] = totaltimearr[i]
        axes[0][1].plot(range(maxn+1),np.ma.log10(maxtotaltimeperq),'b')
        # axes[0][1].set_ylabel('log$_{10}$(max CPU time in sec)', fontsize = 12)

        axes[1][0].scatter(narr,np.ma.log10(qptimearr),marker = 'o', c = "red",s=100, alpha = 0.7)
        axes[1][0].set_ylabel('log$_{10}$(fit time in sec)', fontsize = 12)

        maxqptimeperq = np.zeros(maxn+1, dtype=np.float64)
        for i in range(len(narr)):
            n = int(narr[i])
            if(qptimearr[i] > maxqptimeperq[n]):
                maxqptimeperq[n] = qptimearr[i]
        axes[1][1].plot(range(maxn+1),np.ma.log10(maxqptimeperq),'r')

        axes[2][0].scatter(narr,np.ma.log10(mstimearr),marker = 'o', c = "green",s=100, alpha = 0.7)
        axes[2][0].set_ylabel('log$_{10}$(multistart time in sec)', fontsize = 12)

        maxmstimeperq = np.zeros(maxn+1, dtype=np.float64)
        for i in range(len(narr)):
            n = int(narr[i])
            if(mstimearr[i] > maxmstimeperq[n]):
                maxmstimeperq[n] = mstimearr[i]
        axes[2][1].plot(range(maxn+1),np.ma.log10(maxmstimeperq),'g')

        axes[3][0].scatter(narr,np.ma.log10(noofitersarr),marker = 'o', c = "magenta",s=100, alpha = 0.7)
        axes[3][0].set_ylabel('log$_{10}$(no. of iterations)', fontsize = 12)
        axes[3][0].set_xlabel('n', fontsize = 12)

        colors = ['b','g','r','m','k','c','y','b','g','r','m','k','c','y']
        linestyles = ['-','--','-.',':','-','--','-.',':','-','--','-.',':','-','--']
        thickness = [2.5,2.5,2.5,2.5,2,2,2,2,1.5,1.5,1.5,1.5,1,1,1,1]
        dindex = 0
        if(maxm > len(colors)):
            raise Exception("Plot strategy not equipped to handle m value of %d"%(maxm))

        for m in range(maxm+1):
            itermarr = np.zeros(maxn+1)
            for i in range(len(noofitersarr)):
                if(marr[i] == m):
                    itermarr[int(narr[i])] = noofitersarr[i]
            axes[3][1].plot(range(maxn+1),np.ma.log10(itermarr),color=colors[dindex],linestyle=linestyles[dindex],linewidth=thickness[dindex], label="m = %d"%(m))
            dindex +=1
        from matplotlib.font_manager import FontProperties
        fontP = FontProperties()
        fontP.set_size('small')
        axes[3][1].legend(prop=fontP)
        axes[3][1].set_xlabel('n', fontsize = 12)

        f.suptitle("%s algorithm statistics"%(desc), size=15)
        plt.savefig(outfilepng)
        # plt.show()
        plt.clf()

    elif(plotdim=="3d"):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(15,10))

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot3D(marr,narr,np.ma.log10(totaltimeArr) ,"b.")
        ax.set_xlabel('$m$', fontsize = 12)
        ax.set_ylabel('$n$', fontsize = 12)
        ax.set_zlabel("log$_{10}$(total CPU time in sec)", fontsize = 12)

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.plot3D(marr,narr,np.ma.log10(noofitersArr) ,"r.")
        ax.set_xlabel('$m$', fontsize = 12)
        ax.set_ylabel('$n$', fontsize = 12)
        ax.set_zlabel("log$_{10}$(no. of iterations)", fontsize = 12)


        fig.suptitle("%s algorithm statistics"%(desc), size=15)
        plt.savefig(outfilepng)
        # plt.show()
        plt.clf()
    else:
        raise Exception("plot dimension unknown")

# python plottopniterationinfo.py f20_2x ../benchmarkdata/f20_test.txt f20 10 all

if __name__ == "__main__":
    import os, sys
    if len(sys.argv)!=4:
        print("Usage: {} infolder fndesc 2d_or_3d".format(sys.argv[0]))
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print("Input folder '{}' not found.".format(sys.argv[1]))

    plotalgostat(sys.argv[1], sys.argv[2], sys.argv[3])
###########
