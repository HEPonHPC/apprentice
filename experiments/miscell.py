import numpy as np
from apprentice import RationalApproximationSIP, RationalApproximationONB, PolynomialApproximation
from apprentice import tools, readData
import matplotlib.ticker as mtick
import os

def getfarr():
    farr = ["f1","f2","f3","f4","f5","f7","f8","f9","f10","f12","f13","f14","f15","f16",
            "f17","f18","f19","f20","f21","f22"]
    # farr = ["f1","f2","f3","f4","f5","f7","f8","f9","f10","f12","f13","f14","f15","f16",
    #         "f17","f18","f19","f21","f22"]
    # farr = ["f20"]

    return farr

def getnoiseinfo(noise):
    noisearr = ["0","10-2","10-4","10-6"]
    noisestr = ["","_noisepct10-2","_noisepct10-4","_noisepct10-6"]
    noisepct = [0,10**-2,10**-4,10**-6]

    for i,n in enumerate(noisearr):
        if(n == noise):
            return noisestr[i],noisepct[i]

def checkiffileexits():
    noiselevels = ['0','10-2','10-6']
    allsamples = ['mc','lhs','so','sg']
    # allsamples = ['mc','lhs']
    # allsamples = ['sg']
    fff = getfarr()
    for snum,sample in enumerate(allsamples):
        for nnum,noise in enumerate(noiselevels):
            noisestr,noisepct = getnoiseinfo(noise)
            for fnum,fname in enumerate(fff):
                ts = "2x"

                for run in ["exp1","exp2","exp3","exp4","exp5"]:
                    fndesc = "%s%s_%s_%s"%(fname,noisestr,sample,ts)
                    folder = "results/%s/%s"%(run,fndesc)
                    m = 5
                    n = 5
                    pq = "p%d_q%d"%(m,n)
                    # print(run, fname,noisestr,sample,m,n)

                    rappsipfile = "%s/outrasip/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)
                    rappfile = "%s/outra/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)
                    rapprdfile = "%s/outrard/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)
                    rapprd1file = "%s/outrard1/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)
                    pappfile = "%s/outpa/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)
                    if not os.path.exists(rappsipfile):
                        print("rappsipfile %s not found\n"%(rappsipfile))


                    if not os.path.exists(rappfile):
                        print("rappfile %s not found\n"%(rappfile))

                    if not os.path.exists(rapprdfile):
                        print("rappfile %s not found\n"%(rapprdfile))

                    # if not os.path.exists(rapprd1file):
                    #     print("rappfile %s not found\n"%(rapprd1file))

                    if not os.path.exists(pappfile):
                        print("pappfile %s not found\n"%(pappfile))

                    if(sample == "sg"):
                        break

def diffrarddegrees():
    noiselevels = ['0','10-2','10-6']
    allsamples = ['mc','lhs','so','sg']
    # allsamples = ['mc','lhs']
    # allsamples = ['sg']
    fff = getfarr()
    for snum,sample in enumerate(allsamples):
        for nnum,noise in enumerate(noiselevels):
            noisestr,noisepct = getnoiseinfo(noise)
            for fnum,fname in enumerate(fff):
                ts = "2x"

                for run in ["exp1","exp2","exp3","exp4","exp5"]:
                    fndesc = "%s%s_%s_%s"%(fname,noisestr,sample,ts)
                    folder = "results/%s/%s"%(run,fndesc)
                    m = 5
                    n = 5
                    pq = "p%d_q%d"%(m,n)
                    # print(run, fname,noisestr,sample,m,n)

                    rapprdfile = "%s/outrard/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)
                    rapprd1file = "%s/outrard1/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)

                    if not os.path.exists(rapprdfile):
                        print("rappfile %s not found\n"%(rapprdfile))
                        if(sample == "sg"):
                            break
                        continue

                    if not os.path.exists(rapprd1file):
                        print("rappfile %s not found\n"%(rapprd1file))
                        if(sample == "sg"):
                            break
                        continue

                    import json
                    if rapprdfile:
                        with open(rapprdfile, 'r') as fn:
                            datastore = json.load(fn)
                    mrd = datastore['m']
                    nrd = datastore['n']

                    if rapprd1file:
                        with open(rapprd1file, 'r') as fn:
                            datastore = json.load(fn)
                    mrd1 = datastore['m']
                    nrd1 = datastore['n']

                    if(mrd !=mrd1 or nrd != nrd1):
                        print("%s %d %d"%(fndesc, mrd,nrd))
                        print("%s %d %d\n"%(fndesc,mrd1,nrd1))


                    if(sample == "sg"):
                        break

def printrarddegree():
    noiselevels = ['10-6']
    # allsamples = ['mc','lhs','so','sg']
    allsamples = ['lhs']
    # allsamples = ['sg']
    fff = getfarr()
    for snum,sample in enumerate(allsamples):
        for nnum,noise in enumerate(noiselevels):
            noisestr,noisepct = getnoiseinfo(noise)
            for fnum,fname in enumerate(fff):
                ts = "2x"

                for run in ["exp1","exp2","exp3","exp4","exp5"]:
                    fndesc = "%s%s_%s_%s"%(fname,noisestr,sample,ts)
                    folder = "results/%s/%s"%(run,fndesc)
                    m = 5
                    n = 5
                    pq = "p%d_q%d"%(m,n)
                    # print(run, fname,noisestr,sample,m,n)

                    rapprdfile = "%s/outrard/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)

                    if not os.path.exists(rapprdfile):
                        print("rappfile %s not found\n"%(rapprdfile))
                        if(sample == "sg"):
                            break
                        continue

                    import json
                    if rapprdfile:
                        with open(rapprdfile, 'r') as fn:
                            datastore = json.load(fn)
                    mrd = datastore['m']
                    nrd = datastore['n']
                    tol = datastore['tol']


                    if(mrd !=5 or nrd != 5):
                        print("%s %f %d %d"%(fndesc, tol, mrd,nrd))



                    if(sample == "sg"):
                        break
def plotpoledata():
    fcorner = "results/plots/poledata_corner2D.csv"
    fin = "results/plots/poledata_inside2D.csv"
    import matplotlib.pyplot as plt

    X1,X2 = tools.readData(fcorner)
    plt.scatter(X1[1:],X2[1:],label="$X^{(corner)}$",s=5)
    # print(np.c_[X1[1:10],X2[1:10]])
    X1,X2 = tools.readData(fin)
    plt.scatter(X1[1:500000],X2[1:500000],label="$X^{(in)}$",s=1)
    plt.legend()
    plt.savefig("results/plots/Ppoledata2D.png")

    X1,X2 = tools.readData(fin)

def renamelogfilesnD(sample = 'sg',dim = 2):
    folder = "f20-"+str(dim)+"D-special_d"+str(dim)+"_l10-6_u4pi"
    m = 5
    n = 5
    import os
    import shutil
    for l in range(1,11):
        if(sample == 'sg'):
            folderplus = folder+"/f20-"+str(dim)+"D_"+sample+"_l"+str(l)
            logf = "%s/log/f20-%dD_%s_l%d_p%d_q%d_tsCp_i0.log"%(folderplus,dim,sample,l,m,n)
        else:
            folderplus = folder+"/f20-"+str(dim)+"D_"+sample
            logf = "%s/log/f20-%dD_%s_p%d_q%d_tsCp_i0.log"%(folderplus,dim,sample,m,n)
        f = open(logf, "r")
        fline = f.readline()
        myList = fline.split(",")
        nlname = myList[1]
        myList = nlname.split("/")
        nlname = myList[2]
        nlname = nlname[:-1]
        print(nlname)
        print(folder+"/level_"+str(l)+"_iter0.nl")
        #os.rename("/tmp/"+ nlname, folder+"/level_"+str(l)+"_iter0.nl")
        shutil.move("/tmp/"+ nlname, folder+"/level_"+str(l)+"_iter0.nl")
        # os.rename(logf,folderplus+"/log/level_"+str(l)+"_iter0.log")
        if(sample != 'sg'):
            break

def renamelogfiles4D():
    folder = "f20_sg_2x"
    import os
    for iter in range(71):
        logf = "%s/log/f20_sg_2x_p5_q5_tsCp_i%d.log"%(folder,iter)
        f = open(logf, "r")
        fline = f.readline()
        myList = fline.split(",")
        nlname = myList[1]
        myList = nlname.split("/")
        nlname = myList[2]
        nlname = nlname[:-1]
        # os.rename(folder + "/"+ nlname, folder+"/log/iter_"+str(iter)+".nl")
        os.rename(logf, folder+"/log/iter_"+str(iter)+".log")

def cubeplot():
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import product, combinations


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("equal")

    vertices = np.empty(shape=(0,3))
    # draw cube
    r = [-1, 1]
    coord = combinations(np.array(list(product(r, r, r))), 2)
    for s, e in coord:
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="b")
        # print(s)
        vertices = np.append(vertices,[s],axis =0)
        vertices = np.append(vertices,[e],axis =0)
    vertices = np.unique(vertices,axis=0)
    print(np.shape(vertices))


    print(vertices)

    npoints = 50
    eps = 0.2
    dim = 3
    from pyDOE import lhs
    import apprentice

    # for v in vertices:
    #     minarr = []
    #     maxarr = []
    #     for p in v:
    #         if(p==1):
    #             maxarr.append(p)
    #             minarr.append(p-eps)
    #         elif(p==-1):
    #             maxarr.append(p+eps)
    #             minarr.append(p)
    #     for d in range(dim):
    #         if minarr[d] == -1+eps:
    #             maxarr[d] == 1
    #         elif maxarr[d] == 1-eps:
    #             minarr[d] == -1
    #         print(v,minarr,maxarr)
    #         s = apprentice.Scaler(np.array(X, dtype=np.float64), a=minarr, b=maxarr)
    #         X = s.scaledPoints
    #         # ax.scatter3D(X[:,0],X[:,1],X[:,2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # plt.show()
    # exit(1)


    Xmain = np.empty([0,3])
    index = -1
    for d in range(dim):
        for x in [-1,1]:
            index+=1
            seedarr = [221,323,545,435,944,664]
            np.random.seed(seedarr[index])
            X = lhs(dim, samples=npoints, criterion='maximin')
            minarr = [0,0,0]
            maxarr = [0,0,0]
            for p in range(dim):
                if(p==d):
                    if x == 1:
                        minarr[p] = x + ((-1*x)/np.abs(x)) * eps
                        maxarr[p] = x
                    else:
                        minarr[p] = x
                        maxarr[p] = x + ((-1*x)/np.abs(x)) * eps
                else:
                    minarr[p] = -1
                    maxarr[p] = 1

            print(minarr,maxarr)
            s = apprentice.Scaler(np.array(X, dtype=np.float64), a=minarr, b=maxarr)
            X = s.scaledPoints
            # X = np.append(X,[[0,0,0]],axis = 0)
            Xmain = np.vstack((Xmain,X))
            ax.scatter3D(X[:,0],X[:,1],X[:,2])
            # if(index == 4): break
        # print(Xmain)
    print(np.shape(Xmain))
    Xmain = np.unique(Xmain,axis = 0)
    print(np.shape(Xmain))


    # minarr = [-1,   1-eps,  -1]
    # maxarr = [1,    1,      1]
    # print(minarr,maxarr)
    # s = apprentice.Scaler(np.array(X, dtype=np.float64), a=minarr, b=maxarr)
    # X = s.scaledPoints
    # ax.scatter3D(X[:,0],X[:,1],X[:,2])

    # minarr = [-1,   -1,  -1]
    # maxarr = [1,    1,      -1+eps]
    # s = apprentice.Scaler(np.array(X, dtype=np.float64), a=minarr, b=maxarr)
    # X = s.scaledPoints
    # ax.scatter3D(X[:,0],X[:,1],X[:,2])

    plt.show()

def plotfnassubplot():
    import matplotlib.pyplot as plt
    rasiplhs = range(20)
    rasiphybridlhs = range(20,40)
    rasipsg = range(40,60)
    rasg = range(60,80)
    plt.figure(0,figsize=(10, 8))
    plots = []


    axfirst = plt.subplot2grid((5,4), (0,0))
    for x,typearr,m in zip(range(1,5),[rasiplhs,rasiphybridlhs,rasipsg,rasg],['o','x','s','*']):
        axfirst.scatter(x,typearr[0],marker=m)
    axfirst.label_outer()

    for i in range(5):
        for j in range(4):
            if(i==0 and j==0): continue
            ax = plt.subplot2grid((5,4), (i,j),sharex=axfirst,sharey=axfirst)
            # xpoints = [1,2,3,4] #f1-f20
            ax.label_outer()
            for x,typearr,m in zip(range(1,5),[rasiplhs,rasiphybridlhs,rasipsg,rasg],['o','x','s','*']):
                ax.scatter(x,typearr[i*4+j],marker=m)


    # plt.subplots_adjust(wspace=0,hspace=0)
    plt.tight_layout()
    plt.show()



    # plt.show()

def splitlhsnpoints():
    from apprentice import tools
    m=5
    n=5
    print ("n\tM\tN\tTP\tFPp\tTFP\tIN\n")
    for dim in range(2,10):
        npoints = 2*tools.numCoeffsRapp(dim,[m,n])
        facepointsper = int(2*tools.numCoeffsRapp(dim-1,[m,n])/(2*dim))
        totalfacepoints = 2*dim*facepointsper
        inpoints = int(npoints - totalfacepoints)

        print("%d\t%d\t%d\t%d\t%d\t%d\t%d"%(dim,m,n,npoints,facepointsper,totalfacepoints,inpoints))
        # for m in range(2,8):
            # for n in range(2,9):


def plotsamplingstrategies():

    folder = "results/exp1/benchmarkdata/f1_"
    lhsfile = folder+"lhs.txt"
    splilhsfile = folder+"splitlhs.txt"
    sgfile   = folder+"sg.txt"
    for file, name in zip([lhsfile,splilhsfile,sgfile],["LHS","d-LHS","SG"]):
        X,Y = tools.readData(file)
        import matplotlib.pyplot as plt
        plt.figure(0,figsize=(15, 10))
        plt.scatter(X[:,0],X[:,1])
        plt.xlabel("$x_1$",fontsize = 32)
        plt.ylabel("$x_2$",fontsize = 32)
        plt.tick_params(labelsize=28)
        plt.savefig("results/plots/"+name+".pdf", bbox_inches='tight')
        plt.clf()
        plt.close('all')





if __name__ == "__main__":
    import sys
    # plotpoledata()
    # printrarddegree()
    # checkiffileexits()
    # diffrarddegrees()
    # renamelogfiles2D()
    # renamelogfiles4D()
    # if len(sys.argv)==2:
    #     renamelogfilesnD(sys.argv[1])
    # if len(sys.argv)==3:
    #     renamelogfilesnD(sys.argv[1],int(sys.argv[2]))
    # else:renamelogfilesnD()

    # cubeplot()
    # plotfnassubplot()
    # splitlhsnpoints()
    plotsamplingstrategies()

 ###########
