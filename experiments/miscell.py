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

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    folder = "results/exp1/benchmarkdata/f1_"
    lhsfile = folder+"lhs.txt"
    splilhsfile = folder+"splitlhs.txt"
    sgfile   = folder+"sg.txt"
    for file, name in zip([lhsfile,splilhsfile,sgfile],["LHS","d-LHS","SG"]):
        X,Y = tools.readData(file)
        mpl.rc('text', usetex = True)
        mpl.rc('font', family = 'serif', size=12)
        mpl.rc('font', weight='bold')
        mpl.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
        # mpl.style.use("ggplot")
        plt.figure(0,figsize=(15, 10))
        plt.scatter(X[:,0],X[:,1],s = 100,c='r')
        plt.xlabel("$x_1$",fontsize = 44)
        plt.ylabel("$x_2$",fontsize = 44)
        plt.tick_params(labelsize=28)
        plt.savefig("../../log/"+name+".pdf", bbox_inches='tight')
        plt.clf()
        plt.close('all')

def plotminimizeranderror(usejson = 0):
    def getData(X_train, fn, noisepct):
        """
        TODO use eval or something to make this less noisy
        """
        from apprentice import testData
        if fn=="f17":
            Y_train = [testData.f17(x) for x in X_train]
        else:
            raise Exception("function {} not implemented, exiting".format(fn))

        return Y_train


    import json
    import apprentice
    from apprentice import RationalApproximationSIP
    folder = "results"
    samplearr = ['lhs','splitlhs','sg']
    noise = "0"
    fname = "f17"
    noisestr,noisepct = getnoiseinfo(noise)
    dim = 3
    infile=[
            "results/plots/poledata_corner"+str(dim)+"D.csv",
            "results/plots/poledata_inside"+str(dim)+"D.csv"
            ]



    if(usejson == 0):
        minarr  = [80,5,90]
        maxarr  = [100,10,93]
        X_testfc = np.loadtxt(infile[0], delimiter=',')
        X_testin = np.loadtxt(infile[1], delimiter=',')
        s = apprentice.Scaler(np.array(X_testfc, dtype=np.float64), a=minarr, b=maxarr)
        X_testfc = s.scaledPoints

        s = apprentice.Scaler(np.array(X_testin, dtype=np.float64), a=minarr, b=maxarr)
        X_testin = s.scaledPoints

        Y_testfc = np.array(getData(X_testfc,fname,0))
        Y_testin = np.array(getData(X_testin,fname,0))

        Y_testall = np.concatenate((Y_testfc,Y_testin), axis=None)


        maxiterexp = [0,0,0]
        for snum,sample in enumerate(samplearr):
            maxiter = 0
            for exp in ['exp1','exp2','exp3','exp4','exp5']:
                file = "%s/%s/%s%s_%s_2x/outrasip/%s%s_%s_2x_p5_q5_ts2x.json"%(folder,exp,fname,noisestr,sample,fname,noisestr,sample)
                if not os.path.exists(file):
                    print("rappsipfile %s not found"%(file))
                    exit(1)
                if file:
                    with open(file, 'r') as fn:
                        datastore = json.load(fn)
                if len(datastore['iterationinfo']) > maxiter:
                    maxiter = len(datastore['iterationinfo'])
                    maxiterexp[snum] = exp
                if(sample == 'sg'):
                    break


        data = {}
        for snum,sample in enumerate(samplearr):
            data[sample]  = {}
            exp = maxiterexp[snum]

            file = "%s/%s/%s%s_%s_2x/outrasip/%s%s_%s_2x_p5_q5_ts2x.json"%(folder,exp,fname,noisestr,sample,fname,noisestr,sample)
            if not os.path.exists(file):
                print("rappsipfile %s not found"%(file))
                exit(1)
            if file:
                with open(file, 'r') as fn:
                    datastore = json.load(fn)

            data[sample]['x'] = [i for i in range(len(datastore['iterationinfo']))]
            data[sample]['minimizer'] = []
            data[sample]['error'] = []
            print('starting error calc for %s'%(sample))
            for inum,iter in enumerate(datastore['iterationinfo']):
                print('staring iter %d'%(inum))
                data[sample]['minimizer'].append(iter['robOptInfo']['info'][0]['robustObj'])

                pcoeff = iter['pcoeff']
                qcoeff = iter['qcoeff']

                if file:
                    with open(file, 'r') as fn:
                        tempds = json.load(fn)
                tempds['pcoeff'] = pcoeff
                tempds['qcoeff'] = qcoeff

                rappsip = RationalApproximationSIP(tempds)

                Y_pred_rappsipfc = rappsip.predictOverArray(X_testfc)
                Y_pred_rappsipin = rappsip.predictOverArray(X_testin)

                Y_pred_rappsipall = np.concatenate((Y_pred_rappsipfc,Y_pred_rappsipin), axis=None)
                l2allrappsip = np.sum((Y_pred_rappsipall-Y_testall)**2)
                data[sample]['error'].append(np.sqrt(l2allrappsip))
                print('ending iter %d'%(inum))
        print(data)
        import json
        with open('results/plots/Jminimizeranderror.json', "w") as f:
            json.dump(data, f,indent=4, sort_keys=True)


    elif(usejson == 1):
        outfilejson = "results/plots/Jminimizeranderror.json"

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.text as text
        mpl.rc('text', usetex = True)
        mpl.rc('font', family = 'serif', size=12)
        mpl.rc('font', weight='bold')
        mpl.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
        # mpl.style.use("ggplot")

        f, axarr = plt.subplots(2,1, sharex=True, sharey=False, figsize=(15,8))
        f.subplots_adjust(hspace=0)
        f.subplots_adjust(wspace=0)

        style  =['b--','r-.','g-']
        linewidth = [1,1,2]
        labelarr = ['$LHS$','$\\mathrm{d-LHD}$','$SG$']
        marker = ['x','*','o']

        index = 0
        if outfilejson:
            with open(outfilejson, 'r') as fn:
                data = json.load(fn)
        for snum,sample in enumerate(samplearr):
            x = data[sample]['x']
            y = data[sample]['minimizer']
            x.insert(0,-1)
            y.insert(0,-2.5)
            x = np.array(x)+1
            axarr[index].plot(x,y,style[snum],label=labelarr[snum],
                lineWidth=linewidth[snum],markevery=(1,1),marker = marker[snum])
        axarr[index].axhline(0,linestyle=":",linewidth='1',color='k')
        axarr[index].legend(fontsize = 18,frameon=False)
        axarr[index].set_ylabel('$min\\quad q(x)$',fontsize = 24)
        axarr[index].tick_params(labelsize=20)

        index = 1
        if outfilejson:
            with open(outfilejson, 'r') as fn:
                data = json.load(fn)
        for snum,sample in enumerate(samplearr):
            x = data[sample]['x']
            y = data[sample]['error']
            x.insert(0,-1)
            y.insert(0,10**2)
            x = np.array(x)+1
            axarr[index].plot(x,y,style[snum],label=labelarr[snum],
                lineWidth=linewidth[snum],markevery=(1,1),marker = marker[snum])
            if(sample == 'splitlhs'):
                min = np.min(y)
        axarr[index].axhline(min,linestyle=":",linewidth='1',color='k')
        axarr[index].set_yscale('log')
        axarr[index].legend(fontsize = 18,frameon=False)
        axarr[index].set_xlabel('$\\mathrm{Iteration\\ number}$',fontsize = 24)
        axarr[index].set_ylabel('$\\Delta_r$',fontsize = 24)
        axarr[index].tick_params(labelsize=20)



        # plt.yscale("log")

        plt.savefig("../../log/minimizererror.pdf", bbox_inches='tight')








def plotfnamepoles():
    import json
    folder = "results"
    sample = 'lhs'
    noise = "0"
    fname = "f17"
    noisestr,noisepct = getnoiseinfo(noise)
    data = {}
    maxiter = 0
    maxiterexp = ""
    for exp in ['exp1','exp2','exp3','exp4','exp5']:
        file = "%s/%s/%s%s_%s_2x/outrasip/%s%s_%s_2x_p5_q5_ts2x.json"%(folder,exp,fname,noisestr,sample,fname,noisestr,sample)
        if not os.path.exists(file):
            print("rappsipfile %s not found"%(file))
            exit(1)
        if file:
            with open(file, 'r') as fn:
                datastore = json.load(fn)
        # print(exp,len(datastore['iterationinfo']))
        if len(datastore['iterationinfo']) > maxiter:
            maxiter = len(datastore['iterationinfo'])
            maxiterexp = exp

    file = "%s/%s/%s%s_%s_2x/outrasip/%s%s_%s_2x_p5_q5_ts2x.json"%(folder,maxiterexp,fname,noisestr,sample,fname,noisestr,sample)
    if not os.path.exists(file):
        print("rappsipfile %s not found"%(file))
        exit(1)
    if file:
        with open(file, 'r') as fn:
            datastore = json.load(fn)
    maxnum = 0
    for iter in datastore['iterationinfo']:
        # print(iter['robOptInfo']['robustArg'])
        for num,x in enumerate(iter['robOptInfo']['robustArg']):
            data[num] = []
            maxnum = num+1
        break
    for iter in datastore['iterationinfo']:
        for num,x in enumerate(iter['robOptInfo']['robustArg']):
            data[num].append(x)
    X = np.stack((data[0],data[1],data[2]),axis=-1)
    outfile = folder+"/plots/Cpoledata.csv"
    np.savetxt(outfile, X, delimiter=",")


def runfacevsinner():
    def getData(X_train, fn, noisepct,seed):
        """
        TODO use eval or something to make this less noisy
        """
        from apprentice import testData
        if fn=="f18":
            Y_train = [testData.f18(x) for x in X_train]
        elif fn=="f20":
            Y_train = [testData.f20(x) for x in X_train]
        else:
            raise Exception("function {} not implemented, exiting".format(fn))
        np.random.seed(seed)

        stdnormalnoise = np.zeros(shape = (len(Y_train)), dtype =np.float64)
        for i in range(len(Y_train)):
            stdnormalnoise[i] = np.random.normal(0,1)
        # return Y_train
        return np.atleast_2d(np.array(Y_train)*(1+ noisepct*stdnormalnoise))

    def getbox(f):
        minbox = []
        maxbox = []
        if(f=="f18"):
            minbox  = [-0.95,-0.95,-0.95,-0.95]
            maxbox  = [0.95,0.95,0.95,0.95]
        elif(f=="f20"):
            minbox  = [10**-6,10**-6,10**-6,10**-6]
            maxbox  = [4*np.pi,4*np.pi,4*np.pi,4*np.pi]
        else:
            minbox  = [-1,-1]
            maxbox = [1,1]
        return minbox,maxbox

    from apprentice import tools
    data = {'info':[]}
    import apprentice
    dim = 4
    seedarr = [54321,456789,9876512,7919820,10397531]

    m = 5
    n = 5
    tstimes = 2
    ts = "2x"
    # fname = "f20-"+str(dim)+"D_ts"+ts
    fname = "f20-"+str(dim)+"D"
    from apprentice import tools
    from pyDOE import lhs
    npoints = tstimes * tools.numCoeffsRapp(dim,[m,n])
    print(npoints)
    epsarr = []
    for d in range(dim):
        #epsarr.append((maxarr[d] - minarr[d])/10)
        epsarr.append(10**-6)


    facespctarr = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    folder = "f18_f20_facevsinner"
    for numex,ex in enumerate(["exp1","exp2","exp3","exp4","exp5"]):
        seed = seedarr[numex]
        for facenum,facepct in enumerate(facespctarr):
            facepoints = int(np.ceil(npoints * facepct))
            insidepoints = int(npoints - facepoints)
            # print(insidepoints)
            # Generate inside points
            for fname in ['f18','f20']:
                Xmain = np.empty([0,dim])
                minarrinside = []
                maxarrinside = []
                minarr,maxarr = getbox(fname)
                if(insidepoints>1):
                    for d in range(dim):
                        minarrinside.append(minarr[d] + epsarr[d])
                        maxarrinside.append(maxarr[d] - epsarr[d])
                    X = lhs(dim, samples=insidepoints, criterion='maximin')
                    s = apprentice.Scaler(np.array(X, dtype=np.float64), a=minarrinside, b=maxarrinside)
                    X = s.scaledPoints
                    Xmain = np.vstack((Xmain,X))

                #Generate face points
                perfacepoints = int(np.ceil(facepoints/(2*dim)))
                if(perfacepoints>1):
                    index = 0
                    for d in range(dim):
                        for e in [minarr[d],maxarr[d]]:
                            index+=1
                            np.random.seed(seed+index*100)
                            X = lhs(dim, samples=perfacepoints, criterion='maximin')
                            minarrface = np.empty(shape=dim,dtype=np.float64)
                            maxarrface = np.empty(shape=dim,dtype=np.float64)
                            for p in range(dim):
                                if(p==d):
                                    if e == maxarr[d]:
                                        minarrface[p] = e - epsarr[d]
                                        maxarrface[p] = e
                                    else:
                                        minarrface[p] = e
                                        maxarrface[p] = e + epsarr[d]
                                else:
                                    minarrface[p] = minarr[p]
                                    maxarrface[p] = maxarr[p]
                            s = apprentice.Scaler(np.array(X, dtype=np.float64), a=minarrface, b=maxarrface)
                            X = s.scaledPoints
                            Xmain = np.vstack((Xmain,X))
                Xmain = np.unique(Xmain,axis = 0)
                X = Xmain
                # formatStr = "{0:0%db}"%(dim)
                # for d in range(2**dim):
                #     binArr = [int(x) for x in formatStr.format(d)[0:]]
                #     val = []
                #     for i in range(dim):
                #         if(binArr[i] == 0):
                #             val.append(minarr[i])
                #         else:
                #             val.append(maxarr[i])
                #     X[d] = val
                if not os.path.exists(folder+"/"+ex+'/benchmarkdata'):
                    os.makedirs(folder+"/"+ex+'/benchmarkdata',exist_ok = True)
                noise = "0"
                noisestr,noisepct = getnoiseinfo(noise)

                Y = getData(X, fn=fname, noisepct=noisepct,seed=seed)
                infile = "%s/%s/benchmarkdata/%s%s_splitlhs_f%d_i%d.txt"%(folder,ex,fname,noisestr,facepoints,insidepoints)
                print(infile)
                np.savetxt(infile, np.hstack((X,Y.T)), delimiter=",")

                folderplus = "%s/%s/%s%s_splitlhs"%(folder,ex,fname,noisestr)
                fndesc = "%s%s_splitlhs_f%d_i%d"%(fname,noisestr,facepoints,insidepoints)
                if not os.path.exists(folderplus + "/outrasip"):
                    os.makedirs(folderplus + "/outrasip",exist_ok = True)
                if not os.path.exists(folderplus + "/log/consolelograsip"):
                    os.makedirs(folderplus + "/log/consolelograsip",exist_ok = True)
                m = str(m)
                n = str(n)
                consolelog=folderplus + "/log/consolelograsip/"+fndesc+"_p"+m+"_q"+n+"_ts2x.log";
                outfile = folderplus + "/outrasip/"+fndesc+"_p"+m+"_q"+n+"_ts2x.json";
                data['info'].append({'exp':ex,'fname':fname,'outfile':outfile,'facepoints':facepoints,'insidepoints':insidepoints})
                penaltyparam = 0
                cmd = 'nohup python runrappsip.py %s %s %s %s Cp %f %s %s >%s 2>&1 &'%(infile,fndesc,m,n,penaltyparam,folderplus,outfile,consolelog)
                # print(cmd)
                # exit(1)
                os.system(cmd)

    import json
    with open(folder+"/data.json", "w") as f:
        json.dump(data, f,indent=4, sort_keys=True)

def analyzefacevsinner():
    folder = "f18_f20_facevsinner"
    m = 5
    n = 5
    dim=4
    tstimes = 2
    ts = "2x"
    # fname = "f20-"+str(dim)+"D_ts"+ts
    fname = "f20-"+str(dim)+"D"
    from apprentice import tools
    from pyDOE import lhs
    npoints = tstimes * tools.numCoeffsRapp(dim,[m,n])
    facespctarr = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    import json
    file = folder + "/data.json"
    if file:
        with open(file, 'r') as fn:
            data = json.load(fn)
    X = []
    Y18m =[]
    Y18s =[]
    Y20m =[]
    Y20s =[]
    for facenum,facepct in enumerate(facespctarr):
        facepoints = int(np.ceil(npoints * facepct))
        insidepoints = int(npoints - facepoints)
        X.append(facepoints/insidepoints)
    for facenum,facepct in enumerate(facespctarr):
        facepoints = int(np.ceil(npoints * facepct))
        insidepoints = int(npoints - facepoints)
        f18iter=[]
        f20iter=[]
        for numex,ex in enumerate(["exp1","exp2","exp3","exp4","exp5"]):
        # for numex,ex in enumerate(["exp3"]):
            f18file = ""
            f20file = ""
            for d in data['info']:
                if(d['exp'] == ex and d['facepoints'] == facepoints and d['insidepoints'] == insidepoints
                    and d['fname'] == 'f18'):
                    f18file = d['outfile']

                if(d['exp'] == ex and d['facepoints'] == facepoints and d['insidepoints'] == insidepoints
                    and d['fname'] == 'f20'):
                    f20file = d['outfile']
            if not os.path.exists(f18file):
                print("f18 file not found: %s"%(f18file))
                exit(1)
            if not os.path.exists(f20file):
                print("f20 file not found: %s"%(f20file))
                exit(1)
            import json
            if f18file:
                with open(f18file, 'r') as fn:
                    datastore = json.load(fn)
            f18iter.append(np.log10(len(datastore['iterationinfo'])))
            if f20file:
                with open(f20file, 'r') as fn:
                    datastore = json.load(fn)
            f20iter.append(np.log10(len(datastore['iterationinfo'])))



        Y18m.append(np.average(f18iter))
        Y18s.append(np.std(f18iter))
        Y20m.append(np.average(f20iter))
        Y20s.append(np.std(f20iter))

    print(X)
    print(Y18m)
    print(Y20m)
    # Y18m = np.log10(Y18m)
    # Y20m = np.log10(Y20m)
    import matplotlib.pyplot as plt
    plt.figure(0,figsize=(15, 10))
    plt.plot(X[1:],Y18m[1:],label='\\ref{fn:f18}')
    plt.errorbar(X[1:],Y18m[1:], yerr=Y18s[1:], fmt='-o')
    plt.plot(X[1:],Y20m[1:],label='\\ref{fn:f20}')
    plt.errorbar(X[1:],Y20m[1:], yerr=Y20s[1:], fmt='-o')
    # plt.xlabel("$x_1$",fontsize = 32)
    # plt.ylabel("$x_2$",fontsize = 32)
    plt.tick_params(labelsize=28)
    plt.savefig("../../log/facevsinnter.pdf", bbox_inches='tight')
    # plt.show()
    # plt.clf()
    # plt.close('all')







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
    # plotsamplingstrategies()
    # plotfnamepoles()
    # runfacevsinner()
    # analyzefacevsinner()
    if len(sys.argv)==2:
        plotminimizeranderror(int(sys.argv[1]))
    else:
        plotminimizeranderror()

 ###########
