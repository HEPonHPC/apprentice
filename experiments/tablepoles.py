import numpy as np
from apprentice import RationalApproximationSIP, RationalApproximation, PolynomialApproximation
from apprentice import tools, readData
import os

def tablepoles(farr,noisearr, tarr, testfilearr, bottomallarr, ts, table_or_latex):
    print farr
    print noisearr
    print thresholdarr
    print testfilearr
    print bottomallarr

    thresholdvalarr = np.array([float(t) for t in tarr])
    thresholdvalarr = np.sort(thresholdvalarr)

    results = {}

    # import glob
    import json
    # import re
    if not os.path.exists("plots"):
        os.mkdir('plots')
    for num,fname in enumerate(farr):
        results[fname] = {}
        testfile = testfilearr[num]
        bottom_or_all = bottomallarr[num]
        try:
            X, Y = readData(testfile)
        except:
            DATA = tools.readH5(testfile, [0])
            X, Y= DATA[0]

        if(bottom_or_all == "bottom"):
            testset = [i for i in range(trainingsize,len(X_test))]
            X_test = X[testset]
            Y_test = Y[testset]
        else:
            X_test = X
            Y_test = Y

        maxY_test = max(Y_test)
        for noise in noisearr:
            noisestr = ""
            if(noise!="0"):
                noisestr = "_noisepct"+noise
            folder = "%s%s_%s"%(fname,noisestr,ts)

            optjsonfile = folder+"/plots/Joptdeg_"+fname+noisestr+"_jsdump.json"

            if not os.path.exists(optjsonfile):
                print("optjsonfile: " + optjsonfile+ " not found")
                continue

            if optjsonfile:
                with open(optjsonfile, 'r') as fn:
                    optjsondatastore = json.load(fn)

            optm = optjsondatastore['optdeg']['m']
            optn = optjsondatastore['optdeg']['n']

            rappsipfile = "%s/out/%s%s_%s_p%d_q%d_ts%s.json"%(folder,fname,noisestr,ts,optm,optn,ts)
            rappfile = "%s/outra/%s%s_%s_p%d_q%d_ts%s.json"%(folder,fname,noisestr,ts,optm,optn,ts)

            if not os.path.exists(rappsipfile):
                print("rappsipfile %s not found"%(rappsipfile))
                exit(1)

            if not os.path.exists(rappfile):
                print("rappfile %s not found"%(rappfile))
                exit(1)

            rappsip = RationalApproximationSIP(rappsipfile)
            Y_pred_rappsip = rappsip.predictOverArray(X_test)
            rapp = RationalApproximation(fname=rappfile)
            Y_pred_rapp = np.array([rapp(x) for x in X_test])

            results[fname][noise] = {"rapp":{},"rappsip":{}}

            for tval in thresholdvalarr:
                # print(fname, maxY_test)
                # print(Y_pred_rappsip)

                # rappsipcount = ((sum(abs(i)/abs(maxY_test) >= tval for i in Y_pred_rappsip))/float(len(Y_test))) *100
                # rappcount = ((sum(abs(i)/abs(maxY_test) >= tval for i in Y_pred_rapp))/float(len(Y_test))) *100

                rappsipcount = sum(abs(i)/abs(maxY_test) >= tval for i in Y_pred_rappsip)
                rappcount = sum(abs(i)/abs(maxY_test) >= tval for i in Y_pred_rapp)


                # rappsipcount = sum(abs(i) >= tval for i in Y_pred_rappsip)
                # rappcount = sum(abs(i) >= tval for i in Y_pred_rapp)

                # print("----------------")
                # print(maxY_test,tval)
                # for i in Y_pred_rappsip:
                #     if abs(i)/abs(maxY_test) >= tval:
                #         print(abs(i))
                # for i in Y_pred_rapp:
                #     if abs(i)/abs(maxY_test) >= tval:
                #         print(abs(i))

                tvalstr = str(tval)
                results[fname][noise]["rapp"][tvalstr] = rappcount
                results[fname][noise]["rappsip"][tvalstr] = rappsipcount


    print (json.dumps(results,indent=4, sort_keys=True))


    s = ""
    if(table_or_latex == "table"):
        s+= "\t\t\t"
        for noise in noisearr:
            s+= "%s\t\t\t\t\t\t\t"%(noise)
        s+="\n"
        for noise in noisearr:
            s += "\t\tRat Apprx\tRat Apprx SIP\t\t"
        s+="\n\n"
        for noise in noisearr:
            for tval in thresholdvalarr:
                s += "\t%d"%(int(tval))
            s+="\t"
            for tval in thresholdvalarr:
                s += "\t%d"%(int(tval))
            s+="\t"
        s += "\n"
        for fname in farr:
            s += "%s"%(fname)
            for noise in noisearr:
                for tval in thresholdvalarr:
                    tvalstr = str(tval)
                    s += "\t%d"%(int(results[fname][noise]["rapp"][tvalstr]))
                s+="\t"
                for tval in thresholdvalarr:
                    tvalstr = str(tval)
                    s += "\t%d"%(int(results[fname][noise]["rappsip"][tvalstr]))
                s+="\t"
            s+="\n"
    elif(table_or_latex =="latex"):
        for fname in farr:
            s += "\\ref{fn:%s}"%(fname)
            for noise in noisearr:
                for tval in thresholdvalarr:
                    tvalstr = str(tval)
                    s += "&%d"%(int(results[fname][noise]["rapp"][tvalstr]))
                for tval in thresholdvalarr:
                    tvalstr = str(tval)
                    s += "&%d"% (int(results[fname][noise]["rappsip"][tvalstr]))
            s+="\\\\\hline\n"



    print(s)










    exit(1)
    import json
    if infile:
        with open(infile, 'r') as fn:
            datastore = json.load(fn)
    dim = datastore['dim']
    if(dim != 2):
        raise Exception("plot2Dsurface can only handle dim = 2")
    m = datastore['m']
    try:
        n = datastore['n']
    except:
        n=0
    try:
        ts = datastore['trainingscale']
    except:
        ts = ""
    trainingsize = datastore['trainingsize']


    X, Y = readData(testfile)
    if(bottom_or_all == "bottom"):
        testset = [i for i in range(trainingsize,len(X_test))]
        X_test = X[testset]
        Y_test = Y[testset]
    else:
        X_test = X
        Y_test = Y

    if not os.path.exists(folder+"/plots"):
        os.mkdir(folder+'/plots')

    outfilepng = "%s/plots/P2d_%s_p%d_q%d_ts%s_2Dsurface.png"%(folder,desc,m,n,ts)

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(2, 2, 1, projection='3d')

    ax.plot3D(X_test[:,0],X_test[:,1], Y[:],"r.")
    ax.set_xlabel('$x1$', fontsize = 12)
    ax.set_ylabel('$x2$', fontsize = 12)
    ax.set_zlabel('$y$', fontsize = 12)
    ax.set_title('Original data', fontsize = 13)
    nnzthreshold = 1e-6
    for i, p in enumerate(datastore['pcoeff']):
        if(abs(p)<nnzthreshold):
            datastore['pcoeff'][i] = 0.
    if('qcoeff' in datastore):
        for i, q in enumerate(datastore['qcoeff']):
            if(abs(q)<nnzthreshold):
                datastore['qcoeff'][i] = 0.

    try:
        rappsip = RationalApproximationSIP(datastore)
        Y_pred = rappsip.predictOverArray(X_test)
    except:
        if(n==0):
            papp = PolynomialApproximation(initDict=datastore)
            Y_pred = np.array([papp(x) for x in X_test])
        else:
            rapp = RationalApproximation(initDict=datastore)
            Y_pred = np.array([rapp(x) for x in X_test])

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.plot3D(X_test[:,0],X_test[:,1], Y_pred[:],"b.")
    ax.set_xlabel('$x1$', fontsize = 12)
    ax.set_ylabel('$x2$', fontsize = 12)
    ax.set_zlabel('$y$', fontsize = 12)
    ax.set_title('Predicted data', fontsize = 13)

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.plot3D(X_test[:,0],X_test[:,1], np.absolute(Y_pred-Y_test),"g.")
    ax.set_xlabel('$x1$', fontsize = 12)
    ax.set_ylabel('$x2$', fontsize = 12)
    ax.set_zlabel('$y$', fontsize = 12)
    ax.set_title('|Predicted - Original|', fontsize = 13)

    l1 = np.sum(np.absolute(Y_pred-Y_test))
    l2 = np.sqrt(np.sum((Y_pred-Y_test)**2))
    linf = np.max(np.absolute(Y_pred-Y_test))
    if(linf>10**3): print("FOUND===>%f"%(linf))
    try:
        nnz = tools.numNonZeroCoeff(rappsip,nnzthreshold)
    except:
        if(n==0):
            nnz = tools.numNonZeroCoeff(papp,nnzthreshold)
        else:
            nnz = tools.numNonZeroCoeff(rapp,nnzthreshold)

    # print(l2)
    # print(nnz)
    # print(l2/nnz)
    # print(np.log10(l2/nnz))
    fig.suptitle("%s. m = %d, n = %d, ts = %d (%s). l1 = %.4f, l2 = %.4f, linf = %.4f, nnz = %d, l2/nnz = %f"%(desc,m,n,trainingsize,ts,l1,l2,linf,nnz,l2/nnz))

    plt.savefig(outfilepng)
    plt.close('all')

# python plot2Dsurface.py f21_2x/out/f21_2x_p12_q12_ts2x.json ../benchmarkdata/f21_test.txt f21_2x f21_2x all

if __name__ == "__main__":

    # import apprentice
    # name = "f14"
    # noisestr = "_noisepct10-3"
    # # noisestr = ""
    # trainfile = "../benchmarkdata/"+name+noisestr+".txt"
    # X, Y = readData(trainfile)
    # folder = "poletest"
    # if not os.path.exists(folder):
    #     os.mkdir(folder)
    # for m in range(1,6):
    #     for n in range(1,6):
    #         trainingsize = 2 * tools.numCoeffsRapp(2,(m,n))
    #         i_train = [i for i in range(trainingsize)]
    #         rapp = apprentice.RationalApproximation(X[i_train],Y[i_train],order=(m,n), strategy=1)
    #
    #         # rappsip  = apprentice.RationalApproximationSIP(X[i_train],Y[i_train],m=m,n=n,trainingscale="Cp",
    #         #                     strategy=0,roboptstrategy = 'msbarontime',fitstrategy = 'filter',localoptsolver = 'scipy')
    #         # rappsip.save(folder+"/rappsip.json")
    #
    #
    #         rapp.save(folder+"/rapp.json")
    #
    #         testfile = "../benchmarkdata/"+name+"_test.txt"
    #
    #         plot2Dsurface(folder+"/rapp.json", testfile, folder, name+noisestr+"_rapp","all")
    #
    #         rappsipfile = "%s%s_2x/out/%s%s_2x_p%d_q%d_ts2x.json"%(name,noisestr,name,noisestr,m,n)
    #         plot2Dsurface(rappsipfile, testfile, folder, name+noisestr+"_rappsip","all")
    #
    #         # plot2Dsurface(folder+"/rappsip.json", testfile, folder, name+noisestr+"_rappsip","all")
    #
    # exit(1)





    import os, sys
    if len(sys.argv) != 8:
        print("Usage: {} function noise thresholds ts testfilelist bottom_or_all table_or_latex".format(sys.argv[0]))
        sys.exit(1)

    farr = sys.argv[1].split(',')
    if len(farr) == 0:
        print("please specify comma saperated functions")
        sys.exit(1)

    noisearr = sys.argv[2].split(',')
    if len(noisearr) == 0:
        print("please specify comma saperated noise levels")
        sys.exit(1)

    thresholdarr = sys.argv[3].split(',')
    if len(thresholdarr) == 0:
        print("please specify comma saperated threshold levels")
        sys.exit(1)

    testfilearr = sys.argv[5].split(',')
    if len(testfilearr) == 0:
        print("please specify comma saperated testfile paths")
        sys.exit(1)

    bottomallarr = sys.argv[6].split(',')
    if len(bottomallarr) == 0:
        print("please specify comma saperated bottom or all options")
        sys.exit(1)


    tablepoles(farr,noisearr, thresholdarr, testfilearr, bottomallarr,sys.argv[4],sys.argv[7])
###########
