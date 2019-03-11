
import numpy as np
from apprentice import RationalApproximationSIP, RationalApproximation, PolynomialApproximation
from apprentice import tools, readData
import os


def suppresscoeffs(datastore, nnzthreshold):
    nnzthreshold = 1e-6
    for i, p in enumerate(datastore['pcoeff']):
        if(abs(p)<nnzthreshold):
            datastore['pcoeff'][i] = 0.
    if('qcoeff' in datastore):
        for i, q in enumerate(datastore['qcoeff']):
            if(abs(q)<nnzthreshold):
                datastore['qcoeff'][i] = 0.

def calculatetesterror(Y_test,Y_pred,app,nnzthreshold):
    nnz = float(tools.numNonZeroCoeff(app,nnzthreshold))
    l2 = np.sqrt(np.sum((Y_pred-Y_test)**2))
    ret =  l2 / nnz
    return ret

def tablecompareall(farr,noisearr, testfilearr, bottomallarr, ts, table_or_latex):
    print farr
    print noisearr
    print testfilearr
    print bottomallarr

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

        for noise in noisearr:
            noisestr = ""
            if(noise!="0"):
                noisestr = "_noisepct"+noise
            folder = "%s%s_%s"%(fname,noisestr,ts)

            optjsonfile = folder+"/plots/Joptdeg_"+fname+noisestr+"_jsdump.json"

            if not os.path.exists(optjsonfile):
                print("optjsonfile: " + optjsonfile+ " not found")
                exit(1)

            if optjsonfile:
                with open(optjsonfile, 'r') as fn:
                    optjsondatastore = json.load(fn)

            optm = optjsondatastore['optdeg']['m']
            optn = optjsondatastore['optdeg']['n']

            rappsipfile = "%s/out/%s%s_%s_p%d_q%d_ts%s.json"%(folder,fname,noisestr,ts,optm,optn,ts)
            rappfile = "%s/outra/%s%s_%s_p%d_q%d_ts%s.json"%(folder,fname,noisestr,ts,optm,optn,ts)
            pafile = "%s/outpa/%s%s_%s_p%d_q%d_ts%s.json"%(folder,fname,noisestr,ts,optm,optn,ts)



            if not os.path.exists(rappsipfile):
                print("rappsipfile %s not found"%(rappsipfile))
                exit(1)

            if not os.path.exists(rappfile):
                print("rappfile %s not found"%(rappfile))
                exit(1)

            if not os.path.exists(pafile):
                print("pappfile %s not found"%(pafile))
                exit(1)

            nnzthreshold =1e-6

            if rappsipfile:
                with open(rappsipfile, 'r') as fn:
                    datastore = json.load(fn)
            suppresscoeffs(datastore,nnzthreshold)
            rappsip = RationalApproximationSIP(datastore)
            Y_pred_rappsip = rappsip.predictOverArray(X_test)
            te_rappsip = calculatetesterror(Y_test,Y_pred_rappsip,rappsip,nnzthreshold)



            if rappfile:
                with open(rappfile, 'r') as fn:
                    datastore = json.load(fn)
            suppresscoeffs(datastore,nnzthreshold)
            rapp = RationalApproximation(initDict=datastore)
            Y_pred_rapp = np.array([rapp(x) for x in X_test])
            te_rapp = calculatetesterror(Y_test,Y_pred_rapp,rapp,nnzthreshold)

            if pafile:
                with open(pafile, 'r') as fn:
                    datastore = json.load(fn)
            suppresscoeffs(datastore,nnzthreshold)
            papp = PolynomialApproximation(initDict=datastore)
            Y_pred_papp = np.array([papp(x) for x in X_test])
            te_papp = calculatetesterror(Y_test,Y_pred_papp,papp,nnzthreshold)

            results[fname][noise] = {"rapp":te_rapp, "rappsip":te_rappsip, "papp":te_papp}


    s = ""
    if(table_or_latex == "table"):
        s+= "\t\t\t"
        for noise in noisearr:
            s+= "%s\t\t\t\t\t\t\t"%(noise)
        s+="\n"
        for noise in noisearr:
            s += "\t\tRat Apprx SIP\tRat Apprx\tPoly App\t\t"
        s+="\n\n"
        for fname in farr:
            s += "%s"%(fname)
            for noise in noisearr:
                s += "\t%.4f"%(results[fname][noise]["rappsip"])
                s+="\t"
                s += "\t%.4f"%(results[fname][noise]["rapp"])
                s+="\t"
                s += "\t%.4f"%(results[fname][noise]["papp"])
                s+="\t"
            s+="\n"
    elif(table_or_latex =="latex"):
        for fname in farr:
            s += "\\ref{fn:%s}"%(fname)
            for noise in noisearr:
                s += "&%.6f"%(results[fname][noise]["rappsip"])
                s += "&%.6f"%(results[fname][noise]["rapp"])
                s += "&%.6f"%(results[fname][noise]["papp"])
            s+="\\\\\hline\n"

    print(s)

# python tablecompareall.py f7,f8  0,10-1 2x ../benchmarkdata/f7.txt,../benchmarkdata/f8.txt all,all latex
if __name__ == "__main__":
    import os, sys
    if len(sys.argv) != 7:
        print("Usage: {} function noise ts testfilelist bottom_or_all table_or_latex".format(sys.argv[0]))
        sys.exit(1)

    farr = sys.argv[1].split(',')
    if len(farr) == 0:
        print("please specify comma saperated functions")
        sys.exit(1)

    noisearr = sys.argv[2].split(',')
    if len(noisearr) == 0:
        print("please specify comma saperated noise levels")
        sys.exit(1)

    testfilearr = sys.argv[4].split(',')
    if len(testfilearr) == 0:
        print("please specify comma saperated testfile paths")
        sys.exit(1)

    bottomallarr = sys.argv[5].split(',')
    if len(bottomallarr) == 0:
        print("please specify comma saperated bottom or all options")
        sys.exit(1)


    tablecompareall(farr, noisearr, testfilearr, bottomallarr, sys.argv[3], sys.argv[6])

###########
