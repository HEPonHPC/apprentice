import numpy as np
from apprentice import RationalApproximationSIP, RationalApproximationONB, PolynomialApproximation
from apprentice import tools, readData
import matplotlib.ticker as mtick
import os,sys
def sinc(X,dim):
    ret = 10
    for d in range(dim):
        x = X[d]
        ret *= np.sin(x)/x
    return ret

def knowmissing(filename):
    arr = [
        "results/exp1/f18_noisepct10-2_sg_2x/outrard/f18_noisepct10-2_sg_2x_p5_q5_ts2x.json",
        "results/exp1/f18_noisepct10-6_sg_2x/outrard/f18_noisepct10-6_sg_2x_p5_q5_ts2x.json"
    ]
    for a in arr:
        if(filename == a):
            return 1
    return 0


def getdim(fname):
    dim = {"f1":2,"f2":2,"f3":2,"f4":2,"f5":2,"f7":2,"f8":2,"f9":2,"f10":4,"f12":2,"f13":2,
            "f14":2,"f15":2,"f16":2,"f17":3,"f18":4,"f19":4,"f20":4,"f21":2,"f22":2}
    return dim[fname]


def getXin(dim,num=10**6,bounday=10**-3):
    X = np.array([])
    Xperdim = ()
    np.random.seed(9999)

    max = 1-bounday
    min = -1+bounday
    for d in range(dim):
        Xperdim = Xperdim + (np.random.rand(num,)*(max-min)+min,)

    X = np.column_stack(Xperdim)
    return X

def getXcorner(dim,Xr,cornerL,cornerU,num):

    X = np.array([])
    for ddd in range(dim):
        X = np.append(X,10**-12)
    if dim==2:
        for ddd in range(dim):
            for c in range(2):
                if(c==0): ccc = cornerL
                elif(c==1): ccc = cornerU
                for i in range(num):
                    corner = ccc[i]
                    for a in Xr:
                        if(ddd==0):
                            xt = [corner,a]
                        elif(ddd==1):
                            xt = [a,corner]
                        X = np.vstack([X, xt])

    elif dim ==3:
        for ddd in range(dim):
            for c in range(2):
                if(c==0): ccc = cornerL
                elif(c==1): ccc = cornerU
                for i in range(num):
                    corner = ccc[i]
                    for a in Xr:
                        for b in Xr:
                            if(ddd==0):
                                xt = [corner,a,b]
                            elif(ddd==1):
                                xt = [a,corner,b]
                            elif(ddd==2):
                                xt = [a,b,corner]
                            X = np.vstack([X, xt])
    elif dim ==4:
        for ddd in range(dim):
            for c in range(2):
                if(c==0): ccc = cornerL
                elif(c==1): ccc = cornerU
                for i in range(num):
                    corner = ccc[i]
                    for a in Xr:
                        for b in Xr:
                            for c in Xr:
                                if(ddd==0):
                                    xt = [corner,a,b,c]
                                elif(ddd==1):
                                    xt = [a,corner,b,c]
                                elif(ddd==2):
                                    xt = [a,b,corner,c]
                                elif(ddd==3):
                                    xt = [a,b,c,corner]
                                X = np.vstack([X, xt])
    elif dim ==7:
        for ddd in range(dim):
            for c in range(2):
                if(c==0): ccc = cornerL
                elif(c==1): ccc = cornerU
                for i in range(num):
                    corner = ccc[i]
                    for a in Xr:
                        for b in Xr:
                            for c in Xr:
                                for d in Xr:
                                    for e in Xr:
                                        for f in Xr:
                                            if(ddd==0):
                                                xt = [corner,a,b,c,d,e,f]
                                            elif(ddd==1):
                                                xt = [a,corner,b,c,d,e,f]
                                            elif(ddd==2):
                                                xt = [a,b,corner,c,d,e,f]
                                            elif(ddd==3):
                                                xt = [a,b,c,corner,d,e,f]
                                            elif(ddd==4):
                                                xt = [a,b,c,d,corner,e,f]
                                            elif(ddd==5):
                                                xt = [a,b,c,d,e,corner,f]
                                            elif(ddd==6):
                                                xt = [a,b,c,d,e,f,corner]
                                            X = np.vstack([X, xt])
    else:
        print("%s is not capable of handling dim = %d"%(sys.argv[0],dim))
        sys.exit(1)
    return X
def getnoiseinfo(noise):
    noisearr = ["0","10-2","10-4","10-6"]
    noisestr = ["","_noisepct10-2","_noisepct10-4","_noisepct10-6"]
    noisepct = [0,10**-2,10**-4,10**-6]

    for i,n in enumerate(noisearr):
        if(n == noise):
            return noisestr[i],noisepct[i]

def getData(X_train, fn, noisepct):
    """
    TODO use eval or something to make this less noisy
    """
    from apprentice import testData
    if fn=="f1":
        Y_train = [testData.f1(x) for x in X_train]
    elif fn=="f2":
        Y_train = [testData.f2(x) for x in X_train]
    elif fn=="f3":
        Y_train = [testData.f3(x) for x in X_train]
    elif fn=="f4":
        Y_train = [testData.f4(x) for x in X_train]
    elif fn=="f5":
        Y_train = [testData.f5(x) for x in X_train]
    elif fn=="f6":
        Y_train = [testData.f6(x) for x in X_train]
    elif fn=="f7":
        Y_train = [testData.f7(x) for x in X_train]
    elif fn=="f8":
        Y_train = [testData.f8(x) for x in X_train]
    elif fn=="f9":
        Y_train = [testData.f9(x) for x in X_train]
    elif fn=="f10":
        Y_train = [testData.f10(x) for x in X_train]
    elif fn=="f12":
        Y_train = [testData.f12(x) for x in X_train]
    elif fn=="f13":
        Y_train = [testData.f13(x) for x in X_train]
    elif fn=="f14":
        Y_train = [testData.f14(x) for x in X_train]
    elif fn=="f15":
        Y_train = [testData.f15(x) for x in X_train]
    elif fn=="f16":
        Y_train = [testData.f16(x) for x in X_train]
    elif fn=="f17":
        Y_train = [testData.f17(x) for x in X_train]
    elif fn=="f18":
        Y_train = [testData.f18(x) for x in X_train]
    elif fn=="f19":
        Y_train = [testData.f19(x) for x in X_train]
    elif fn=="f20":
        Y_train = [testData.f20(x) for x in X_train]
    elif fn=="f21":
        Y_train = [testData.f21(x) for x in X_train]
    elif fn=="f22":
        Y_train = [testData.f22(x) for x in X_train]
    elif fn=="f23":
        Y_train = [testData.f23(x) for x in X_train]
    elif fn=="f24":
        Y_train = [testData.f24(x) for x in X_train]
    else:
        raise Exception("function {} not implemented, exiting".format(fn))

    # stdnormalnoise = np.zeros(shape = (len(Y_train)), dtype =np.float64)
    # for i in range(len(Y_train)):
    #     stdnormalnoise[i] = np.random.normal(0,1)

    # return np.atleast_2d(np.array(Y_train)*(1+ noisepct*stdnormalnoise))
    return Y_train

def findpredval(X_test,app):
    numer = np.array([app.numer(x) for x in X_test])
    denom = np.array([app.denom(x) for x in X_test])
    Y_pred = np.array([])
    for n,d in zip(numer,denom):
        if d==0:
            Y_pred = np.append(Y_pred,np.Infinity)
        else:
            Y_pred = np.append(Y_pred,n/d)
    return Y_pred

def getbox(f):
    minbox = []
    maxbox = []
    if(f=="f7"):
        minbox  = [0,0]
        maxbox = [1,1]
    elif(f=="f10" or f=="f19"):
        minbox  = [-1,-1,-1,-1]
        maxbox = [1,1,1,1]
    elif(f=="f17"):
        minbox  = [80,5,90]
        maxbox  = [100,10,93]
    # elif(f=="f17"):
    #     minbox  = [-1,-1,-1]
    #     maxbox  = [1,1,1]
    elif(f=="f18"):
        minbox  = [-0.95,-0.95,-0.95,-0.95]
        maxbox  = [0.95,0.95,0.95,0.95]
    elif(f=="f20"):
        minbox  = [10**-6,10**-6,10**-6,10**-6]
        maxbox  = [4*np.pi,4*np.pi,4*np.pi,4*np.pi]
    elif(f=="f21"):
        minbox  = [10**-6,10**-6]
        maxbox  = [4*np.pi,4*np.pi]
    else:
        minbox  = [-1,-1]
        maxbox = [1,1]
    return minbox,maxbox

def getresults(farr,noisearr, tarr, ts, allsamples):
    import apprentice
    m=5
    n=5
    thresholdvalarr = np.array([float(t) for t in tarr])
    thresholdvalarr = np.sort(thresholdvalarr)

    results = {}
    for fnum,fname in enumerate(farr):
        results[fname] = {}
        dim = getdim(fname)
        infile=[
                "results/plots/poledata_corner"+str(dim)+"D.csv",
                "results/plots/poledata_inside"+str(dim)+"D.csv"
                ]

        X_testfc = np.loadtxt(infile[0], delimiter=',')
        X_testin = np.loadtxt(infile[1], delimiter=',')

        # X_testall = np.vstack(X_test,X_testin)

        print(len(X_testfc),len(X_testin))
        minarr,maxarr = getbox(fname)

        s = apprentice.Scaler(np.array(X_testfc, dtype=np.float64), a=minarr, b=maxarr)
        X_testfc = s.scaledPoints

        s = apprentice.Scaler(np.array(X_testin, dtype=np.float64), a=minarr, b=maxarr)
        X_testin = s.scaledPoints

        Y_testfc = np.array(getData(X_testfc,fname,0))
        maxY_testfc = max(1,abs(np.max(Y_testfc)))

        Y_testin = np.array(getData(X_testin,fname,0))
        maxY_testin = max(1,abs(np.max(Y_testin)))
        print(fname,maxY_testfc,maxY_testin)

        results[fname]['npoints_face'] = len(Y_testfc)
        results[fname]['npoints_inside'] = len(Y_testin)

        for snum, sample in enumerate(allsamples):
            results[fname][sample] = {}
            for noise in noisearr:
                results[fname][sample][noise] = {}
                noisestr,noisepct = getnoiseinfo(noise)

                resdata = {}
                resdata['papp'] = {}
                resdata['rapp'] = {}
                resdata['rapprd'] = {}
                resdata['rappsip'] = {}

                resdata['papp']['l2all'] = []
                resdata['rapp']['l2all'] = []
                resdata['rapprd']['l2all'] = []
                resdata['rappsip']['l2all'] = []

                for tval in thresholdvalarr:
                    for method in ['papp','rapp','rapprd','rappsip']:
                        resdata[method][str(tval)] = {}
                        resdata[method][str(tval)]['no'] = []
                        resdata[method][str(tval)]['no_face'] = []
                        resdata[method][str(tval)]['no_inside'] = []
                        resdata[method][str(tval)]['l2count'] = []
                        resdata[method][str(tval)]['l2notcount'] = []

                for rnum,run in enumerate(["exp1","exp2","exp3","exp4","exp5"]):
                    fndesc = "%s%s_%s_%s"%(fname,noisestr,sample,ts)
                    folder = "results/%s/%s"%(run,fndesc)
                    # print(folder)
                    pq = "p%d_q%d"%(m,n)
                    # print(run, fname,noisestr,sample,m,n)

                    rappsipfile = "%s/outrasip/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)
                    rappfile = "%s/outra/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)
                    rapprdfile = "%s/outrard/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)
                    pappfile = "%s/outpa/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)

                    if not os.path.exists(rappsipfile):
                        print("rappsipfile %s not found"%(rappsipfile))
                        if(knowmissing(rappsipfile)==1):
                            if(sample == "sg"):
                                break
                            continue
                        exit(1)

                    if not os.path.exists(rappfile):
                        print("rappfile %s not found"%(rappfile))
                        if(knowmissing(rappfile)==1):
                            if(sample == "sg"):
                                break
                            continue
                        exit(1)

                    if not os.path.exists(rapprdfile):
                        print("rappfile %s not found"%(rapprdfile))
                        if(knowmissing(rapprdfile)==1):
                            if(sample == "sg"):
                                break
                            continue
                        exit(1)

                    if not os.path.exists(pappfile):
                        print("rappfile %s not found"%(pappfile))
                        if(knowmissing(pappfile)==1):
                            if(sample == "sg"):
                                break
                            continue
                        exit(1)
                    print(fndesc + " " + run + " Start")

                    papp = PolynomialApproximation(fname=pappfile)
                    try:
                        Y_pred_pappfc = np.array([papp(x) for x in X_testfc])
                        Y_pred_pappin = np.array([papp(x) for x in X_testin])
                    except:
                        Y_pred_pappfc = findpredval(X_testfc,papp)
                        Y_pred_pappin = findpredval(X_testin,papp)

                    rappsip = RationalApproximationSIP(rappsipfile)
                    try:
                        Y_pred_rappsipfc = rappsip.predictOverArray(X_testfc)
                        Y_pred_rappsipin = rappsip.predictOverArray(X_testin)
                    except:
                        Y_pred_rappsipfc = findpredval(X_testfc,rappsip)
                        Y_pred_rappsipin = findpredval(X_testin,rappsip)

                    rapp = RationalApproximationONB(fname=rappfile)
                    try:
                        Y_pred_rappfc = np.array([rapp(x) for x in X_testfc])
                        Y_pred_rappin = np.array([rapp(x) for x in X_testin])
                    except:
                        Y_pred_rappfc = findpredval(X_testfc,rapp)
                        Y_pred_rappin = findpredval(X_testin,rapp)

                    rapprd = RationalApproximationONB(fname=rapprdfile)
                    try:
                        Y_pred_rapprdfc = np.array([rapprd(x) for x in X_testfc])
                        Y_pred_rapprdin = np.array([rapprd(x) for x in X_testin])
                    except:
                        Y_pred_rapprdfc = findpredval(X_testfc,rapprd)
                        Y_pred_rapprdin = findpredval(X_testin,rapprd)


                    print(fndesc + " " + run + " Done ")
                    sys.stdout.flush()

                    Y_testall = np.concatenate((Y_testfc,Y_testin), axis=None)

                    Y_pred_pappall = np.concatenate((Y_pred_pappfc,Y_pred_pappin), axis=None)
                    Y_pred_rappsipall = np.concatenate((Y_pred_rappsipfc,Y_pred_rappsipin), axis=None)
                    Y_pred_rappall = np.concatenate((Y_pred_rappfc,Y_pred_rappin), axis=None)
                    Y_pred_rapprdall = np.concatenate((Y_pred_rapprdfc,Y_pred_rapprdin), axis=None)

                    l2allrapp = np.sum((Y_pred_rappall-Y_testall)**2)
                    l2allrapprd = np.sum((Y_pred_rapprdall-Y_testall)**2)
                    l2allrappsip = np.sum((Y_pred_rappsipall-Y_testall)**2)
                    l2allpapp = np.sum((Y_pred_pappall-Y_testall)**2)
                    # print(l2allrapp,l2allrapprd,l2allrappsip)

                    resdata['rapp']['l2all'].append(np.sqrt(l2allrapp))
                    resdata['rapprd']['l2all'].append(np.sqrt(l2allrapprd))
                    resdata['rappsip']['l2all'].append(np.sqrt(l2allrappsip))
                    resdata['papp']['l2all'].append(np.sqrt(l2allpapp))

                    for tnum,tval in enumerate(thresholdvalarr):
                        for method in ['papp','rapp','rapprd','rappsip']:
                            resdata[method][str(tval)]['no_face'].append(0)
                            resdata[method][str(tval)]['no_inside'].append(0)
                            resdata[method][str(tval)]['no'].append(0)
                            resdata[method][str(tval)]['l2count'].append(0.)
                            resdata[method][str(tval)]['l2notcount'].append(0.)


                    for num,yt in enumerate(Y_testfc):
                        for method, pred in zip(['papp','rapp','rapprd','rappsip'],
                                                [Y_pred_pappfc,Y_pred_rappfc,Y_pred_rapprdfc,Y_pred_rappsipfc]):
                            yp = pred[num]
                            for tnum,tval in enumerate(thresholdvalarr):
                                if(abs(yp)/maxY_testfc > tval):
                                    resdata[method][str(tval)]['no'][rnum] +=1
                                    resdata[method][str(tval)]['no_face'][rnum] +=1
                                    resdata[method][str(tval)]['l2count'][rnum] += (yp-yt)**2

                    for num,yt in enumerate(Y_testin):
                        for method, pred in zip(['papp','rapp','rapprd','rappsip'],
                                                [Y_pred_pappin,Y_pred_rappin,Y_pred_rapprdin,Y_pred_rappsipin]):
                            yp = pred[num]
                            for tnum,tval in enumerate(thresholdvalarr):
                                if(abs(yp)/maxY_testin > tval):
                                    resdata[method][str(tval)]['no'][rnum] +=1
                                    resdata[method][str(tval)]['no_inside'][rnum] +=1
                                    resdata[method][str(tval)]['l2count'][rnum] += (yp-yt)**2

                    for tnum,tval in enumerate(thresholdvalarr):
                        for method, l2all in zip(['papp','rapp','rapprd','rappsip'],
                                [l2allpapp,l2allrapp,l2allrapprd,l2allrappsip]):
                            l2count = resdata[method][str(tval)]['l2count'][rnum]
                            resdata[method][str(tval)]['l2notcount'][rnum] = np.sqrt(l2all - l2count)
                            resdata[method][str(tval)]['l2count'][rnum] = np.sqrt(l2count)

                    if(sample == "sg"):
                        break
                missingmean = -1
                for method in ['papp','rapp','rapprd','rappsip']:
                    l2allarr = resdata[method]['l2all']
                    results[fname][sample][noise][method] = {}
                    if(len(l2allarr)!=0):
                        results[fname][sample][noise][method]['l2all'] = float(getstats(l2allarr,'amean'))
                        results[fname][sample][noise][method]['l2allgm'] = float(getstats(l2allarr,'gmean'))
                        results[fname][sample][noise][method]['l2allmed'] = float(getstats(l2allarr,'median'))
                        results[fname][sample][noise][method]['l2allra'] = float(getstats(l2allarr,'range'))
                        results[fname][sample][noise][method]['l2allsd'] = float(np.std(l2allarr))
                    else:
                        results[fname][sample][noise][method]['l2all'] = missingmean
                        results[fname][sample][noise][method]['l2allgm'] = missingmean
                        results[fname][sample][noise][method]['l2allmed'] = missingmean
                        results[fname][sample][noise][method]['l2allra'] = missingmean
                        results[fname][sample][noise][method]['l2allsd'] = 0

                for tval in thresholdvalarr:
                    for method in ['papp','rapp','rapprd','rappsip']:
                        results[fname][sample][noise][method][str(tval)] = {}
                        for key in ['l2notcount','l2count','no','no_face','no_inside']:

                            arr = resdata[method][str(tval)][key]
                            if(len(arr)!=0):
                                results[fname][sample][noise][method][str(tval)][key] = float(getstats(arr,'amean'))
                                results[fname][sample][noise][method][str(tval)][key+'gm'] = float(getstats(arr,'gmean'))
                                results[fname][sample][noise][method][str(tval)][key+'med'] = float(getstats(arr,'median'))
                                results[fname][sample][noise][method][str(tval)][key+'ra'] = float(getstats(arr,'range'))
                                results[fname][sample][noise][method][str(tval)][key+'sd'] = float(np.std(arr))
                            else:
                                results[fname][sample][noise][method][str(tval)][key] = missingmean
                                results[fname][sample][noise][method][str(tval)][key+'gm'] = missingmean
                                results[fname][sample][noise][method][str(tval)][key+'med'] = missingmean
                                results[fname][sample][noise][method][str(tval)][key+'ra'] = missingmean
                                results[fname][sample][noise][method][str(tval)][key+'sd'] = 0


        print("done with fn: %s"%(fname))

    return results

def generatedata():
     m = 5
     n = 5
     bounday = 10**-3
     numarr = [0,0,1000,100,100,100]
     import math
     if not os.path.exists("results/plots"):
         os.makedirs("results/plots", exist_ok = True)

     for dim in range(2,5):
         num = numarr[dim]
         # num=math.ceil(10**(6/dim))
         cnum = int(0.1*num)
         innum = 10**6
         # num=5
         # cnum = 5
         # innum = 5**dim
         Xr = np.linspace(-1, 1, num=num)
         cornerL = np.linspace(-1, -1+bounday, num=cnum)
         cornerU = np.linspace(1-bounday, 1, num=cnum)
         X_test = getXcorner(dim,Xr,cornerL,cornerU,cnum)
         outfile = "results/plots/poledata_corner"+str(dim)+"D.csv"
         np.savetxt(outfile, X_test, delimiter=",")


         X_test = getXin(dim,num=innum,bounday=bounday)

         outfile = "results/plots/poledata_inside"+str(dim)+"D.csv"
         np.savetxt(outfile, X_test, delimiter=",")


def checkfloat(fmtstr,fl):
    if fl != 0:
        return fmtstr%(fl)
    else:
        return " & -"

def getstats(data,metr):
    def geomean(iterable):
        a = np.array(iterable)
        return a.prod()**(1.0/len(a))

    if metr == 'amean': return np.average(data)
    elif metr == 'gmean': return geomean(data)
    elif metr == 'median': return np.median(data)
    elif metr == 'range': return np.max(data) - np.min(data)
    else:
        print("metric not known")
        exit(1)

def tablepoles(farr,noisearr, tarr, ts, table_or_latex,usejson=0):
    print (farr)
    print (noisearr)
    print (thresholdarr)
    methodarr = ['rapp','rapprd', 'rappsip','papp']
    if not os.path.exists("results/plots"):
        os.makedirs("results/plots", exist_ok = True)

    # allsamples = ['sg']
    # allsamples = ['lhs']
    # allsamples = ['mc','lhs','so','sg']
    allsamples = ['sg','lhs','splitlhs']
    allsampleslabels = ['SG','LHS','d-LHD']
    # allsamples = ['splitlhs']
    # allsampleslabels = ['d-LHD']
    s= ""


    outfilejson = "results/plots/Jpoleanderrorinfo"+farr[0]+".json"
    import json
    if(usejson ==0):
        results = getresults(farr,noisearr, tarr, ts,allsamples)
        with open(outfilejson, "w") as f:
            json.dump(results, f,indent=4, sort_keys=True)
    elif(usejson ==1):
        thresholdvalarr = np.array([float(t) for t in tarr])
        thresholdvalarr = np.sort(thresholdvalarr)
        tval = thresholdvalarr[0]
        meanarr = []
        meanp1arr = []
        fmtstr = "& %.1E "
        for fnum,fname in enumerate(farr):
            outfilejson = "results/plots/Jpoleanderrorinfo"+fname+".json"
            if outfilejson:
                with open(outfilejson, 'r') as fn:
                    results = json.load(fn)
            s+= "\\multirow{8}{*}{\\ref{fn:%s}}\n"%(fname)
            for sample in allsamples:
                s+= "& \\multirow{2}{*}{$|W_{r,10^2}|$}\n"
                s+= "& M "
                for noise in noisearr:
                    for method in methodarr:
                        no = results[fname][sample][noise][method][str(tval)]['no']
                        s+=checkfloat(fmtstr,no)
                s+="\\\\ \n"
                s+= "& & SD "
                for noise in noisearr:
                    for method in methodarr:
                        nosd = results[fname][sample][noise][method][str(tval)]['nosd']
                        s+=checkfloat(fmtstr,nosd)
                s+="\\\\\n"
                s+= "& \\multirow{2}{*}{$E_{r,10^2}$}\n"
                s+= "& M "
                for noise in noisearr:
                    for method in methodarr:
                        l2count = results[fname][sample][noise][method][str(tval)]['l2count']
                        s+=checkfloat(fmtstr,l2count)
                s+="\\\\\n"
                s+= "& & SD "
                for noise in noisearr:
                    for method in methodarr:
                        l2countsd = results[fname][sample][noise][method][str(tval)]['l2countsd']
                        s+=checkfloat(fmtstr,l2countsd)
                s+="\\\\\n"
                s+= "& \\multirow{2}{*}{$E'_{r,10^2}$}\n"
                s+= "& M "
                for noise in noisearr:
                    for method in methodarr:
                        l2notcount = results[fname][sample][noise][method][str(tval)]['l2notcount']
                        s+=checkfloat(fmtstr,l2notcount)
                s+="\\\\\n"
                s+= "& & SD "
                for noise in noisearr:
                    for method in methodarr:
                        l2notcountsd = results[fname][sample][noise][method][str(tval)]['l2notcountsd']
                        s+=checkfloat(fmtstr,l2notcountsd)
                s+="\\\\\n"
                s+= "& \\multirow{2}{*}{$\\Delta_r$}\n"
                s+= "& M "
                for noise in noisearr:
                    for method in methodarr:
                        l2all = results[fname][sample][noise][method]['l2all']
                        s+=checkfloat(fmtstr,l2all)
                s+="\\\\\n"
                s+= "& & SD "
                for noise in noisearr:
                    for method in methodarr:
                        l2allsd = results[fname][sample][noise][method]['l2allsd']
                        s+=checkfloat(fmtstr,l2allsd)
                s+="\\\\\\cline{3-15}\\hline\n"
        print(s)
    elif(usejson == 2):
        dumpr = {}
        thresholdvalarr = np.array([float(t) for t in tarr])
        thresholdvalarr = np.sort(thresholdvalarr)
        tval = thresholdvalarr[0]
        for fnum,fname in enumerate(farr):
            outfilejson = "results/plots/Jpoleanderrorinfo"+fname+".json"
            if outfilejson:
                with open(outfilejson, 'r') as fn:
                    results = json.load(fn)
            dumpr[fname] = {}
            for sample in allsamples:
                for noise in noisearr:
                    dumpr[fname][noise] = {}
                    for method in methodarr:
                        dumpr[fname][noise][method] = {
                            "E_rt_mean":results[fname][sample][noise][method][str(tval)]['l2count'],
                            "E_rt_sd":results[fname][sample][noise][method][str(tval)]['l2countsd'],
                            "Eprime_rt_mean":results[fname][sample][noise][method][str(tval)]['l2notcount'],
                            "Eprime_rt_sd":results[fname][sample][noise][method][str(tval)]['l2notcountsd']
                        }
        import json
        with open("results/plots/Jerrordata.json", "w") as f:
            json.dump(dumpr, f,indent=4, sort_keys=True)

    elif(usejson == 3):
        thresholdvalarr = np.array([float(t) for t in tarr])
        thresholdvalarr = np.sort(thresholdvalarr)
        tval = thresholdvalarr[0]
        metricarr = ['amean','gmean','median','range']
        metricarrlabel = ['Arithmetic Mean','Geometric Mean','Median','Range']
        sample = 'splitlhs'
        s = ""
        for mnum, metr in enumerate(metricarr):
            s+="%s"%(metricarrlabel[mnum])
            for noise in noisearr:
                for menum,method in enumerate(methodarr):
                    data = []
                    for fnum,fname in enumerate(farr):
                        outfilejson = "results/plots/Jpoleanderrorinfo"+fname+".json"
                        if outfilejson:
                            with open(outfilejson, 'r') as fn:
                                results = json.load(fn)
                        data.append(results[fname][sample][noise][method]['l2all'])


                    from sklearn import preprocessing
                    minmaxscaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
                    data = minmaxscaler.fit_transform(np.reshape(data,(-1,1)))
                    stat = getstats(data,metr)
                    s+="&%.2E"%(stat)
            s+="\n\\\\\\hline\n"
        print(s)
    elif(usejson == 4):
        # python tablepolesnerror.py f1,f2,f3,f4,f5,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22 0 100,1000 2x table 4
        # python tablepolesnerror.py f1,f2,f3,f4,f5,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22 10-6 100,1000 2x table 4
        # python tablepolesnerror.py f1,f2,f3,f4,f5,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22 10-2 100,1000 2x table 4
        thresholdvalarr = np.array([float(t) for t in tarr])
        thresholdvalarr = np.sort(thresholdvalarr)
        tval = thresholdvalarr[0]
        noise = noisearr[0]
        wetkeyarr = ['no_face','no_inside','l2count','l2notcount','l2all']
        WETarr = [
            "|W^{(fc)}_{r,10^2}|",
            "|W^{(in)}_{r,10^2}|",
            "E_{r,10^2}",
            "E'_{r,10^2}",
            "\\Delta_r"
        ]

        import math
        for fnum, fname in enumerate(farr):
            outfilejson = "results/plots/Jpoleanderrorinfo"+fname+".json"
            if outfilejson:
                with open(outfilejson, 'r') as fn:
                    results = json.load(fn)
            number = len(allsamples) * 5
            s += "\\multirow{%d}{*}{\\ref{fn:%s}}"%(number,fname)
            for snum, sample in enumerate(allsamples):
                s+="&\\multirow{5}{*}{%s}"%(allsampleslabels[snum])
                for wnum, wet in enumerate(WETarr):
                    statsarr = [
                                    wetkeyarr[wnum],
                                    wetkeyarr[wnum]+'med'
                    ]
                    if wnum>0:
                        s+="&"
                    s+="&$%s$"%(wet)
                    for mnum, method in enumerate(['rapp','rapprd','rappsip','papp']):
                        for stnum, stat in enumerate(statsarr):
                            if wnum < 4:
                                val = results[fname][sample][noise][method][str(tval)][stat]
                            else:
                                val = results[fname][sample][noise][method][stat]
                            if sample == 'sg' and stnum > 0:
                                s+="&-"
                            elif method == 'papp' and wnum < 4:
                                s+="&-"
                            elif val == int(val):
                                s+="&%d"%(int(val))
                            elif val < 10**-2 or val >10**2:
                                s+="&%.2E"%(val)
                            else:
                                s+="&%.2f"%(val)
                    if wnum < len(WETarr)-1:
                        s+="\n\\\\*  \\cline{3-11}\n"
                if snum < len(allsamples)-1:
                    s+="\n\\\\*  \\cline{2-11}\n"
            s+="\n\\\\ \\hline\n"
        print(s)













if __name__ == "__main__":
    import os, sys

    if(sys.argv[1] == 'gen'):
        generatedata()
        exit(0)


 # python tablepoles.py f1,f2,f3,f4,f5,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f19,f20,f22  0,10-1 10,100,1000 2x  table
 # for fno in {1..5} {7..10} {12..20} 22; do  name="f"$fno; nohup python tablepoles.py $name 0,10-1 10,100,1000 2x  table> ../../debug/"tablepoles_"$name".log" 2>&1 & done
 # for fno in 3 5 9 13 14 18 19; do  name="f"$fno; nohup python tablepoles.py $name 0,10-1 10,100,1000 2x  latex> ../../debug/"tablepoles_latex_"$name".log" 2>&1 & done

# for fno in {1..5} {7..10} {12..22}; do  name="f"$fno; nohup python tablepoles.py $name 0,10-2,10-6 100,1000 2x table 0 > ../../log/"tablepoles_"$name".log" 2>&1 &  done
# python tablepoles.py f1,f2,f3,f4,f5,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22 0,10-6,10-2 100,1000 2x table 1

# for fno in 4 7 17 18 19; do  name="f"$fno; nohup python tablepolesnerror.py $name 0,10-2,10-6 100,1000 2x table 0 > ../../log/"tablepoles_"$name".log" 2>&1 &  done
# python tablepolesnerror.py f4,f7,f17,f18,f19 0,10-2,10-6 100 2x table 1
# python tablepolesnerror.py f4,f8,f17,f18,f19 0,10-2,10-6 100 2x table 2
# for fno in {1..3} 5 {9..10} {12..16} 20 21 22; do  name="f"$fno; nohup python tablepolesnerror.py $name 0,10-2,10-6 100,1000 2x table 0 > ../../log/"tablepoles_"$name".log" 2>&1 &  done
    if len(sys.argv) != 7:
        print("Usage: {} function noise thresholds ts table_or_latex_or_latexall usejson(0 or 1)".format(sys.argv[0]))
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

    # tablepoles(farr,noisearr, thresholdarr, testfilearr, bottomallarr,sys.argv[4],sys.argv[7])
    tablepoles(farr,noisearr, thresholdarr, sys.argv[4],sys.argv[5],int(sys.argv[6]))


###########
