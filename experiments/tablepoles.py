import numpy as np
from apprentice import RationalApproximationSIP, RationalApproximationONB, PolynomialApproximation
from apprentice import tools, readData
import os,sys
def sinc(X,dim):
    ret = 10
    for d in range(dim):
        x = X[d]
        ret *= np.sin(x)/x
    return ret

def knowmissing(filename):
    arr = [
        "results/exp1/f20_sg_2x/outrasip/f20_sg_2x_p5_q5_ts2x.json",
        "results/exp1/f18_noisepct10-2_sg_2x/outrard/f18_noisepct10-2_sg_2x_p5_q5_ts2x.json",
        "results/exp1/f18_noisepct10-2_sg_2x/outrard1/f18_noisepct10-2_sg_2x_p5_q5_ts2x.json",
        "results/exp1/f20_noisepct10-2_sg_2x/outrasip/f20_noisepct10-2_sg_2x_p5_q5_ts2x.json",
        "results/exp1/f18_noisepct10-6_sg_2x/outrard/f18_noisepct10-6_sg_2x_p5_q5_ts2x.json",
        "results/exp1/f18_noisepct10-6_sg_2x/outrard1/f18_noisepct10-6_sg_2x_p5_q5_ts2x.json",
        "results/exp1/f20_noisepct10-6_sg_2x/outrasip/f20_noisepct10-6_sg_2x_p5_q5_ts2x.json"
    ]
    for a in arr:
        if(filename == a):
            return 1
    return 0

# def getpqstr(fname):
#     pq = ""
#     if (fname=='f1'): pq = "p2_q4"
#     if (fname=='f2'): pq = "p5_q2"
#     if (fname=='f3'): pq = "p4_q3"
#     if (fname=='f4'): pq = 'p3_q2'
#     if (fname=='f5') : pq = "p2_q3"
#     if (fname=='f7') : pq = 'p2_q7'
#     if fname=='f8' : pq = "p3_q3"
#     if fname=='f9' : pq = "p3_q7"
#     if fname=='f10' : pq = 'p2_q4'
#     if fname=='f12' : pq = 'p3_q3'
#     if fname=='f13' : pq = "p2_q7"
#     if fname=='f14' : pq = "p3_q6"
#     if fname=='f15' : pq = "p2_q5"
#     if fname=='f16' : pq = "p3_q7"
#     if fname=='f17' : pq = 'p4_q6'
#     if fname=='f18' : pq = "p2_q3"
#     if fname=='f19' : pq = "p3_q3"
#     if fname=='f20' : pq = "p2_q3"
#     if fname=='f21' : pq = "p5_q2"
#     if fname=='f22' : pq = "p2_q4"
#     return pq

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


def getresults(farr,noisearr, tarr, ts, allsamples, usecornerpoints):
    m=5
    n=5
    thresholdvalarr = np.array([float(t) for t in tarr])
    thresholdvalarr = np.sort(thresholdvalarr)

    results = {}
    for fnum,fname in enumerate(farr):
        results[fname] = {}
        dim = getdim(fname)
        if(usecornerpoints == 1):
            infile = "results/plots/poledata_corner"+str(dim)+"D.csv"
        else:
            infile = "results/plots/poledata_inside"+str(dim)+"D.csv"


        X_test = np.loadtxt(infile, delimiter=',')

        print(len(X_test))

        Y_test = getData(X_test,fname,0)
        maxY_test = max(1,abs(np.max(Y_test)))
        print(fname,maxY_test)

        results[fname]['npoints'] = len(Y_test)

        for snum, sample in enumerate(allsamples):
            results[fname][sample] = {}
            for noise in noisearr:
                results[fname][sample][noise] = {}
                noisestr,noisepct = getnoiseinfo(noise)

                resdata = {}
                resdata['rapp'] = {}
                resdata['rapprd'] = {}
                resdata['rappsip'] = {}


                resdata['rapp']['l2all'] = []
                resdata['rapprd']['l2all'] = []
                resdata['rappsip']['l2all'] = []

                for tval in thresholdvalarr:
                    for method in ['rapp','rapprd','rappsip']:
                        resdata[method][str(tval)] = {}
                        resdata[method][str(tval)]['no'] = []
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
                    print(fndesc)
                    rappsip = RationalApproximationSIP(rappsipfile)
                    Y_pred_rappsip = findpredval(X_test,rappsip)
                    rapp = RationalApproximationONB(fname=rappfile)
                    Y_pred_rapp = findpredval(X_test,rapp)
                    rapprd = RationalApproximationONB(fname=rapprdfile)
                    Y_pred_rapprd = findpredval(X_test,rapprd)

                    l2allrapp = np.sum((Y_pred_rapp-Y_test)**2)
                    l2allrapprd = np.sum((Y_pred_rapprd-Y_test)**2)
                    l2allrappsip = np.sum((Y_pred_rappsip-Y_test)**2)

                    resdata['rapp']['l2all'].append(np.sqrt(l2allrapp))
                    resdata['rapprd']['l2all'].append(np.sqrt(l2allrapprd))
                    resdata['rappsip']['l2all'].append(np.sqrt(l2allrappsip))

                    for tnum,tval in enumerate(thresholdvalarr):
                        for method in ['rapp','rapprd','rappsip']:
                            resdata[method][str(tval)]['no'].append(0)
                            resdata[ method][str(tval)]['l2count'].append(0.)
                            resdata[method][str(tval)]['l2notcount'].append(0.)


                    for num,yt in enumerate(Y_test):
                        for method, pred in zip(['rapp','rapprd','rappsip'],[Y_pred_rapp,Y_pred_rapprd,Y_pred_rappsip]):
                            yp = pred[num]
                            for tnum,tval in enumerate(thresholdvalarr):
                                if(abs(yp)/maxY_test > tval):
                                    resdata[method][str(tval)]['no'][rnum] +=1
                                    resdata[method][str(tval)]['l2count'][rnum] += (yp-yt)**2

                    for tnum,tval in enumerate(thresholdvalarr):
                        for method, l2all in zip(['rapp','rapprd','rappsip'],[l2allrapp,l2allrapprd,l2allrappsip]):
                            l2count = resdata[method][str(tval)]['l2count'][rnum]
                            resdata[method][str(tval)]['l2notcount'][rnum] = np.sqrt(l2all - l2count)
                            resdata[method][str(tval)]['l2count'][rnum] = np.sqrt(l2count)

                    if(sample == "sg"):
                        break
                missingmean = -1
                for method in ['rapp','rapprd','rappsip']:
                    l2allarr = resdata[method]['l2all']
                    results[fname][sample][noise][method] = {}
                    if(len(l2allarr)!=0):
                        results[fname][sample][noise][method]['l2all'] = np.average(l2allarr)
                        results[fname][sample][noise][method]['l2allsd'] = np.std(l2allarr)
                    else:
                        results[fname][sample][noise][method]['l2all'] = missingmean
                        results[fname][sample][noise][method]['l2allsd'] = 0

                for tval in thresholdvalarr:
                    for method in ['rapp','rapprd','rappsip']:
                        results[fname][sample][noise][method][str(tval)] = {}
                        for key in ['l2notcount','l2count','no']:

                            arr = resdata[method][str(tval)][key]
                            if(len(arr)!=0):
                                results[fname][sample][noise][method][str(tval)][key] = np.average(arr)
                                results[fname][sample][noise][method][str(tval)][key+'sd'] = np.std(arr)
                            else:
                                results[fname][sample][noise][method][str(tval)][key] = missingmean
                                results[fname][sample][noise][method][str(tval)][key+'sd'] = 0

        print("done with fn: %s for usecornerpoints = %d"%(fname,usecornerpoints))

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



def tablepoles(farr,noisearr, tarr, ts, table_or_latex,usejson=0):
    print (farr)
    print (noisearr)
    print (thresholdarr)
    if not os.path.exists("results/plots"):
        os.makedirs("results/plots", exist_ok = True)

    # allsamples = ['sg']
    # allsamples = ['lhs']
    allsamples = ['mc','lhs','so','sg']

    outfilejson = "results/plots/Jpoleinfo"+farr[0]+".json"
    import json
    if(usejson ==0):
        resultsnotcorner = getresults(farr,noisearr, tarr, ts,allsamples,usecornerpoints=0)
        resultscorner = getresults(farr,noisearr, tarr, ts,allsamples,usecornerpoints=1)
        results = {
            'resultscorner' : resultscorner,
            'resultsnotcorner' : resultsnotcorner,
        }

        with open(outfilejson, "w") as f:
            json.dump(results, f,indent=4, sort_keys=True)
    elif(usejson ==1):
        thresholdvalarr = np.array([float(t) for t in tarr])
        thresholdvalarr = np.sort(thresholdvalarr)
        strdata = {}
        for snum, sample in enumerate(allsamples):
            strdata[sample] = {}
            for noise in noisearr:
                strdata[sample][noise] = ""
        for fnum,fname in enumerate(farr):
            outfilejson = "results/plots/Jpoleinfo"+fname+".json"
            if outfilejson:
                with open(outfilejson, 'r') as fn:
                    results = json.load(fn)
            Nin = results['resultsnotcorner'][fname]['npoints']
            Ncorner = results['resultscorner'][fname]['npoints']-1
            for snum, sample in enumerate(allsamples):
                for noise in noisearr:
                    s= "\\multirow{2}{*}{\\ref{fn:%s}}&$|W_{r,t}^{cner}|/N$&%.1E"%(fname,Ncorner)

                    noisestr,noisepct = getnoiseinfo(noise)
                    for method in ['rapp','rapprd', 'rappsip']:
                        for tval in thresholdvalarr:
                            obj = results['resultscorner'][fname][sample][noise][method][str(tval)]
                            if(obj['no'] == 0):
                                s+="&0"
                            else: s+="&%.1E"%(obj['no']/Ncorner)
                            if(obj['nosd'] == 0):
                                s+="&0"
                            else: s+="&%.1E"%(obj['nosd']/Ncorner)


                    s+="\\\\\\cline{2-15}\n"
                    s+="&$|W_{r,t}^{in}|/N$&%.1E"%(Nin)

                    for method in ['rapp','rapprd', 'rappsip']:
                        for tval in thresholdvalarr:
                            obj = results['resultsnotcorner'][fname][sample][noise][method][str(tval)]
                            if(obj['no'] == 0):
                                s+="&0"
                            else: s+="&%.1E"%(obj['no']/Nin)
                            if(obj['nosd'] == 0):
                                s+="&0"
                            else: s+="&%.1E"%(obj['nosd']/Nin)
                    s+="\\\\\\cline{2-15}\n"
                    s+="\\hline\n"
                    strdata[sample][noise] += s
        for snum, sample in enumerate(allsamples):
            for noise in noisearr:
                print("sample = %s noise = %s"%(sample,noise))
                print("\n%s\n\n\n"%(strdata[sample][noise]))















    exit(0)

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
                s += "\t%s"%(int(tval))
            s+="\t"
            for tval in thresholdvalarr:
                s += "\t%s"%(int(tval))
            s+="\t"
        s += "\n"
        for fname in farr:
            s += "%s\n"%(fname)
            for pq in results[fname][noisearr[0]].keys():
                s += "%s"%(pq)
                for noise in noisearr:
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        sss = "-"
                        if(results[fname][noise][pq][tvalstr]["rapp"] != "0"):
                            sss= results[fname][noise][pq][tvalstr]["rapp"]
                        s += "\t%s"%(sss)
                    s+="\t"
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        sss = "-"
                        if(results[fname][noise][pq][tvalstr]["rappsip"] != "0"):
                            sss= results[fname][noise][pq][tvalstr]["rappsip"]
                        s += "\t%s"%(sss)
                    s+="\t"
                s+="\n"

    elif(table_or_latex =="latex"):
        for fname in farr:
            for pq in results[fname][noisearr[0]].keys():
                sspecific = ""
                s+= '%'+" %s %s\n"%(fname,pq)
                s+= "\\multirow{4}{*}{\\ref{fn:%s}}&$|W_{r,t}|$"%(fname)
                sspecific+= '%'+" %s %s\n"%(fname,pq)
                sspecific += "\\multirow{4}{*}{\\ref{fn:%s}}&$|W_{r,t}|$"%(fname)
                for noise in noisearr:
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        s+="&%s"%(results[fname][noise][pq][tvalstr]["rapp"])
                        sspecific+="&%s"%(results[fname][noise][pq][tvalstr]["rapp"])
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        s+="&%s"%(results[fname][noise][pq][tvalstr]["rappsip"])
                        sspecific+="&%s"%(results[fname][noise][pq][tvalstr]["rappsip"])
                s+="\\\\\\cline{2-10}\n"
                s+="&$E_{r,t}$"
                sspecific+="\\\\\\cline{2-10}\n"
                sspecific+="&$E_{r,t}$"
                for noise in noisearr:
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        if(results[fname][noise][pq][tvalstr]["l2countrapp"] ==0):
                            s+="&0"
                            sspecific+="&0"
                        else:
                            s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2countrapp"])
                            sspecific+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2countrapp"])
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        if(results[fname][noise][pq][tvalstr]["l2countrappsip"] ==0):
                            s+="&0"
                            sspecific+="&0"
                        else:
                            s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2countrappsip"])
                            sspecific+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2countrappsip"])
                s+="\\\\\\cline{2-10}\n"
                s+="&$E'_{r,t}$"
                sspecific+="\\\\\\cline{2-10}\n"
                sspecific+="&$E'_{r,t}$"
                for noise in noisearr:
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        if(results[fname][noise][pq][tvalstr]["l2notcountrapp"] == 0):
                            s+="&0"
                            sspecific+="&0"
                        else:
                            s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2notcountrapp"])
                            sspecific+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2notcountrapp"])
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        if(results[fname][noise][pq][tvalstr]["l2notcountrappsip"] == 0):
                            s+="&0"
                            sspecific+="&0"
                        else:
                            s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2notcountrappsip"])
                            sspecific+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2notcountrappsip"])
                s+="\\\\\\cline{2-10}\n"
                s+="&$\\Delta_r$"
                sspecific+="\\\\\\cline{2-10}\n"
                sspecific+="&$\\Delta_r$"
                for noise in noisearr:
                    tvalstr = str(int(thresholdvalarr[0]))
                    if(results[fname][noise][pq][tvalstr]["l2allrapp"]==0):
                        s+="&\\multicolumn{2}{|c|}{0}"
                        sspecific+="&\\multicolumn{2}{|c|}{0}"
                    else:
                        s+="&\\multicolumn{2}{|c|}{%.1E}"%(results[fname][noise][pq][tvalstr]["l2allrapp"])
                        sspecific+="&\\multicolumn{2}{|c|}{%.1E}"%(results[fname][noise][pq][tvalstr]["l2allrapp"])
                    if(results[fname][noise][pq][tvalstr]["l2allrappsip"]==0):
                        s+="&\\multicolumn{2}{|c|}{0}"
                        sspecific+="&\\multicolumn{2}{|c|}{0}"
                    else:
                        s+="&\\multicolumn{2}{|c|}{%.1E}"%(results[fname][noise][pq][tvalstr]["l2allrappsip"])
                        sspecific+="&\\multicolumn{2}{|c|}{%.1E}"%(results[fname][noise][pq][tvalstr]["l2allrappsip"])
                s+="\\\\\\cline{2-10}\n"
                s+="\\hline\n\n"
                sspecific+="\\\\\\cline{2-10}\n"
                sspecific+="\\hline\n\n"
                # if (fname=='f3' and pq == "p4_q3")\
                #     or (fname=='f5' and pq == "p2_q3")\
                #     or (fname=='f8' and pq == "p3_q3")\
                #     or (fname=='f9' and pq == "p3_q7")\
                #     or (fname=='f13' and pq == "p2_q7")\
                #     or (fname=='f14' and pq == "p3_q6")\
                #     or (fname=='f18' and pq == "p2_q3")\
                #     or (fname=='f19' and pq == "p3_q3"):
                #     print(sspecific)

    elif(table_or_latex =="latexall"):
        for fname in farr:
            for pq in results[fname][noisearr[0]].keys():
                sspecific = ""
                s+= '%'+" %s %s\n"%(fname,pq)
                s+= "\\multirow{3}{*}{\\ref{fn:%s}}&$r$~(Algorithm~\\ref{A:Polyak})"%(fname)
                sspecific+= '%'+" %s %s\n"%(fname,pq)
                sspecific+= "\\multirow{3}{*}{\\ref{fn:%s}}&$r$~(Algorithm~\\ref{A:Polyak})"%(fname)
                for noise in noisearr:
                    tvalstr = str(int(thresholdvalarr[0]))
                    if(results[fname][noise][pq][tvalstr]["l2allrappsip"]==0):
                        s+="&0"
                        sspecific+="&0"
                    else:
                        s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2allrappsip"])
                        sspecific+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2allrappsip"])
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        if(results[fname][noise][pq][tvalstr]["l2countrappsip"]==0):
                            s+="&0"
                            sspecific+="&0"
                        else:
                            s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2countrappsip"])
                            sspecific+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2countrappsip"])

                        if(results[fname][noise][pq][tvalstr]["l2notcountrappsip"] ==0):
                            s+="&0"
                            sspecific+="&0"
                        else:
                            s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2notcountrappsip"])
                            sspecific+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2notcountrappsip"])

                s+="\\\\\\cline{2-12}\n"
                s+="&$r$ (Algorithm \\ref{ALG:MVVandQR})"
                sspecific+="\\\\\\cline{2-12}\n"
                sspecific+="&$r$ (Algorithm \\ref{ALG:MVVandQR})"
                for noise in noisearr:
                    tvalstr = str(int(thresholdvalarr[0]))
                    if(results[fname][noise][pq][tvalstr]["l2allrapp"] == 0):
                        s+="&0"
                        sspecific+="&0"
                    else:
                        s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2allrapp"])
                        sspecific+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2allrapp"])

                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        if(results[fname][noise][pq][tvalstr]["l2countrapp"] ==0):
                            s+="&0"
                            sspecific+="&0"
                        else:
                            s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2countrapp"])
                            sspecific+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2countrapp"])
                        if(results[fname][noise][pq][tvalstr]["l2notcountrapp"] == 0):
                            s+="&0"
                            sspecific+="&0"
                        else:
                            s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2notcountrapp"])
                            sspecific+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2notcountrapp"])
                s+="\\\\\\cline{2-12}\n"
                s+="&$r_{N=0}$ (Algorithm \\ref{A:Polyak})"
                sspecific+="\\\\\\cline{2-12}\n"
                sspecific+="&$r_{N=0}$ (Algorithm \\ref{A:Polyak})"
                for noise in noisearr:
                    tvalstr = str(int(thresholdvalarr[0]))
                    if(results[fname][noise][pq][tvalstr]["l2allpapp"] == 0):
                        s+="&0"
                        sspecific+="&0"
                    else:
                        s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2allpapp"])
                        sspecific+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2allpapp"])
                    s+="&\\multicolumn{4}{c|}{}"
                    sspecific+="&\\multicolumn{4}{c|}{}"
                s+="\\\\\\cline{2-12}\n"
                s+="\\hline\n\n"
                sspecific+="\\\\\\cline{2-12}\n"
                sspecific+="\\hline\n\n"
                if (fname=='f3' and pq == "p4_q3")\
                    or (fname=='f5' and pq == "p2_q3")\
                    or (fname=='f8' and pq == "p3_q3")\
                    or (fname=='f9' and pq == "p3_q7")\
                    or (fname=='f13' and pq == "p2_q7")\
                    or (fname=='f14' and pq == "p3_q6")\
                    or (fname=='f18' and pq == "p2_q3")\
                    or (fname=='f19' and pq == "p3_q3"):
                    print(sspecific)

    print(s)


if __name__ == "__main__":
    import os, sys

    if(sys.argv[1] == 'gen'):
        generatedata()
        exit(0)


 # python tablepoles.py f1,f2,f3,f4,f5,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f19,f20,f22  0,10-1 10,100,1000 2x  table
 # for fno in {1..5} {7..10} {12..20} 22; do  name="f"$fno; nohup python tablepoles.py $name 0,10-1 10,100,1000 2x  table> ../../debug/"tablepoles_"$name".log" 2>&1 & done
 # for fno in 3 5 9 13 14 18 19; do  name="f"$fno; nohup python tablepoles.py $name 0,10-1 10,100,1000 2x  latex> ../../debug/"tablepoles_latex_"$name".log" 2>&1 & done

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

    # testfilearr = sys.argv[5].split(',')
    # if len(testfilearr) == 0:
    #     print("please specify comma saperated testfile paths")
    #     sys.exit(1)
    #
    # bottomallarr = sys.argv[6].split(',')
    # if len(bottomallarr) == 0:
    #     print("please specify comma saperated bottom or all options")
    #     sys.exit(1)



    # tablepoles(farr,noisearr, thresholdarr, testfilearr, bottomallarr,sys.argv[4],sys.argv[7])
    tablepoles(farr,noisearr, thresholdarr, sys.argv[4],sys.argv[5],int(sys.argv[6]))

    # import matplotlib.pyplot as plt
    #
    # fname = "f12"
    # fn0 = "../benchmarkdata/"+fname+".txt"
    # fnn = "../benchmarkdata/"+fname+"_noisepct10-1.txt"
    #
    # dim = 2
    # m1= 2
    # n1= 3
    #
    # m2 = 3
    # n2= 1
    #
    # dof1 = tools.numCoeffsRapp(dim,[m1,n1])
    # dof2 = tools.numCoeffsRapp(dim,[m2,n2])
    #
    # X0, Y0 = readData(fn0)
    # Xn, Yn = readData(fnn)
    #
    # plt.scatter(X0[:2*dof1,0],X0[:2*dof1,1])
    # # plt.show()
    # plt.scatter(Xn[:2*dof2,0],Xn[:2*dof2,1])
    #
    #
    # plt.show()
    #
    # plt.clf()
    #
    # np.random.seed(54321)
    # X0 = np.random.uniform(np.array([-1,-1]),np.array([1,1]),(2*dof1,2))
    # Xn = np.random.uniform(np.array([-1,-1]),np.array([1,1]),(2*dof2,2))
    #
    # plt.scatter(X0[:,0],X0[:,1])
    # plt.show()
    # plt.scatter(Xn[:,0],Xn[:,1])
    #
    #
    # # plt.show()






###########
