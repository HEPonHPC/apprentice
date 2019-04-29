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

def getpqstr(fname):
    pq = ""
    if (fname=='f1'): pq = "p2_q4"
    if (fname=='f2'): pq = "p5_q2"
    if (fname=='f3'): pq = "p4_q3"
    if (fname=='f4'): pq = 'p3_q2'
    if (fname=='f5') : pq = "p2_q3"
    if (fname=='f7') : pq = 'p2_q7'
    if fname=='f8' : pq = "p3_q3"
    if fname=='f9' : pq = "p3_q7"
    if fname=='f10' : pq = 'p2_q4'
    if fname=='f12' : pq = 'p3_q3'
    if fname=='f13' : pq = "p2_q7"
    if fname=='f14' : pq = "p3_q6"
    if fname=='f15' : pq = "p2_q5"
    if fname=='f16' : pq = "p3_q7"
    if fname=='f17' : pq = 'p4_q6'
    if fname=='f18' : pq = "p2_q3"
    if fname=='f19' : pq = "p3_q3"
    if fname=='f20' : pq = "p2_q3"
    if fname=='f21' : pq = "p5_q2"
    if fname=='f22' : pq = "p2_q4"
    return pq

def getdim(fname):
    dim = {"f1":2,"f2":2,"f3":2,"f4":2,"f5":2,"f7":2,"f8":2,"f9":2,"f10":4,"f12":2,"f13":2,
            "f14":2,"f15":2,"f16":2,"f17":3,"f18":4,"f19":4,"f20":7,"f21":2,"f22":2}
    return dim[fname]

def getX(dim,Xr,cornerpoints):
    X = np.array([])
    for ddd in range(dim):
        X = np.append(X,10**-12)
    if dim==2:
        for ddd in range(dim):
            for corner in cornerpoints:
                for a in Xr:
                    if(ddd==0):
                        xt = [corner,a]
                    elif(ddd==1):
                        xt = [a,corner]
                    X = np.vstack([X, xt])

    elif dim ==3:
        for ddd in range(dim):
            for corner in cornerpoints:
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
            for corner in cornerpoints:
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
            for corner in cornerpoints:
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

    stdnormalnoise = np.zeros(shape = (len(Y_train)), dtype =np.float64)
    for i in range(len(Y_train)):
        stdnormalnoise[i] = np.random.normal(0,1)

    return np.atleast_2d(np.array(Y_train)*(1+ noisepct*stdnormalnoise))

def tablepoles(farr,noisearr, tarr, ts, table_or_latex):
    print (farr)
    print (noisearr)
    print (thresholdarr)

    if not os.path.exists("results/plots"):
        os.makedirs("results/plots", exist_ok = True)

    # allsamples = ['mc','lhs','sc','sg']
    allsamples = ['sg']

    thresholdvalarr = np.array([float(t) for t in tarr])
    thresholdvalarr = np.sort(thresholdvalarr)
    results = {}

    # import glob
    import json
    import math
    # import re
    m=5
    n=5
    for num,fname in enumerate(farr):
        results[fname] = {}
        dim = getdim(fname)
        Xr = np.linspace(-1, 1, num=math.ceil(10**(6/dim)))
        # Xr = np.linspace(-1, 1, num=10)
        xrindicies = np.zeros(dim,dtype=np.int64)
        X_test = getX(dim,Xr,[-1,1])
        Y_test = getData(X_test,fname,0)
        maxY_test = max(1,abs(np.max(Y_test)))
        print(fname,maxY_test)
        for snum, sample in enumerate(allsamples):
            results[fname][sample] = {}
            for noise in noisearr:
                results[fname][noise] = {}
                noisestr = ""
                if(noise!="0"):
                    noisestr = "_noisepct"+noise
                for run in ["exp1","exp2","exp3","exp4","exp5"]:
                    fndesc = "%s%s_%s_%s"%(fname,noisestr,sample,ts)
                    folder = "results/%s/%s"%(run,fndesc)
                    # print(folder)
                    pq = "p%d_q%d"%(m,n)
                    # print(run, fname,noisestr,sample,m,n)

                    rappsipfile = "%s/outrasip/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)
                    rappfile = "%s/outra/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)

                    if not os.path.exists(rappsipfile):
                        print("rappsipfile %s not found"%(rappsipfile))
                        exit(1)

                    if not os.path.exists(rappfile):
                        print("rappfile %s not found"%(rappfile))
                        exit(1)


                    rappsip = RationalApproximationSIP(rappsipfile)
                    Y_pred_rappsip = rappsip.predictOverArray(X_test)
                    rapp = RationalApproximationONB(fname=rappfile)
                    Y_pred_rapp = np.array([rapp(x) for x in X_test])
                    # papp = PolynomialApproximation(fname=pappfile)
                    # Y_pred_papp = np.array([papp(x) for x in X_test])
                    # # results[fname][noise] = {"rapp":{},"rappsip":{}}
                    # print(maxY_test)
                    maxY_test = max(1,abs(maxY_test))
                    # for tval in thresholdvalarr:
                    #     # print(fname, maxY_test)
                    #     # print(Y_pred_rappsip)
                    #
                    #     # rappsipcount = ((sum(abs(i)/abs(maxY_test) >= tval for i in Y_pred_rappsip))/float(len(Y_test))) *100
                    #     # rappcount = ((sum(abs(i)/abs(maxY_test) >= tval for i in Y_pred_rapp))/float(len(Y_test))) *100
                    #
                    #     l2allrappsip = np.sum((Y_pred_rappsip-Y_test)**2)
                    #     l2countrappsip = 0.
                    #     rappsipcount = 0
                    #     for num,yp in enumerate(Y_pred_rappsip):
                    #         if abs(yp)/abs(maxY_test) >= tval:
                    #             rappsipcount+=1
                    #             l2countrappsip += np.sum((yp-Y_test[num])**2)
                    #     l2notcountrappsip = l2allrappsip - l2countrappsip
                    #
                    #     l2countrappsip = np.sqrt(l2countrappsip)
                    #     l2notcountrappsip = np.sqrt(l2notcountrappsip)
                    #     l2allrappsip = np.sqrt(l2allrappsip)
                    #
                    #     l2allrapp = np.sum((Y_pred_rapp-Y_test)**2)
                    #     l2countrapp = 0.
                    #     rappcount = 0
                    #     for num,yp in enumerate(Y_pred_rapp):
                    #         if abs(yp)/abs(maxY_test) >= tval:
                    #             rappcount+=1
                    #             l2countrapp += np.sum((yp-Y_test[num])**2)
                    #     l2notcountrapp = l2allrapp - l2countrapp
                    #
                    #     l2countrapp = np.sqrt(l2countrapp)
                    #     l2notcountrapp = np.sqrt(l2notcountrapp)
                    #     l2allrapp = np.sqrt(l2allrapp)

                    for tval in thresholdvalarr:
                        # print(fname, maxY_test)
                        # print(Y_pred_rappsip)

                        # rappsipcount = ((sum(abs(i)/abs(maxY_test) >= tval for i in Y_pred_rappsip))/float(len(Y_test))) *100
                        # rappcount = ((sum(abs(i)/abs(maxY_test) >= tval for i in Y_pred_rapp))/float(len(Y_test))) *100

                        # l2allrappsip = np.sum((Y_pred_rappsip-Y_test)**2)
                        # l2countrappsip = 0.
                        # rappsipcount = 0
                        # for num,yp in enumerate(Y_pred_rappsip):
                        #     if abs(yp)/abs(maxY_test) >= tval:
                        #         rappsipcount+=1
                        #         l2countrappsip += np.sum((yp-Y_test[num])**2)
                        # l2notcountrappsip = l2allrappsip - l2countrappsip
                        #
                        # l2countrappsip = np.sqrt(l2countrappsip)
                        # l2notcountrappsip = np.sqrt(l2notcountrappsip)
                        # l2allrappsip = np.sqrt(l2allrappsip)

                        l2allrapp = np.sum((Y_pred_rapp-Y_test)**2)
                        l2countrapp = 0.
                        rappcount100 = 0
                        rappcount1000 = 0
                        rappsipcount100 = 0
                        rappsipcount1000 = 0
                        # for num,yp in enumerate(Y_pred_rapp):
                        import math
                        dim = rappsip.dim
                        Xd = np.linspace(-1, 1, num=1000)
                        # Xd = np.linspace(-1, 1, num=int(1000000/dim))
                        # for a in Xd:
                        #     for b in Xd:
                        #         for c in Xd:
                        #             for  d in Xd:
                        #                 xt = [a,b,c,d]
                        #                 xtscaled = np.array(xt)
                        #                 yt = sinc(xtscaled,dim)
                        #                 if math.isnan(yt) == False:
                        #
                        #                     numer = rappsip.numer(xtscaled)
                        #                     denom = rappsip.denom(xtscaled)
                        #
                        #                     yp =numer/denom
                        #                     if math.isnan(yp)==False:
                        #                         if abs(yp)/abs(maxY_test) >= 100:
                        #                             print ("rappsip",xtscaled,abs(yp)/abs(maxY_test))
                        #                             rappsipcount100 +=1
                        #                         if abs(yp)/abs(maxY_test) >= 1000:
                        #                             rappsipcount1000 +=1
                        #                     numer = rapp.numer(xtscaled)
                        #                     denom = rapp.denom(xtscaled)
                        #
                        #                     yp =numer/denom
                        #                     if math.isnan(yp)==False:
                        #
                        #                         if abs(yp)/abs(maxY_test) >= 100:
                        #                             print ("rapp",xtscaled,abs(yp)/abs(maxY_test),yt)
                        #                             rappcount100+=1
                        #                         if abs(yp)/abs(maxY_test) >= 1000:
                        #                             rappcount1000+=1
                        #
                        # print(rappsipcount100,rappsipcount1000,rappcount100,rappcount1000)
                        # exit(1)
                        #                     # l2countrapp += np.sum((yp-Y_test[num])**2)
                        # l2notcountrapp = l2allrapp - l2countrapp
                        #
                        # l2countrapp = np.sqrt(l2countrapp)
                        # l2notcountrapp = np.sqrt(l2notcountrapp)
                        # l2allrapp = np.sqrt(l2allrapp)
                    import math

                    dim = rappsip.dim

                    ddd = 0
                    # Xd = np.linspace(-1, 1, num=math.ceil(10**(6/dim)))
                    Xd = np.linspace(-1, 1, num=math.ceil(1000))
                    no=0

                    for ddd in range(dim):
                        for corner in [-1.,1.]:
                            for a in Xd:
                                for b in Xd:
                                    # for c in Xd:
                                    if(ddd==0):
                                        xt = [corner,a,b]
                                    elif(ddd==1):
                                        xt = [a,corner,b]
                                    elif(ddd==2):
                                        xt = [a,b,corner]
                                    # elif(ddd==3):
                                    #     xt = [a,b,c,corner]
                                    # if(corner == -1 and b==-1):
                                    #     print (xt)
                                    no+=1
                                        # xtscaled = rappsip._scaler.scale(xt)
                                    xtscaled = np.array(xt)
                                    numer = rappsip.numer(xtscaled)
                                    denom = rappsip.denom(xtscaled)
                                    nl10 = np.log10(abs(numer))
                                    dl10 = np.log10(abs(denom))
                                    ymaxl10 = round(np.log10(abs(maxY_test)))
                                    yp=numer/denom
                                    if(abs(yp)/maxY_test > 100):
                                        print ("rappsip",xtscaled,round(nl10),round(dl10),numer/denom)
                                        exit(1)

                                        # print(dl10)
                                        # if dl10 <= -6:
                                        #     if(nl10<0 and ((abs(round(nl10)-ymaxl10))-abs(round(dl10))) <= -2):
                                        #         print ("rappsip",round(nl10),round(dl10),numer/denom)
                                        #         exit(1)
                                        #     # elif(nl10>0):
                                        #     #     print ("rappsip",round(nl10),round(dl10),numer/denom)
                                        #     # print ("rappsip",round(nl10),round(dl10))
                                        #     # xtscaled = rapp._scaler.scale(xt)
                                        #
                                    numer = rapp.numer(xtscaled)
                                    denom = rapp.denom(xtscaled)
                                    nl10 = np.log10(abs(numer))
                                    dl10 = np.log10(abs(denom))
                                    yp=numer/denom
                                    if(abs(yp)/maxY_test > 100):
                                        print ("rapp",xtscaled,round(nl10),round(dl10),numer/denom)
                                    # print(dl10)
                                    # if dl10 <= -6:
                                    #
                                    #     if(nl10<0 and ((abs(round(nl10)-ymaxl10))-abs(round(dl10))) <= -2):
                                    #         print ("rapp",round(nl10),round(dl10),numer/denom)
                                    #     # elif(nl10>0):
                                    #     #     print ("rapp",round(nl10),round(dl10),numer/denom)
                    print(no)



                    # # Use recursion later
                    # dim = rappsip.dim
                    # no = 0
                    # Xd = np.linspace(-1, 1, num=int(1000000/dim))
                    # for a in Xd:
                    #     for b in Xd:
                    #         for c in Xd:
                    #             for  d in Xd:
                    #                 xt = [a,b,c,d]
                    #                 no+=1
                    #                 if(no>100000):
                    #                     exit(1)
                    #                 # xtscaled = rappsip._scaler.scale(xt)
                    #                 xtscaled = np.array(xt)
                    #                 numer = rappsip.numer(xtscaled)
                    #                 denom = rappsip.denom(xtscaled)
                    #                 nl10 = np.log10(abs(numer))
                    #                 dl10 = np.log10(abs(denom))
                    #                 ymaxl10 = round(np.log10(abs(maxY_test)))
                    #                 yp=numer/denom
                    #                 if(abs(yp)/maxY_test > 100):
                    #                     print ("rappsip",round(nl10),round(dl10),numer/denom)
                    #                     exit(1)
                    #                 print(xtscaled)
                    #                 # print(dl10)
                    #                 # if dl10 <= -6:
                    #                 #     if(nl10<0 and ((abs(round(nl10)-ymaxl10))-abs(round(dl10))) <= -2):
                    #                 #         print ("rappsip",round(nl10),round(dl10),numer/denom)
                    #                 #         exit(1)
                    #                 #     # elif(nl10>0):
                    #                 #     #     print ("rappsip",round(nl10),round(dl10),numer/denom)
                    #                 #     # print ("rappsip",round(nl10),round(dl10))
                    #                 #     # xtscaled = rapp._scaler.scale(xt)
                    #                 #
                    #                 numer = rapp.numer(xtscaled)
                    #                 denom = rapp.denom(xtscaled)
                    #                 nl10 = np.log10(abs(numer))
                    #                 dl10 = np.log10(abs(denom))
                    #                 yp=numer/denom
                    #                 if(abs(yp)/maxY_test > 100):
                    #                     print ("rapp",round(nl10),round(dl10),numer/denom)
                    #                 # print(dl10)
                    #                 # if dl10 <= -6:
                    #                 #
                    #                 #     if(nl10<0 and ((abs(round(nl10)-ymaxl10))-abs(round(dl10))) <= -2):
                    #                 #         print ("rapp",round(nl10),round(dl10),numer/denom)
                    #                 #     # elif(nl10>0):
                    #                 #     #     print ("rapp",round(nl10),round(dl10),numer/denom)
                    #
                    # exit(1)

                        # l2allpapp = np.sum((Y_pred_papp-Y_test)**2)



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

        #                 data = {
        #                     'm':optm,
        #                     'n':optn,
        #                     'rapp':str(int(rappcount)),
        #                     'rappsip':str(int(rappsipcount)),
        #                     'l2countrappsip' : l2countrappsip,
        #                     'l2notcountrappsip' : l2notcountrappsip,
        #                     'l2allrappsip' : l2allrappsip,
        #                     'l2countrapp' : l2countrapp,
        #                     'l2notcountrapp' : l2notcountrapp,
        #                     'l2allrapp' : l2allrapp
        #                     # 'l2allpapp': l2allpapp
        #                 }
        #                 tvalstr = str(int(tval))
        #                 pq = "p%d_q%d"%(optm,optn)
        #
        #                 if(pq in results[fname][noise]):
        #                     resultsdata = results[fname][noise][pq]
        #                     resultsdata[tvalstr] = data
        #                 else:
        #                     results[fname][noise][pq] = {tvalstr:data}
        #
        #
        #                 # results[fname][noise][tvalstr] = str(int(rappcount))
        #                 # results[fname][noise][tvalstr] = str(int(rappsipcount))
        #
        # print(results)
        # print (json.dumps(results,indent=4, sort_keys=True))


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


 # python tablepoles.py f1,f2,f3,f4,f5,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f19,f20,f22  0,10-1 10,100,1000 2x  table
 # for fno in {1..5} {7..10} {12..20} 22; do  name="f"$fno; nohup python tablepoles.py $name 0,10-1 10,100,1000 2x  table> ../../debug/"tablepoles_"$name".log" 2>&1 & done
 # for fno in 3 5 9 13 14 18 19; do  name="f"$fno; nohup python tablepoles.py $name 0,10-1 10,100,1000 2x  latex> ../../debug/"tablepoles_latex_"$name".log" 2>&1 & done
    import os, sys
    if len(sys.argv) != 6:
        print("Usage: {} function noise thresholds ts table_or_latex_or_latexall".format(sys.argv[0]))
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
    tablepoles(farr,noisearr, thresholdarr, sys.argv[4],sys.argv[5])

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
