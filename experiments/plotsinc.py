
import apprentice
import numpy as np
import os, sys
import json
from apprentice import RationalApproximationSIP
from apprentice import tools, readData
from mpl_toolkits.mplot3d import Axes3D

def sinc(X,dim):
    ret = 10
    for d in range(dim):
        x = X[d]
        ret *= np.sin(x)/x
    return ret

def raNorm(ra, X, Y, norm=2):
    nrm = 0
    for num, x in enumerate(X):
        nrm+= abs(ra.predict(x) - Y[num])**norm
    return nrm

def tablesinc(m,n,ts,table_or_latex):
    from apprentice import monomial
    print(apprentice.monomialStructure(3, 3))
    fname = "f20"

    larr = [10**-6,10**-3]
    uarr = [2*np.pi,4*np.pi]
    lbdesc = {0:"-6",1:"-3"}
    ubdesc = {0:"2pi",1:"4pi"}
    lblatex = {0:"$10^{-6}$",1:"$10^{-3}$"}
    ublatex = {0:"$2\\pi$",1:"$4\\pi$"}


    noisestr = ""

    folder = "%s%s_%s/sincrun"%(fname,noisestr,ts)
    if not os.path.exists(folder):
        print("folder %s not found")

    if not os.path.exists(folder+"/benchmarkdata"):
        os.mkdir(folder+'/benchmarkdata')

    data = {}
    for dim in range(2,8):
        data[dim] = {}
        for numlb,lb in enumerate(larr):
            for numub,ub in enumerate(uarr):
                key = lbdesc[numlb]+ubdesc[numub]
                data[dim][key] = {}
                fndesc = "%s%s_%s_p%d_q%d_ts%s_d%d_lb%s_ub%s"%(fname,noisestr,ts,m,n, ts, dim,lbdesc[numlb],ubdesc[numub])
                file = folder+"/"+fndesc+'/out/'+fndesc+"_p"+str(m)+"_q"+str(n)+"_ts"+ts+".json"
                if not os.path.exists(file):
                    print("%s not found"%(file))
                    exit(1)

                if file:
                    with open(file, 'r') as fn:
                        datastore = json.load(fn)

                data[dim][key]['l2error'] = 0


                testfile = "%s/benchmarkdata/%s%s_d%d_lb%s_ub%s_test.csv"%(folder,fname,noisestr,dim,lbdesc[numlb],ubdesc[numub])
                if not os.path.exists(testfile):
                    print("%s not found"%(testfile))
                    exit(1)
                bottom_or_all = all
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

                rappsip = RationalApproximationSIP(datastore)
                Y_pred_rappsip = rappsip.predictOverArray(X_test)
                Y_diff = (Y_pred_rappsip-Y_test)**2
                print(dim,key)
                print(np.c_[Y_test[1:10],Y_pred_rappsip[1:10],Y_diff[1:10]])
                l2allrappsip = np.sum((Y_pred_rappsip-Y_test)**2)
                l2allrappsip = np.sqrt(l2allrappsip)
                data[dim][key]['l2error'] = l2allrappsip



                rappsiptime = datastore['log']['fittime']
                rdof = int(datastore['M'] + datastore['N'])
                rnoiters = len(datastore['iterationinfo'])
                rpnnl = datastore['M'] - (dim+1)
                rqnnl = datastore['N'] - (dim+1)
                data[dim][key]['rappsiptime'] = rappsiptime
                data[dim][key]['rdof'] = rdof
                data[dim][key]['rnoiters'] = rnoiters
                data[dim][key]['rpnnl'] = rappsiptime
                data[dim][key]['rqnnl'] = rqnnl

    # print(data)
    s =""
    if(table_or_latex == "table"):
        print("TBD")
    elif(table_or_latex =="latex"):
        for dim in range(2,8):
            for numlb,lb in enumerate(larr):
                for numub,ub in enumerate(uarr):

                    key = lbdesc[numlb]+ubdesc[numub]
                    s += "%d&%d&%d&%s&%s&%.3f&%d&%.3f"%(dim,data[dim][key]['rdof'],data[dim][key]['rqnnl'],
                                lblatex[numlb],ublatex[numub],data[dim][key]['rappsiptime'],data[dim][key]['rnoiters'],
                                data[dim][key]['l2error'])
                    s+="\\\\\hline\n"
    # print(s)

    import matplotlib.pyplot as plt
    X = range(2,8)
    rangearr = []
    labelarr = []
    for numub,ub in enumerate(uarr):
        for numlb,lb in enumerate(larr):
            rangearr.append(lbdesc[numlb]+ubdesc[numub])
            labelarr.append(lblatex[numlb]+ " - "+ ublatex[numub])
    for r in rangearr:
        Y = []
        for x in X:
            Y.append(data[x][r]['l2error'])
        plt.plot(X,np.log10(Y), linewidth=1)
    plt.legend(labelarr,loc='upper right')
    # plt.show()
    plt.savefig("/Users/mkrishnamoorthy/Desktop/sincerror.pdf")
    plt.clf()

    # ##############################################

    import matplotlib.pyplot as plt
    X = range(2,8)
    rangearr = []
    labelarr = []
    for numub,ub in enumerate(uarr):
        for numlb,lb in enumerate(larr):
            rangearr.append(lbdesc[numlb]+ubdesc[numub])
            labelarr.append(lblatex[numlb]+ " - "+ ublatex[numub])
    for r in rangearr:
        Y = []
        for x in X:
            Y.append(data[x][r]['rnoiters'])
        plt.plot(X,np.log10(Y), linewidth=1)
    plt.legend(labelarr,loc='upper left')
    # plt.show()
    plt.savefig("/Users/mkrishnamoorthy/Desktop/sinc.pdf")
    plt.clf()

    exit(1)
    # ##############################################
    dim =3
    fndesc = "%s%s_%s_p%d_q%d_ts%s_d%d_lb%s_ub%s"%(fname,noisestr,ts,m,n, ts, dim,lbdesc[0],ubdesc[1])
    file = folder+"/"+fndesc+'/out/'+fndesc+"_p"+str(m)+"_q"+str(n)+"_ts"+ts+".json"
    if not os.path.exists(file):
        print("%s not found"%(file))

    if file:
        with open(file, 'r') as fn:
            datastore = json.load(fn)

    iterinfo = datastore['iterationinfo']
    print("#################")
    for iter in iterinfo:
        print(iter['robOptInfo']['robustArg'])
    print("#################")

    rappsip = RationalApproximationSIP(datastore)

    X1vals = np.arange(lb,ub,0.1)
    X2vals = np.arange(lb,ub,0.1)
    X3vals = np.arange(lb,ub,0.1)
    print(len(X1vals)*len(X2vals)*len(X3vals))

    Y_pred = []
    Y_orig = []
    for x1 in X1vals:
        for x2 in X2vals:
            for x3 in X3vals:
                Y_pred.append(rappsip([x1,x2,x3]))
                Y_orig.append(sinc([x1,x2,x3],3))
    l22 =np.sum((np.array(Y_pred)-np.array(Y_orig))**2)
    l22 = l22/(len(X1vals)*len(X2vals)*len(X3vals))

    print("\nUnscaled\n")
    print(datastore['scaler'])
    print("#################")
    for iter in iterinfo:
        x = rappsip._scaler.unscale(iter['robOptInfo']['robustArg'])
        print(x)

    print("#################")
    print("Min max  for n=3 after final iteration")
    print(min(Y_pred),max(Y_pred))
    print(min(Y_orig),max(Y_orig))
    print("#################")

    print("#################")
    print("\nMean error after the final approximation = %f\n"%(l22))
    print("#################")


    datastore['pcoeff'] = iterinfo[0]['pcoeff']
    datastore['qcoeff'] = iterinfo[0]['qcoeff']
    rappsip = RationalApproximationSIP(datastore)
    lb = larr[0]
    ub = uarr[1]

    Y_pred = []
    Y_orig = []
    for x1 in X1vals:
        for x2 in X2vals:
            for x3 in X3vals:
                Y_pred.append(rappsip([x1,x2,x3]))
                Y_orig.append(sinc([x1,x2,x3],3))
    print("#################")
    print("Min max  for n=3 after first iteration")
    print(min(Y_pred),max(Y_pred))
    print(min(Y_orig),max(Y_orig))
    l22 =np.sum((np.array(Y_pred)-np.array(Y_orig))**2)
    l22 = l22/(len(X1vals)*len(X2vals)*len(X3vals))
    print("#################")
    print("\nMean error after the first approximation = %f\n"%(l22))
    print("#################")
    print("#################")
    print("#################")
    print("#################")
    # exit(1)
    # ##############################################
    # Plotting
    import matplotlib.pyplot as plt


    if file:
        with open(file, 'r') as fn:
            datastore = json.load(fn)

    iterinfo = datastore['iterationinfo']
    iterinfono = len(iterinfo)
    for iterno in range(iterinfono):
        if file:
            with open(file, 'r') as fn:
                datastore = json.load(fn)
        iterinfo = datastore['iterationinfo']
        datastore['pcoeff'] = iterinfo[iterno]['pcoeff']
        datastore['qcoeff'] = iterinfo[iterno]['qcoeff']
        rappsip = RationalApproximationSIP(datastore)
        fig = plt.figure(figsize=(15,15))
        for num,s in enumerate(['x1=-1','x2=-1','x3-1']):
            other1 = []
            other2 = []
            Y_pred=[]
            Y_orig=[]
            q_pred=[]

            for x2 in X1vals:
                for x3 in X1vals:
                    if(num == 0):
                        X111 = [lb,x2,x3]
                    if(num == 1):
                        X111 = [x2,lb,x3]
                    if(num == 2):
                        X111 = [x2,x3,lb]
                    other1.append(x2)
                    other2.append(x3)
                    Y_pred.append(rappsip(X111))
                    Y_orig.append(sinc(X111,3))
                    X111 = rappsip._scaler.scale(np.array(X111))
                    q_pred.append(rappsip.denom(X111))

            # Y_pred = np.reshape(np.array(Y_pred), [len(other1), len(other2)])
            # Y_orig = np.reshape(np.array(Y_orig), [len(other1), len(other2)])
            # q_pred = np.reshape(np.array(q_pred), [len(other1), len(other2)])

            ax = fig.add_subplot(3, 3, 3*num+1, projection='3d')
            ax.plot3D(other1, other2, Y_orig ,"b.",alpha=0.5)
            ax.set_xlabel("x2")
            ax.set_ylabel("x3")
            ax = fig.add_subplot(3, 3, 3*num+2, projection='3d')
            ax.plot3D(other1, other2, Y_pred ,"r.",alpha=0.5)
            ax.set_xlabel("x2")
            ax.set_ylabel("x3")
            ax = fig.add_subplot(3, 3, 3*num+3, projection='3d')
            ax.plot3D(other1, other2, q_pred ,"g.",alpha=0.5)
            ax.set_xlabel("x2")
            ax.set_ylabel("x3")
        plt.savefig("/Users/mkrishnamoorthy/Desktop/sinc/iter" +str(iterno)+".pdf")

        plt.clf()
    exit(1)
    # ##############################################
    dim =4
    fndesc = "%s%s_%s_p%d_q%d_ts%s_d%d_lb%s_ub%s"%(fname,noisestr,ts,m,n, ts, dim,lbdesc[0],ubdesc[1])
    file = folder+"/"+fndesc+'/out/'+fndesc+"_p"+str(m)+"_q"+str(n)+"_ts"+ts+".json"
    if not os.path.exists(file):
        print("%s not found"%(file))

    if file:
        with open(file, 'r') as fn:
            datastore = json.load(fn)

    iterinfo = datastore['iterationinfo']
    print("#################")
    for iter in iterinfo:
        print(iter['robOptInfo']['robustArg'])
    print("#################")

    rappsip = RationalApproximationSIP(datastore)

    X1vals = np.arange(lb,ub,0.3)
    X2vals = np.arange(lb,ub,0.3)
    X3vals = np.arange(lb,ub,0.3)
    X4vals = np.arange(lb,ub,0.3)
    for x1 in X1vals:
        for x2 in X2vals:
            for x3 in X3vals:
                for x4 in X4vals:
                    Y_pred.append(rappsip([x1,x2,x3,x4]))
                    Y_orig.append(sinc([x1,x2,x3,x4],4))
    print("min max for n=4 after final iteration")
    print(min(Y_pred),max(Y_pred))
    print(min(Y_orig),max(Y_orig))
    l22 =np.sum((np.array(Y_pred)-np.array(Y_orig))**2)
    l22 = l22/(len(X1vals)*len(X2vals)*len(X3vals)*len(X4vals))


    print(len(X1vals)*len(X2vals)*len(X3vals)*len(X4vals))

    print("\nUnscaled\n")
    print(datastore['scaler'])
    print("#################")
    for iter in iterinfo:
        x = rappsip._scaler.unscale(iter['robOptInfo']['robustArg'])
        print(x)
    print("#################")

    print("#################")
    print("Min max  for n=4 after final iteration")
    print(min(Y_pred),max(Y_pred))
    print(min(Y_orig),max(Y_orig))
    print("#################")
    print("#################")
    print("Mean error after the final approximation = %f\n"%(l22))
    print("#################")

    datastore['pcoeff'] = iterinfo[0]['pcoeff']
    datastore['qcoeff'] = iterinfo[0]['qcoeff']
    rappsip = RationalApproximationSIP(datastore)
    lb = larr[0]
    ub = uarr[1]


    Y_pred = []
    Y_orig = []
    for x1 in X1vals:
        for x2 in X2vals:
            for x3 in X3vals:
                for x4 in X4vals:
                    Y_pred.append(rappsip([x1,x2,x3,x4]))
                    Y_orig.append(sinc([x1,x2,x3,x4],4))
    print("#################")
    print("Min max  for n=4 after final iteration")
    print(min(Y_pred),max(Y_pred))
    print(min(Y_orig),max(Y_orig))
    print("#################")
    l22 =np.sum((np.array(Y_pred)-np.array(Y_orig))**2)
    l22 = l22/(len(X1vals)*len(X2vals)*len(X3vals)*len(X4vals))
    print("#################")
    print("\nMean error after the first approximation = %f\n"%(l22))
    # ##############################################













if __name__ == "__main__":

    if len(sys.argv)!=5:
        print("Usage: {} m n ts table_or_latex".format(sys.argv[0]))
        sys.exit(1)


    tablesinc(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],sys.argv[4])
