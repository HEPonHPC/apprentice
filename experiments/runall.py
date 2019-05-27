import numpy as np
import os

def my_i4_sobol_generate(dim, n, seed):
    import sobol_seq
    r = np.full((n, dim), np.nan)
    currentseed = seed
    for j in range(n):
        r[j, 0:dim], newseed = sobol_seq.i4_sobol(dim, currentseed)
        currentseed = newseed
    return r

def getData(X_train, fn, noisepct,seed):
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
    np.random.seed(seed)

    stdnormalnoise = np.zeros(shape = (len(Y_train)), dtype =np.float64)
    for i in range(len(Y_train)):
        stdnormalnoise[i] = np.random.normal(0,1)
    # return Y_train
    return np.atleast_2d(np.array(Y_train)*(1+ noisepct*stdnormalnoise))
def getdim(fname):
    dim = {"f1":2,"f2":2,"f3":2,"f4":2,"f5":2,"f7":2,"f8":2,"f9":2,"f10":4,"f12":2,"f13":2,
            "f14":2,"f15":2,"f16":2,"f17":3,"f18":4,"f19":4,"f20":4,"f21":2,"f22":2}
    return dim[fname]

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

def getnoiseinfo(noise):
    noisearr = ["0","10-2","10-4","10-6"]
    noisestr = ["","_noisepct10-2","_noisepct10-4","_noisepct10-6"]
    noisepct = [0,10**-2,10**-4,10**-6]

    for i,n in enumerate(noisearr):
        if(n == noise):
            return noisestr[i],noisepct[i]


def getfarr():
    farr = ["f1","f2","f3","f4","f5","f7","f8","f9","f10","f12","f13","f14","f15","f16",
           "f17","f18","f19","f20","f21","f22"]
    # farr = ["f1","f2","f3","f4","f5","f7","f8","f9","f10","f12","f13","f14","f15","f16",
    #         "f17","f18","f19","f21","f22"]
    # farr = ["f20"]
    # farr = ["f17"]

    return farr

def generatespecialdata():
    from apprentice import tools
    from pyDOE import lhs
    m=5
    n=5
    dim =2
    farr  =["f8","f9","f12"]
    noisearr = ["0","10-12","10-10","10-8","10-6","10-4"]
    ts = 2
    npoints = ts * tools.numCoeffsRapp(dim,[int(m),int(n)])
    # sample = "sg"
    # from dolo.numeric.interpolation.smolyak import SmolyakGrid
    # s = 0
    # l = 2
    # while(s < npoints):
    #     sg = SmolyakGrid(a=[-1,-1],b=[1,1], l=l)
    #     s = sg.grid.shape[0]
    #     l+=1
    # X = sg.grid
    # lennn = sg.grid.shape[0]

    # import apprentice
    # sample = "lhs"
    # X = lhs(dim, samples=npoints, criterion='maximin')
    # s = apprentice.Scaler(np.array(X, dtype=np.float64), a=[-1,-1], b=[1,1])
    # X = s.scaledPoints
    # lennn = npoints

    import apprentice
    seed = 54321
    sample = "so"
    X = my_i4_sobol_generate(dim,npoints,seed)
    s = apprentice.Scaler(np.array(X, dtype=np.float64), a=[-1,-1], b=[1,1])
    X = s.scaledPoints
    lennn = npoints


    stdnormalnoise = np.zeros(shape = (lennn), dtype =np.float64)
    for i in range(lennn):
        stdnormalnoise[i] = np.random.normal(0,1)
    for fname in farr:
        minarr,maxarr = getbox(fname)

        for noise in noisearr:
            noisestr = ""
            noisepct = 0
            if(noise!="0"):
                noisestr = "_noisepct"+noise
            if(noise=="10-4"):
                noisepct=10**-4
            elif(noise=="10-6"):
                noisepct=10**-6
            elif(noise=="10-8"):
                noisepct=10**-8
            elif(noise=="10-10"):
                noisepct=10**-10
            elif(noise=="10-12"):
                noisepct=10**-12
            print(noisepct)
            Y = getData(X, fn=fname, noisepct=0,seed=seed)
            Y_train = np.atleast_2d(np.array(Y)*(1+ noisepct*stdnormalnoise))
            outfolder = "/Users/mkrishnamoorthy/Desktop/Data"
            outfile = "%s/%s%s_%s.txt"%(outfolder,fname,noisestr,sample)
            print(outfile)
            np.savetxt(outfile, np.hstack((X,Y_train.T)), delimiter=",")


def generatebenchmarkdata(m,n):
    seedarr = [54321,456789,9876512,7919820,10397531]
    folder= "results"
    samplearr = ["mc","lhs","so","sg","splitlhs"]
    # samplearr = ["lhs","sg","splitlhs"]
    # samplearr = ["lhs","splitlhs"]
    # samplearr = ["splitlhs"]
    from apprentice import tools
    from pyDOE import lhs
    import apprentice
    ts = 2
    farr = getfarr()
    for fname in farr:
        dim = getdim(fname)
        minarr,maxarr = getbox(fname)
        npoints = ts * tools.numCoeffsRapp(dim,[int(m),int(n)])
        print (npoints)
        for sample in samplearr:
            for numex,ex in enumerate(["exp1","exp2","exp3","exp4","exp5"]):
                seed = seedarr[numex]
                np.random.seed(seed)
                if(sample == "mc"):
                    Xperdim = ()
                    for d in range(dim):
                        Xperdim = Xperdim + (np.random.rand(npoints,)*(maxarr[d]-minarr[d])+minarr[d],)
                    X = np.column_stack(Xperdim)
                    formatStr = "{0:0%db}"%(dim)
                    for d in range(2**dim):
                        binArr = [int(x) for x in formatStr.format(d)[0:]]
                        val = []
                        for i in range(dim):
                            if(binArr[i] == 0):
                                val.append(minarr[i])
                            else:
                                val.append(maxarr[i])
                        X[d] = val
                elif(sample == "sg"):
                    from dolo.numeric.interpolation.smolyak import SmolyakGrid
                    s = 0
                    l=1
                    while(s<npoints):
                        sg = SmolyakGrid(a=minarr,b=maxarr, l=l)
                        s = sg.grid.shape[0]
                        l+=1
                    X = sg.grid
                elif(sample == "so"):
                    X = my_i4_sobol_generate(dim,npoints,seed)
                    s = apprentice.Scaler(np.array(X, dtype=np.float64), a=minarr, b=maxarr)
                    X = s.scaledPoints
                elif(sample == "lhs"):
                    X = lhs(dim, samples=npoints, criterion='maximin')
                    s = apprentice.Scaler(np.array(X, dtype=np.float64), a=minarr, b=maxarr)
                    X = s.scaledPoints
                elif(sample == "splitlhs"):
                    epsarr = []
                    for d in range(dim):
                        #epsarr.append((maxarr[d] - minarr[d])/10)
                        epsarr.append(10**-6)

                    facepoints = int(2 * tools.numCoeffsRapp(dim-1,[int(m),int(n)]))
                    insidepoints = int(npoints - facepoints)
                    Xmain = np.empty([0,dim])
                    # Generate inside points
                    minarrinside = []
                    maxarrinside = []
                    for d in range(dim):
                        minarrinside.append(minarr[d] + epsarr[d])
                        maxarrinside.append(maxarr[d] - epsarr[d])
                    X = lhs(dim, samples=insidepoints, criterion='maximin')
                    s = apprentice.Scaler(np.array(X, dtype=np.float64), a=minarrinside, b=maxarrinside)
                    X = s.scaledPoints
                    Xmain = np.vstack((Xmain,X))

                    #Generate face points
                    perfacepoints = int(np.ceil(facepoints/(2*dim)))
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
                    formatStr = "{0:0%db}"%(dim)
                    for d in range(2**dim):
                        binArr = [int(x) for x in formatStr.format(d)[0:]]
                        val = []
                        for i in range(dim):
                            if(binArr[i] == 0):
                                val.append(minarr[i])
                            else:
                                val.append(maxarr[i])
                        X[d] = val



                if not os.path.exists(folder+"/"+ex+'/benchmarkdata'):
                    os.makedirs(folder+"/"+ex+'/benchmarkdata',exist_ok = True)
                for noise in ["0","10-2","10-4","10-6"]:
                    noisestr,noisepct = getnoiseinfo(noise)

                    Y = getData(X, fn=fname, noisepct=noisepct,seed=seed)

                    outfile = "%s/%s/benchmarkdata/%s%s_%s.txt"%(folder,ex,fname,noisestr,sample)
                    print(outfile)
                    np.savetxt(outfile, np.hstack((X,Y.T)), delimiter=",")
                if(sample == "sg"):
                    break

def findfreecputhreads():
    freecputhreads = []
    threshold = 5
    import psutil
    cpup = psutil.cpu_percent(interval=5, percpu=True)
    for t in range(len(cpup)):
        if cpup[t] < threshold:
            freecputhreads.append(t)
    # for num,p in enumerate(cpup): print(num,p)
    # print(freecputhreads)
    # print(len(freecputhreads))
    return freecputhreads

def runall(type, sample, exparr,noise,m,n,pstarendarr):
    commands = []
    farr = getfarr()
    folder= "results"

    usetaskset = 0
    if int(pstarendarr[0]) == -1 and int(pstarendarr[1]) == -1:
        usetaskset = 0 # Dont use taskset

    elif int(pstarendarr[0]) == -2 and int(pstarendarr[1]) == -2:
        usetaskset = 2 # use auto taskset
        freecputhreads = findfreecputhreads()
        sum = len(freecputhreads)
        if sum < len(farr) * len(exparr):
            print("not enough processes. %d required and only have %d. Try again"%(len(farr),sum))
            exit(1)

    elif int(pstarendarr[0]) >= 0 and int(pstarendarr[1]) >= 0:
        usetaskset = 1 #use taskset using start,end thread# given
        sum = 0
        i=0
        while i < len(pstarendarr):
            sum += int(pstarendarr[i+1]) - int(pstarendarr[i])
            i+=2
        if sum < len(farr) * len(exparr):
            print("not enough processes. %d required and only have %d. Try again"%(len(farr) * len(exparr), sum))
            exit(1)

    pcurr = int(pstarendarr[0])
    pindex = 0
    autocpuindex = 0
    noisestr,noisepct = getnoiseinfo(noise)
    for fname in farr:
        fndesc = "%s%s_%s_2x"%(fname,noisestr,sample)
        for ex in exparr:
            folderplus = folder+"/"+ex+"/"+fndesc
            infile = "%s/%s/benchmarkdata/%s%s_%s.txt"%(folder,ex,fname,noisestr,sample)
            if not os.path.exists(infile):
                print("Infile %s not found"%infile)
                exit(1)
            if(type == "pa"):
                if not os.path.exists(folderplus + "/outpa"):
                    os.makedirs(folderplus + "/outpa",exist_ok = True)
                if not os.path.exists(folderplus + "/log/consolelogpa"):
                    os.makedirs(folderplus + "/log/consolelogpa",exist_ok = True)
                consolelog=folderplus + "/log/consolelogpa/"+fndesc+"_p"+m+"_q"+n+"_ts2x.log";
                outfile = folderplus + "/outpa/"+fndesc+"_p"+m+"_q"+n+"_ts2x.json";
                if not os.path.exists(outfile):
                    cmd = 'nohup python runpappforsimcoeffs.py %s %s %s %s Cp %s >%s 2>&1 &'%(infile,fndesc,m,n,outfile,consolelog)
                    if usetaskset ==2:
                        cmd = "taskset -c %d %s"%(freecputhreads[autocpuindex],cmd)
                        autocpuindex+=1
                    if usetaskset ==1:
                        cmd = "taskset -c %d %s"%(pcurr,cmd)
                        if(pcurr == int(pstarendarr[pindex+1])):
                            pcurr = int(pstarendarr[pindex+2])
                            pindex += 2
                        else:
                            pcurr += 1

                    commands.append(cmd)
                    # print(cmd)
                    os.system(cmd)
                    # exit(1)
            elif(type == "ra"):
                if not os.path.exists(folderplus + "/outra"):
                    os.makedirs(folderplus + "/outra",exist_ok = True)
                if not os.path.exists(folderplus + "/log/consolelogra"):
                    os.makedirs(folderplus + "/log/consolelogra",exist_ok = True)
                consolelog=folderplus + "/log/consolelogra/"+fndesc+"_p"+m+"_q"+n+"_ts2x.log";
                outfile = folderplus + "/outra/"+fndesc+"_p"+m+"_q"+n+"_ts2x.json";
                tol = -1
                denomfirst = -1
                if not os.path.exists(outfile):
                    cmd = 'nohup python runnonsiprapp.py %s %s %s %s Cp %f %d %s >%s 2>&1 &'%(infile,fndesc,m,n,tol,denomfirst,outfile,consolelog)
                    if usetaskset ==2:
                        cmd = "taskset -c %d %s"%(freecputhreads[autocpuindex],cmd)
                        autocpuindex+=1
                    if usetaskset ==1:
                        cmd = "taskset -c %d %s"%(pcurr,cmd)
                        if(pcurr == int(pstarendarr[pindex+1])):
                            pcurr = int(pstarendarr[pindex+2])
                            pindex += 2
                        else:
                            pcurr += 1
                    commands.append(cmd)
                    # print(cmd)
                    os.system(cmd)
                    # exit(1)
            elif(type == "rard"):
                if not os.path.exists(folderplus + "/outrard"):
                    os.makedirs(folderplus + "/outrard",exist_ok = True)
                if not os.path.exists(folderplus + "/log/consolelogrard"):
                    os.makedirs(folderplus + "/log/consolelogrard",exist_ok = True)
                consolelog=folderplus + "/log/consolelogrard/"+fndesc+"_p"+m+"_q"+n+"_ts2x.log";
                outfile = folderplus + "/outrard/"+fndesc+"_p"+m+"_q"+n+"_ts2x.json";
                if noise =="0":
                    tol = 10**-12
                else:
                    noisestr,noisepct = getnoiseinfo(noise)
                    # rard
                    tol = noisepct * 10

                    # rard1 (k/2)
                    # if noise =="10-2":
                    #     tol = 10**-1
                    # elif noise =="10-4":
                    #     tol = 10**-2
                    # elif noise =="10-6":
                    #     tol = 10**-3

                denomfirst = 1
                if not os.path.exists(outfile):
                    cmd = 'nohup python runnonsiprapp.py %s %s %s %s Cp %f %d %s >%s 2>&1 &'%(infile,fndesc,m,n,tol,denomfirst,outfile,consolelog)
                    if usetaskset ==2:
                        cmd = "taskset -c %d %s"%(freecputhreads[autocpuindex],cmd)
                        autocpuindex+=1
                    if usetaskset ==1:
                        cmd = "taskset -c %d %s"%(pcurr,cmd)
                        if(pcurr == int(pstarendarr[pindex+1])):
                            pcurr = int(pstarendarr[pindex+2])
                            pindex += 2
                        else:
                            pcurr += 1
                    commands.append(cmd)
                    # print(cmd)
                    os.system(cmd)
                    # exit(1)
            elif(type == "rasip"):
                if not os.path.exists(folderplus + "/outrasip"):
                    os.makedirs(folderplus + "/outrasip",exist_ok = True)
                if not os.path.exists(folderplus + "/log/consolelograsip"):
                    os.makedirs(folderplus + "/log/consolelograsip",exist_ok = True)
                consolelog=folderplus + "/log/consolelograsip/"+fndesc+"_p"+m+"_q"+n+"_ts2x.log";
                outfile = folderplus + "/outrasip/"+fndesc+"_p"+m+"_q"+n+"_ts2x.json";
                penaltyparam = 0
                if sample == 'sg':
                    if m == '5' and n == '5':
                        penaltyparam = 10**-1
                    else:
                        print("penalty param not defined for m = %s and n = %s"%(m,n))
                        exit(1)

                if not os.path.exists(outfile):
                    cmd = 'nohup python runrappsip.py %s %s %s %s Cp %f %s %s >%s 2>&1 &'%(infile,fndesc,m,n,penaltyparam,folderplus,outfile,consolelog)
                    if usetaskset ==2:
                        cmd = "taskset -c %d %s"%(freecputhreads[autocpuindex],cmd)
                        autocpuindex+=1
                    if usetaskset ==1:
                        cmd = "taskset -c %d %s"%(pcurr,cmd)
                        if(pcurr == int(pstarendarr[pindex+1])):
                            pcurr = int(pstarendarr[pindex+2])
                            pindex += 2
                        else:
                            pcurr += 1
                    commands.append(cmd)
                    # print(cmd)
                    os.system(cmd)
                    # exit(1)




    # cmdstr = ""
    # for cmd in commands:
    #     cmdstr+= cmd +"\n"
    # filename = folder+"/"+'commands.txt'
    #
    # if os.path.exists(filename):
    #     append_write = 'a'
    # else:
    #     append_write = 'w'
    #
    # f = open(filename,append_write)
    # f.write(cmdstr)
    # f.close()






# Approx 3 sampling 4 fn 20 noise 3 ex 5

if __name__ == "__main__":
    # generatespecialdata()
    # exit(1)

    import os, sys
    if sys.argv[1] == "gen":
        if len(sys.argv) != 4:
            print("Usage: {} gen m n".format(sys.argv[0]))
            sys.exit(1)
        generatebenchmarkdata(m=sys.argv[2],n=sys.argv[3])
        exit(0)

    if len(sys.argv)!=8:
        print("Usage: {} ra_rard_pa_or_rasip m n mc_lhs_splitlhs_so_or_sg exp noise pstart,pend".format(sys.argv[0]))
        sys.exit(1)

    type = sys.argv[1]
    m = sys.argv[2]
    n = sys.argv[3]
    sample = sys.argv[4]

    exparr = sys.argv[5].split(',')
    if len(exparr) == 0:
        print("please specify comma saperated exp#")
        sys.exit(1)

    noise = sys.argv[6]

    pstarendarr = sys.argv[7].split(',')
    if len(pstarendarr) == 0 or len(pstarendarr) % 2 !=0:
        print("please specify comma saperated start and end processes")
        sys.exit(1)


    runall(type=type, sample=sample, exparr=exparr,noise=noise,m=m,n=n,pstarendarr=pstarendarr)
