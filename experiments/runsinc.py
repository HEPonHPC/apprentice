
import apprentice
import numpy as np
import os, sys

def sinc(X,dim):
    ret = 10
    for d in range(dim):
        x = X[d]
        ret *= np.sin(x)/x
    return ret

def runrapp(m,n,ts):
    fname = "f20"

    larr = [10**-6,10**-3]
    uarr = [2*np.pi,4*np.pi]
    lbdesc = {0:"-6",1:"-3"}
    ubdesc = {0:"2pi",1:"4pi"}

    noisestr = ""

    folder = "%s%s_%s/sincrun"%(fname,noisestr,ts)
    if not os.path.exists(folder):
        os.mkdir(folder)

    if not os.path.exists(folder+"/benchmarkdata"):
        os.mkdir(folder+'/benchmarkdata')

    dim = 4
    numlb = 0
    numub = 1
    fndesc = "%s%s_d%d_lb%s_ub%s"%(fname,noisestr,dim,lbdesc[numlb],ubdesc[numub])
    outfile = "%s/benchmarkdata/%s.csv"%(folder,fndesc)
    fndesc = "%s%s_%s_p%d_q%d_ts%s_d%d_lb%s_ub%s"%(fname,noisestr,ts,m,n, ts, dim,lbdesc[numlb],ubdesc[numub])
    if not os.path.exists(folder+"/"+fndesc):
        os.mkdir(folder+'/'+fndesc)
    if not os.path.exists(folder+"/"+fndesc+'/outra'):
        os.mkdir(folder+'/'+fndesc+'/outra')
    if not os.path.exists(folder+"/"+fndesc+'/log/consolelogra'):
        os.makedirs(folder+'/'+fndesc+'/log/consolelogra',exist_ok = True)
    l = "_p%d_q%d_ts%s.json"%(m,n, ts)
    jsonfile = folder+'/'+fndesc+"/outra/"+fndesc+l
    consolelog = folder+'/'+fndesc+'/log/consolelogra/'+fndesc+"_p"+str(m)+"_q"+str(n)+"_ts"+ts+".log"
    cmd = 'nohup python runnonsiprapp.py %s %s %d %d %s %s >%s 2>&1 &'%(outfile,fndesc,m,n,ts,jsonfile,consolelog)
    print(cmd)
    os.system(cmd)

def runrappsip(m,n,ts):
    fname = "f20"

    larr = [10**-6,10**-3]
    uarr = [2*np.pi,4*np.pi]
    lbdesc = {0:"-6",1:"-3"}
    ubdesc = {0:"2pi",1:"4pi"}
    noisestr = ""

    folder = "%s%s_%s/sincrun"%(fname,noisestr,ts)
    if not os.path.exists(folder):
        os.mkdir(folder)

    dim = 3
    numlb = 0
    numub = 1
    fndesc = "%s%s_d%d_lb%s_ub%s"%(fname,noisestr,dim,lbdesc[numlb],ubdesc[numub])
    outfile = "%s/benchmarkdata/%s.csv"%(folder,fndesc)
    fndesc = "%s%s_%s_p%d_q%d_ts%s_d%d_lb%s_ub%s"%(fname,noisestr,ts,m,n, ts, dim,lbdesc[numlb],ubdesc[numub])
    if not os.path.exists(folder+"/"+fndesc):
        os.mkdir(folder+'/'+fndesc)
    if not os.path.exists(folder+"/"+fndesc+'/out'):
        os.mkdir(folder+'/'+fndesc+'/out')
    if not os.path.exists(folder+"/"+fndesc+'/log/consolelog'):
        os.makedirs(folder+'/'+fndesc+'/log/consolelog',exist_ok = True)
    outfolder = folder+'/'+fndesc
    consolelog = folder+'/'+fndesc+'/log/consolelog/'+fndesc+"_p"+str(m)+"_q"+str(n)+"_ts"+ts+".log"
    cmd = 'nohup python runrappsip.py %s %s %d %d %s %s >%s 2>&1 &'%(outfile,fndesc,m,n,ts,outfolder,consolelog)
    print(cmd)
    os.system(cmd)


def runsinc(m,n,ts):
    fname = "f20"

    seed1 = 54321
    seed2 = 456789
    seed3 = 9876512
    seed4 = 7919820
    seed5 = 10397531
    larr = [10**-6,10**-3]
    uarr = [2*np.pi,4*np.pi]
    lbdesc = {0:"-6",1:"-3"}
    ubdesc = {0:"2pi",1:"4pi"}

    seed = seed1

    noisestr = ""

    folder = "%s%s_%s/sincrun"%(fname,noisestr,ts)
    if not os.path.exists(folder):
        os.mkdir(folder)

    if not os.path.exists(folder+"/benchmarkdata"):
        os.mkdir(folder+'/benchmarkdata')

    np.random.seed(seed)


    for dim in range(2,8):
        for numlb,lb in enumerate(larr):
            for numub,ub in enumerate(uarr):
                Xperdim = ()
                for d in range(dim):
                    Xperdim = Xperdim + (np.random.rand(1000,)*(ub-lb)+lb,)

                X = np.column_stack(Xperdim)
                formatStr = "{0:0%db}"%(dim)
                for d in range(2**dim):
                    binArr = [int(x) for x in formatStr.format(d)[0:]]
                    val = []
                    for i in range(dim):
                        if(binArr[i] == 0):
                            val.append(lb)
                        else:
                            val.append(ub)
                    X[d] = val
                Y = [sinc(x,dim) for x in X]
                Y = np.atleast_2d(np.array(Y))
                fndesc = "%s%s_d%d_lb%s_ub%s"%(fname,noisestr,dim,lbdesc[numlb],ubdesc[numub])
                outfile = "%s/benchmarkdata/%s.csv"%(folder,fndesc)
                np.savetxt(outfile, np.hstack((X,Y.T)), delimiter=",")
                fndesc = "%s%s_%s_p%d_q%d_ts%s_d%d_lb%s_ub%s"%(fname,noisestr,ts,m,n, ts, dim,lbdesc[numlb],ubdesc[numub])
                if not os.path.exists(folder+"/"+fndesc):
                    os.mkdir(folder+'/'+fndesc)
                if not os.path.exists(folder+"/"+fndesc+'/out'):
                    os.mkdir(folder+'/'+fndesc+'/out')
                if not os.path.exists(folder+"/"+fndesc+'/log/consolelog'):
                    os.makedirs(folder+'/'+fndesc+'/log/consolelog',exist_ok = True)
                outfolder = folder+'/'+fndesc
                consolelog = folder+'/'+fndesc+'/log/consolelog/'+fndesc+"_p"+str(m)+"_q"+str(n)+"_ts"+ts+".log"
                cmd = 'nohup python runrappsip.py %s %s %d %d %s %s >%s 2>&1 &'%(outfile,fndesc,m,n,ts,outfolder,consolelog)
                print(cmd)
                os.system(cmd)


def createtestfiles(ts):
    fname = "f20"
    testseed =9999

    noisestr = ""
    larr = [10**-6,10**-3]
    uarr = [2*np.pi,4*np.pi]
    lbdesc = {0:"-6",1:"-3"}
    ubdesc = {0:"2pi",1:"4pi"}

    folder = "%s%s_%s/sincrun"%(fname,noisestr,ts)
    if not os.path.exists(folder):
        os.mkdir(folder)

    if not os.path.exists(folder+"/benchmarkdata"):
        os.mkdir(folder+'/benchmarkdata')

    np.random.seed(testseed)
    for dim in range(2,8):
        for numlb,lb in enumerate(larr):
            for numub,ub in enumerate(uarr):
                Xperdim = ()
                for d in range(dim):
                    Xperdim = Xperdim + (np.random.rand(100000,)*(ub-lb)+lb,)

                X = np.column_stack(Xperdim)
                formatStr = "{0:0%db}"%(dim)
                for d in range(2**dim):
                    binArr = [int(x) for x in formatStr.format(d)[0:]]
                    val = []
                    for i in range(dim):
                        if(binArr[i] == 0):
                            val.append(lb)
                        else:
                            val.append(ub)
                    X[d] = val
                Y = [sinc(x,dim) for x in X]
                Y = np.atleast_2d(np.array(Y))
                fndesc = "%s%s_d%d_lb%s_ub%s"%(fname,noisestr,dim,lbdesc[numlb],ubdesc[numub])
                outfile = "%s/benchmarkdata/%s_test.csv"%(folder,fndesc)
                np.savetxt(outfile, np.hstack((X,Y.T)), delimiter=",")


def findroots(m,n,ts):
    def fn111(P,app,c1,c2,printnumer):
        x=P[0]
        y=c1
        z=c2
        # return (10.23524024435259
        # -0.6572458938123837*z
        # +2.7128389882209905*y
        # +1.5441080774930798*x
        # -6.6538979531196025*z**2
        # -0.5141146650542658*y*z
        # + 4.945747292016897y**2
        # +1.6858172869147054*x*z
        # -1.2715117330864414*x*y
        # -5.083875228303005*x**2
        # +1.400580994364511*z**3
        # -12.773709198711515*y*z**2
        # + 4.554457711154156*y**2*z
        # +6.771055260547366*y**3
        # +5.665267474305361*x*z**2
        # -1.2715117330864403*x*y*z
        # +3.4491950497063897*x*y**2
        # -2.854578456759403*x**2*z
        # +2.7757002848888925*x**2*y
        # -8.972753314590127*x**3)
        # print(x,y,z)
        X = app._scaler.scale(np.array([x,y,z]))
        if(printnumer==1):
            print("numer=%f"%(app.numer(X)))

        return app.denom(X)

    # from scipy import optimize
    # sol = optimize.root(fn, [0, 0,0],method='hybr')
    # print(sol)
    fname = "f20"

    larr = [10**-6,10**-3]
    uarr = [2*np.pi,4*np.pi]
    lbdesc = {0:"-6",1:"-3"}
    ubdesc = {0:"2pi",1:"4pi"}
    m=2
    n=3
    ts="2x"

    noisestr = ""

    folder = "%s%s_%s/sincrun"%(fname,noisestr,ts)
    dim = 3
    numlb = 0
    numub = 1
    fndesc = "%s%s_%s_p%d_q%d_ts%s_d%d_lb%s_ub%s"%(fname,noisestr,ts,m,n, ts, dim,lbdesc[numlb],ubdesc[numub])
    l = "_p%d_q%d_ts%s.json"%(m,n, ts)
    jsonfile = folder+'/'+fndesc+"/out/"+fndesc+l
    if not os.path.exists(jsonfile):
        print("%s not found"%(jsonfile))
        exit(1)
    import json
    if jsonfile:
        with open(jsonfile, 'r') as fn:
            datastore = json.load(fn)
    datastore['pcoeff'] = datastore['iterationinfo'][0]['pcoeff']
    datastore['qcoeff'] = datastore['iterationinfo'][0]['qcoeff']
    from apprentice import RationalApproximationSIP
    rappsip = RationalApproximationSIP(datastore)

    X = np.linspace(larr[numlb], uarr[numub], num=1000)
    Y = np.linspace(larr[numlb], uarr[numub], num=1000)
    Z = np.linspace(larr[numlb], uarr[numub], num=1000)
    from scipy.optimize import fsolve
    x0 = 0
    for y in Y:
        for z in Z:
            x = fsolve(fn111, x0, args=(rappsip,y,z,0))
            # print(x)
            f=fn111(x,rappsip,y,z,0)
            if(f==0 and x[0]>larr[numlb] and x[0]<uarr[numub]):
                print(x,y,z,fn111(x,rappsip,y,z,1))




if __name__ == "__main__":

    if len(sys.argv)!=4:
        print("Usage: {} m n ts".format(sys.argv[0]))
        sys.exit(1)


    # runsinc(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3])
    # createtestfiles(sys.argv[3])
    # runrapp(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3])
    # runrappsip(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3])
    findroots(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3])
