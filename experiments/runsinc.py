
import apprentice
import numpy as np
import os, sys

def sinc(X,dim):
    ret = 10
    for d in range(dim):
        x = X[d]
        ret *= np.sin(x)/x
    return ret

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
    m=2
    n=3

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




if __name__ == "__main__":

    if len(sys.argv)!=4:
        print("Usage: {} m n ts".format(sys.argv[0]))
        sys.exit(1)


    runsinc(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3])
