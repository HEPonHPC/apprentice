import numpy as np

def getData(X_train, fn, noisepct):
    """
    TODO use eval or something to make this less noisy
    """
    from apprentice import testData
    if fn==1:
        Y_train = [testData.f1(x) for x in X_train]
    elif fn==2:
        Y_train = [testData.f2(x) for x in X_train]
    elif fn==3:
        Y_train = [testData.f3(x) for x in X_train]
    elif fn==4:
        Y_train = [testData.f4(x) for x in X_train]
    elif fn==5:
        Y_train = [testData.f5(x) for x in X_train]
    elif fn==6:
        Y_train = [testData.f6(x) for x in X_train]
    elif fn==7:
        Y_train = [testData.f7(x) for x in X_train]
    elif fn==8:
        Y_train = [testData.f8(x) for x in X_train]
    elif fn==9:
        Y_train = [testData.f9(x) for x in X_train]
    elif fn==10:
        Y_train = [testData.f10(x) for x in X_train]
    elif fn==12:
        Y_train = [testData.f12(x) for x in X_train]
    elif fn==13:
        Y_train = [testData.f13(x) for x in X_train]
    elif fn==14:
        Y_train = [testData.f14(x) for x in X_train]
    elif fn==15:
        Y_train = [testData.f15(x) for x in X_train]
    elif fn==16:
        Y_train = [testData.f16(x) for x in X_train]
    elif fn==17:
        Y_train = [testData.f17(x) for x in X_train]
    elif fn==18:
        Y_train = [testData.f18(x) for x in X_train]
    elif fn==19:
        Y_train = [testData.f19(x) for x in X_train]
    elif fn==20:
        Y_train = [testData.f20(x) for x in X_train]
    elif fn==21:
        Y_train = [testData.f21(x) for x in X_train]
    elif fn==22:
        Y_train = [testData.f22(x) for x in X_train]
    elif fn==23:
        Y_train = [testData.f23(x) for x in X_train]
    elif fn==24:
        Y_train = [testData.f24(x) for x in X_train]
    else:
        raise Exception("function {} not implemented, exiting".format(fn))

    stdnormalnoise = np.zeros(shape = (len(Y_train)), dtype =np.float64)
    for i in range(len(Y_train)):
        stdnormalnoise[i] = np.random.normal(0,1)

    return np.atleast_2d(np.array(Y_train)*(1+ noisepct*stdnormalnoise))

def mkData(function,seed,npoints,dim,minarr,maxarr,corners,noisepct,outfile):
    np.random.seed(seed)

    Xperdim = ()
    for d in range(dim):
        Xperdim = Xperdim + (np.random.rand(npoints,)*(maxarr[d]-minarr[d])+minarr[d],) # Coordinates are generated in [MIN,MAX]

    X = np.column_stack(Xperdim)

    if corners:
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

    if(noisepct < 0 or noisepct >1):
        raise Exception("Percentage of standard normal nose should be between 0 and 1 and not %f"%(noisepct))
    Y = getData(X, fn=function, noisepct=noisepct)

    np.savetxt(outfile, np.hstack((X,Y.T)), delimiter=",")


if __name__ == "__main__":
    import optparse, os, sys
    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-o", dest="OUTFILE", default="test.dat", help="Output file name (default: %default)")
    op.add_option("-n", dest="NPOINTS", default=100, type=int,  help="Number of data points to generate (default: %default)")
    op.add_option("-f", dest="FUNCTION", default=1, type=int,  help="Test function number [1...6] (default: %default)")
    op.add_option("-r", dest="NOISEPCT", default=0, type=float,  help="Percentage of standard normal noise to use (between 0 and 1, i.e., not in %) (default: %default)")
    op.add_option("--xmin", dest="MIN", default=-1, type=float,  help="Minimum X (default: %default)")
    op.add_option("--xmax", dest="MAX", default=1, type=float,  help="Maximum X (default: %default)")
    op.add_option("-s", "--seed", dest="SEED", default=54321, type=int,  help="Random seed (default: %default)")
    op.add_option("-c", "--corners", dest="CORNERS", default=False, action="store_true",  help="Include corners (default: %default)")
    op.add_option("-d", dest="DIM", default=2, type=int,  help="Dimension (default: %default)")
    opts, args = op.parse_args()
    minarr = []
    maxarr = []
    for d in range(opts.DIM):
        minarr.append(opts.MIN)
        maxarr.append(opts.MAX)

    # Special call for f17
    if(opts.FUNCTION ==17):
        mkData(opts.FUNCTION,opts.SEED,opts.NPOINTS,opts.DIM,[80,5,90],[100,10,93],opts.CORNERS,opts.NOISEPCT,opts.OUTFILE)
    elif(opts.FUNCTION ==20):
        a = 10**-6
        b = 4*np.pi
        mkData(opts.FUNCTION,opts.SEED,opts.NPOINTS,opts.DIM,[a,a,a,a],[b,b,b,b],opts.CORNERS,opts.NOISEPCT,opts.OUTFILE)
    elif(opts.FUNCTION ==21):
        a = 10**-6
        b = 4*np.pi
        mkData(opts.FUNCTION,opts.SEED,opts.NPOINTS,opts.DIM,[a,a],[b,b],opts.CORNERS,opts.NOISEPCT,opts.OUTFILE)
    else:
        mkData(opts.FUNCTION,opts.SEED,opts.NPOINTS,opts.DIM,minarr,maxarr,opts.CORNERS,opts.NOISEPCT,opts.OUTFILE)
