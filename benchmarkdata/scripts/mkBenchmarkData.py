import numpy as np

def getData(X_train, fn, noise):
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
    else:
        raise Exception("function {} not implemented, exiting".format(fn))

    return np.atleast_2d(np.array(Y_train)*(1+ np.random.normal(0,noise)))

if __name__ == "__main__":
    import optparse, os, sys
    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-o", dest="OUTFILE", default="test.dat", help="Output file name (default: %default)")
    op.add_option("-n", dest="NPOINTS", default=100, type=int,  help="Number of data points to generate (default: %default)")
    op.add_option("-f", dest="FUNCTION", default=1, type=int,  help="Test function number [1...6] (default: %default)")
    op.add_option("-r", dest="RANDOM", default=0, type=float,  help="Random noise in pct (default: %default)")
    op.add_option("--xmin", dest="MIN", default=-1, type=float,  help="Minimum X (default: %default)")
    op.add_option("--xmax", dest="MAX", default=1, type=float,  help="Maximum X (default: %default)")
    op.add_option("-s", "--seed", dest="SEED", default=54321, type=int,  help="Random seed (default: %default)")
    op.add_option("-c", "--corners", dest="CORNERS", default=False, action="store_true",  help="Include corners (default: %default)")
    op.add_option("-d", dest="DIM", default=2, type=int,  help="Dimension (default: %default)")
    opts, args = op.parse_args()

    np.random.seed(opts.SEED)

    X = np.random.rand(opts.NPOINTS, opts.DIM)*(opts.MAX-opts.MIN)+opts.MIN # Coordinates are generated in [MIN,MAX]

    if opts.CORNERS:
        min = []
        max = []
        for i in range(opts.DIM):
            min.append(opts.MIN)
            max.append(opts.MAX)
        # Include the corners
        X[0] = min
        X[1] = min
        X[2] = max
        X[3] = max

    Y = getData(X, fn=opts.FUNCTION, noise=opts.RANDOM)

    np.savetxt(opts.OUTFILE, np.hstack((X,Y.T)), delimiter=",")
