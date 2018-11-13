import numpy as np


def plotRes(X_test, res, order, fn, strategy):
    m, n=order


    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rc('text', usetex = True)
    mpl.rc('font', family = 'serif', size=12)
    mpl.style.use("ggplot")
    cmapname   = 'viridis'
    plt.clf()

    plt.scatter(X_test[:,0], X_test[:,1], marker = '.', c = np.log10(res), cmap = cmapname, alpha = 0.8)
    plt.vlines(-1, ymin=-1, ymax=1, linestyle="dashed")
    plt.vlines( 1, ymin=-1, ymax=1, linestyle="dashed")
    plt.hlines(-1, xmin=-1, xmax=1, linestyle="dashed")
    plt.hlines( 1, xmin=-1, xmax=1, linestyle="dashed")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.ylim((-1.5,1.5))
    plt.xlim((-1.5,1.5))
    plt.title("Absolute error for $f_{}$ with $m={},~n={}$".format(fn,m,n))
    b=plt.colorbar()
    b.set_label("$\log_{10}$ (Resdiual)")
    plt.savefig('f{}-residual_{}_{}_strat_{}.jpg'.format(fn,m,n,strategy))


def getData(X_train, fn, noise):
    import testData
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
    else:
        raise Exception("function {} not implemented, exiting".format(fn))

    return np.atleast_2d(np.array(Y_train)*(1+ np.random.normal(0,noise)))


if __name__ == "__main__":
    import optparse, os, sys
    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-v", "--debug", dest="DEBUG", action="store_true", default=False, help="Turn on some debug messages")
    op.add_option("-q", "--quiet", dest="QUIET", action="store_true", default=False, help="Turn off messages")
    op.add_option("-o", dest="OUTFILE", default="test.dat", help="Output file name (default: %default)")
    op.add_option("-n", dest="NPOINTS", default=100, type=int,  help="Number of data points to generate (default: %default)")
    op.add_option("-f", dest="FUNCTION", default=1, type=int,  help="Test function number [1...6] (default: %default)")
    op.add_option("-r", dest="RANDOM", default=0, type=float,  help="Random noise in pct (default: %default)")
    op.add_option("-s", "--seed", dest="SEED", default=54321, type=int,  help="Random seed (default: %default)")
    opts, args = op.parse_args()

    np.random.seed(opts.SEED)

    X = np.random.rand(opts.NPOINTS, 2)*2-1 # Coordinates are generated in [-1,1]
    # Include the corners
    X[0] = [-1, -1]
    X[1] = [-1,  1]
    X[2] = [ 1, -1]
    X[3] = [ 1,  1]

    Y = getData(X, fn=opts.FUNCTION, noise=opts.RANDOM)

    np.savetxt(opts.OUTFILE, np.hstack((X,Y.T)), delimiter=",")


    # np stack bla, savetict delim ","# no header

