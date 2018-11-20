import numpy as np
import apprentice as app

def plotError(f_rapp, f_test, f_out, norm=1):
    R = app.readApprentice(f_rapp)
    X_test, Y_test = app.readData(f_test)
    if norm == 1: error = [abs(R(x)-Y_test[num]) for num, x in enumerate(X_test)]
    if norm == 2: error = [(R(x)-Y_test[num])**2 for num, x in enumerate(X_test)]

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rc('text', usetex = True)
    mpl.rc('font', family = 'serif', size=12)
    mpl.style.use("ggplot")
    cmapname   = 'viridis'
    plt.clf()

    plt.scatter(X_test[:,0], X_test[:,1], marker = '.', c = np.log10(error), cmap = cmapname, alpha = 0.8)
    plt.vlines(-1, ymin=-1, ymax=1, linestyle="dashed")
    plt.vlines( 1, ymin=-1, ymax=1, linestyle="dashed")
    plt.hlines(-1, xmin=-1, xmax=1, linestyle="dashed")
    plt.hlines( 1, xmin=-1, xmax=1, linestyle="dashed")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.ylim((-1.5,1.5))
    plt.xlim((-1.5,1.5))
    b=plt.colorbar()
    b.set_label("$\log_{10}\left|f - \\frac{p^{(%i)}}{q^{(%i)}}\\right|_%i$"%(R.m, R.n, norm))
    plt.savefig(f_out)



if __name__=="__main__":
    import optparse, os, sys
    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-o", dest="OUTFILE", default="plot.pdf", help="Output file name (default: %default)")
    op.add_option("-n", dest="NORM", default=1, type=int, help="Error norm (default: %default)")
    opts, args = op.parse_args()

    plotError(args[0],  args[1], opts.OUTFILE, opts.NORM)
