import numpy as np
import apprentice as app

def plotResidualMap(f_rapp, f_test, f_out, norm=1):
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

    plt.scatter(X_test[:,0], X_test[:,1], marker = '.', c = np.ma.log10(error), cmap = cmapname, alpha = 0.8)
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

def plotError(f_test, f_out, norm=1, *f_rapp):
    X_test, Y_test = app.readData(f_test)
    testSize = len(X_test[:,0])
    # error that maps average error to m,n on x and y axis respectively
    import numpy as np
    error_m_n_all = np.zeros(shape=(4,4))
    error_m_n_1x = np.zeros(shape=(4,4))
    error_m_n_2x = np.zeros(shape=(4,4))
    error_m_n_1k = np.zeros(shape=(4,4))

    for i in range(len(f_rapp[0])):
        # print(f_rapp[0][i])
        R = app.readApprentice(f_rapp[0][i])
        if norm == 1: res = [abs(R(x)-Y_test[num]) for num, x in enumerate(X_test)]
        if norm == 2: res = [(R(x)-Y_test[num])**2 for num, x in enumerate(X_test)]
        m = R.m
        n = R.n
        addTerm = sum(res)/testSize
        if R.trainingsize == R.M + R.N:
            error_m_n_1x[m-1][n-1] = error_m_n_1x[m-1][n-1] + addTerm
        elif R.trainingsize == 2*(R.M + R.N):
            error_m_n_2x[m-1][n-1] = error_m_n_2x[m-1][n-1] + addTerm
        elif R.trainingsize == 1000:
            error_m_n_1k[m-1][n-1] = error_m_n_1k[m-1][n-1] + addTerm
        else:
            raise Exception("Something is wrong here. Incorrect training size used")
        error_m_n_all[m-1][n-1] = error_m_n_all[m-1][n-1] + addTerm

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rc('text', usetex = True)
    mpl.rc('font', family = 'serif', size=12)
    mpl.style.use("ggplot")
    cmapname   = 'viridis'
    X,Y = np.meshgrid(range(1,5), range(1,5))

    f, axarr = plt.subplots(2,2, sharex=True, sharey=True)
    markersize = 400
    sc1 = axarr[0,0].scatter(X,Y, marker = 's', s=markersize, c = error_m_n_all, cmap = cmapname, alpha = 1)
    axarr[0,0].set_title('All training size')
    sc2 = axarr[0,1].scatter(X,Y, marker = 's', s=markersize, c = error_m_n_1x, cmap = cmapname, alpha = 1)
    axarr[0,1].set_title('Training size = 1x')
    sc3 = axarr[1,0].scatter(X,Y, marker = 's', s=markersize, c = error_m_n_2x, cmap = cmapname, alpha = 1)
    axarr[1,0].set_title('Training size = 2x')
    sc4 = axarr[1,1].scatter(X,Y, marker = 's', s=markersize, c = error_m_n_1k, cmap = cmapname, alpha = 1)
    axarr[1,1].set_title('Training size = 1000')

    for ax in axarr.flat:
        ax.set(xlim=(0,5),ylim=(0,5),xlabel='$m$', ylabel='$n$')
    for ax in axarr.flat:
        ax.label_outer()
    b=f.colorbar(sc1,ax=axarr.ravel().tolist(), shrink=0.95)
    b.set_label("Error = $\\frac{\\left|\\left|f - \\frac{p^m}{q^n}\\right|\\right|_%i}{%i}$"%(norm,testSize))
    # plt.show()
    plt.savefig(f_out)


if __name__=="__main__":
    import optparse, os, sys
    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-t", dest="TEST", help="Test File name (default: %default)")
    op.add_option("-o", dest="OUTFILE", default="plot.pdf", help="Output file name (default: %default)")
    op.add_option("-n", dest="NORM", default=1, type=int, help="Error norm (default: %default)")
    op.add_option("-p", dest="PLOT", default="residualMap", help="Plot Type: residualMap or errorPlot (default: %default)")
    opts, args = op.parse_args()


    if opts.PLOT == "residualMap":
        plotResidualMap(args[0],  opts.TEST, opts.OUTFILE, opts.NORM)
    elif opts.PLOT == "errorPlot":
        plotError(opts.TEST, opts.OUTFILE, opts.NORM, args)
    else:
        raise Exception("plot type unknown")
