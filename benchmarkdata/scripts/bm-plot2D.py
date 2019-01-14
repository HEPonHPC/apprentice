import numpy as np
import apprentice as app
from apprentice import RationalApproximationSIP

def plotResidualMap(f_rapp, f_test, f_out,norm=1, fno=1, type=None):
    X_test, Y_test = app.readData(f_test)
    error = ""
    if(type == None):
        R = app.readApprentice(f_rapp)
        if norm == 1: error = [abs(R(x)-Y_test[num]) for num, x in enumerate(X_test)]
        if norm == 2: error = [(R(x)-Y_test[num])**2 for num, x in enumerate(X_test)]
    elif(type == "rappsip"):
        R = RationalApproximationSIP(f_rapp)
        Y_pred = R(X_test)
        error = (abs(Y_pred-Y_test))
        if norm == 1: error = np.array(abs(Y_pred-Y_test),dtype=np.float64)
        if norm == 2: error = np.array((Y_pred-Y_test)**2,)


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
    plt.title(getFunctionLatex(fno))
    plt.savefig(f_out)

def plotError(f_test, f_out, norm=1, fno=1, *f_rapp):
    X_test, Y_test = app.readData(f_test)
    testSize = len(X_test[:,0])
    # error that maps average error to m,n on x and y axis respectively
    import numpy as np
    # error_m_n_all = np.zeros(shape=(4,4))
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
        # error_m_n_all[m-1][n-1] = error_m_n_all[m-1][n-1] + addTerm

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rc('text', usetex = True)
    mpl.rc('font', family = 'serif', size=12)
    mpl.style.use("ggplot")
    cmapname   = 'viridis'
    X,Y = np.meshgrid(range(1,5), range(1,5))

    f, axarr = plt.subplots(3, sharex=True, sharey=True, figsize=(15,15))
    f.suptitle("f"+str(fno)+": "+getFunctionLatex(fno), fontsize = 28)
    markersize = 1000
    vmin = -4
    vmax = 2.5
    v = np.linspace(-6, 3, 1, endpoint=True)
    # sc1 = axarr[0,0].scatter(X,Y, marker = 's', s=markersize, c = np.ma.log10(error_m_n_all), cmap = cmapname, alpha = 1)
    # axarr[0,0].set_title('All training size')
    sc = axarr[0].scatter(X,Y, marker = 's', s=markersize, c = np.ma.log10(error_m_n_1x), cmap = cmapname, vmin=vmin, vmax=vmax, alpha = 1)
    axarr[0].set_title('Training size = 1x', fontsize = 28)
    sc = axarr[1].scatter(X,Y, marker = 's', s=markersize, c = np.ma.log10(error_m_n_2x), cmap = cmapname,  vmin=vmin, vmax=vmax, alpha = 1)
    axarr[1].set_title('Training size = 2x', fontsize = 28)
    sc = axarr[2].scatter(X,Y, marker = 's', s=markersize, c = np.ma.log10(error_m_n_1k), cmap = cmapname,  vmin=vmin, vmax=vmax, alpha = 1)
    axarr[2].set_title('Training size = 1000', fontsize = 28)

    for ax in axarr.flat:
        ax.set(xlim=(0,5),ylim=(0,5))
        ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
        ax.tick_params(axis = 'both', which = 'minor', labelsize = 18)
        ax.set_xlabel('$m$', fontsize = 22)
        ax.set_ylabel('$n$', fontsize = 22)
    for ax in axarr.flat:
        ax.label_outer()
    b=f.colorbar(sc,ax=axarr.ravel().tolist(), shrink=0.95)
    b.set_label("Error = $log_{10}\\left(\\frac{\\left|\\left|f - \\frac{p^m}{q^n}\\right|\\right|_%i}{%i}\\right)$"%(norm,testSize), fontsize = 28)
    # plt.show()
    plt.savefig(f_out)

def getFunctionLatex(fno):
    if fno == 1:
        return "$\\frac{e^{xy}}{(x^2-1.44)(y^2-1.44)}$"
    elif fno == 2:
        return "$\log(2.25-x^2-y^2)$"
    elif fno == 3:
        return "$\\tanh(5(x-y))$"
    elif fno == 4:
        return "$e^{\\frac{-(x^2+y^2)}{1000}}$"
    elif fno == 5:
        return "$|(x-y)|^3$"
    elif fno == 6:
        return "$\\frac{x^3-xy+y^3}{x^2-y^2+xy^2}$"
    elif fno == 7:
        return "$\\frac{x+y^3}{xy^2+1}$"
    elif fno == 8:
        return "$\\frac{x^2+y^2+x-y-1}{(x-1.1)(y-1.1)}$"
    elif fno == 9:
        return "$\\frac{x^4+y^4+x^2y^2+xy}{(x^2-1.1)(y^2-1.1)}$"
    elif fno == 10:
        return "$\\frac{x_1^2+x_2^2+x_1-x_2+1}{(x_3-1.5)(x_4-1.5)}$"
    else: return "N/A"

if __name__=="__main__":
    import optparse, os, sys
    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-t", dest="TEST", help="Test File name (default: %default)")
    op.add_option("-o", dest="OUTFILE", default="plot.pdf", help="Output file name (default: %default)")
    op.add_option("-n", dest="NORM", default=1, type=int, help="Error norm (default: %default)")
    op.add_option("-p", dest="PLOT", default="residualMap", help="Plot Type: residualMap or errorPlot (default: %default)")
    op.add_option("-f", dest="FNO", default=1, type=int, help="Function no (default: %default)")
    op.add_option("-y", dest="TYPE", default=None, help="Type of analysis (default: %default)")
    opts, args = op.parse_args()


    if opts.PLOT == "residualMap":
        plotResidualMap(args[0],  opts.TEST, opts.OUTFILE, opts.NORM, opts.FNO, opts.TYPE)
    elif opts.PLOT == "errorPlot":
        plotError(opts.TEST, opts.OUTFILE, opts.NORM, opts.FNO, args)
    else:
        raise Exception("plot type unknown")
