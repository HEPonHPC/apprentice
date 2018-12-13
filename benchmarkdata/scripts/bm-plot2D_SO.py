import numpy as np
import apprentice as app

    # def plotResidualMap(f_rapp, f_test, f_out, norm=1, fno=1):
    #     R = app.readApprentice(f_rapp)
    #     X_test, Y_test = app.readData(f_test)
    #     if norm == 1: error = [abs(R(x)-Y_test[num]) for num, x in enumerate(X_test)]
    #     if norm == 2: error = [(R(x)-Y_test[num])**2 for num, x in enumerate(X_test)]
    #
    #     import matplotlib as mpl
    #     import matplotlib.pyplot as plt
    #     mpl.rc('text', usetex = True)
    #     mpl.rc('font', family = 'serif', size=12)
    #     mpl.style.use("ggplot")
    #     cmapname   = 'viridis'
    #     plt.clf()
    #
    #     plt.scatter(X_test[:,0], X_test[:,1], marker = '.', c = np.ma.log10(error), cmap = cmapname, alpha = 0.8)
    #     plt.vlines(-1, ymin=-1, ymax=1, linestyle="dashed")
    #     plt.vlines( 1, ymin=-1, ymax=1, linestyle="dashed")
    #     plt.hlines(-1, xmin=-1, xmax=1, linestyle="dashed")
    #     plt.hlines( 1, xmin=-1, xmax=1, linestyle="dashed")
    #     plt.xlabel("$x$")
    #     plt.ylabel("$y$")
    #     plt.ylim((-1.5,1.5))
    #     plt.xlim((-1.5,1.5))
    #     b=plt.colorbar()
    #     b.set_label("$\log_{10}\left|f - \\frac{p^{(%i)}}{q^{(%i)}}\\right|_%i$"%(R.m, R.n, norm))
    #     plt.title(getFunctionLatex(fno))
    #     plt.savefig(f_out)

def plotPorQResidual(dir_in, dir_out, fno=1):
    noiseStr = ""
    if "noise_0.1" in dir_in:
        noiseStr = "_noise_0.1"
    elif "noise_0.5" in dir_in:
        noiseStr = "_noise_0.5"

    # print(fno);
    # print(noiseStr)
    # print(dir_in)
    # print(dir_out)
    # exit(1)

    porqOpt = ["ppen", "qpen"]
    noPointsScale = ["1x", "2x", "1k"]

    # # Static for now
    # pOrq0 = porqOpt[0]
    # pOrqN0 = "qpen"
    # npoints = "1x"


    for pOrq0 in porqOpt:
        if(pOrq0 == "ppen"):
            pOrqN0 = "qpen"
        else:
            pOrqN0 = "ppen"

        for npoints in noPointsScale:

            import matplotlib as mpl
            import matplotlib.pyplot as plt
            mpl.rc('text', usetex = True)
            mpl.rc('font', family = 'serif', size=12)
            mpl.style.use("ggplot")
            cmapname   = 'viridis'

            f, axarr = plt.subplots(4, 4,sharex=True, figsize=(20,20))
            f.suptitle("Training LSQ. FixedPenalty = "+pOrq0 + ". noOfPoints = "+npoints+". f"+str(fno)+": "+getFunctionLatex(fno), fontsize = 28)

            #
            yOrder = ["1111",   # 0     1
                    "0111",     # 1     2
                    "1011",     # 2     3
                    "1101",     # 3     4
                    "1110",     # 4     5.1
                    "0011",     # 5     5.2
                    "0101",     # 6     6
                    "0110",     # 7     7.1
                    "1001",     # 8     7.2
                    "1010",     # 9     8
                    "1100",     # 10    9.1
                    "0001",     # 11    9.2
                    "0010",     # 12    10
                    "0100",     # 13    11
                    "1000",     # 14    12
                    "0000"]     # 15    13

            for pdeg in range(1,5):
                for qdeg in range(1,5):
                    leastSq ={}
                    index = 0

                    while index < len(yOrder):
                        yKey = yOrder[index]
                        yAct = (yKey)[::-1]

                        penStr = ""
                        if(pOrqN0 == "ppen"):
                            if(int(yAct, 2) >= 2**pdeg):
                                index += 1
                                continue
                            penStr = pOrqN0+yKey+"_"+pOrq0+"0000"
                        else:
                            if(int(yAct, 2) >= 2**qdeg):
                                index += 1
                                continue
                            penStr = pOrq0+"0000_"+pOrqN0+yKey
                        jsonfn = dir_in+"/f"+str(fno)+noiseStr+"_p"+str(pdeg)+"_q"+str(qdeg)+"_n"+npoints+"_RA_SIP_LSQSO_Qxge1_Xsample_s10_"+penStr+".json"
                        # print(jsonfn)
                        import json
                        if jsonfn:
                            with open(jsonfn, 'r') as fn:
                                datastore = json.load(fn)
                        iterationInfo = datastore["iterationInfo"]
                        # print(yOrder[index])
                        if(pOrqN0 == "ppen"):
                            yKey = flipBitsUpto(yKey,pdeg)
                        elif(pOrqN0 == "qpen"):
                            yKey = flipBitsUpto(yKey,qdeg)
                        leastSq[yKey] = iterationInfo[len(iterationInfo)-1]['LeastSqObj']
                        index += 1
                    # print(leastSq)

                    X = [];
                    Y = [];
# OPTIONS
                    plotOrder = 2
                    if(plotOrder == 1):
                        # IN the order of Yorder
                        index = 0
                        while index < len(yOrder):
                            yKey = yOrder[index]
                            if(pOrqN0 == "ppen"):
                                yKey = flipBitsUpto(yKey,pdeg)
                            elif(pOrqN0 == "qpen"):
                                yKey = flipBitsUpto(yKey,qdeg)
                            if(yKey in leastSq):
                                X.append(leastSq[yKey])
                                Y.append(yKey+"("+str(calcNumberOfNonZeroCoeff(yKey))+")")
                            index += 1
                    elif(plotOrder == 2):
                        # IN decreasing order of leastSq
                        for key, value in sorted(leastSq.iteritems(), key=lambda (k,v): (v,k), reverse=True):
                            index = 0
                            Y.append(key+"("+str(calcNumberOfNonZeroCoeff(key))+")")
                            X.append(value)

                    # print ("=================",pdeg,qdeg,"=============")


                    nonZeroY = np.array([])
                    keyArr = np.array([])
                    for key in leastSq:
                        nonZeroY = np.append(nonZeroY,calcNumberOfNonZeroCoeff(key))
                        keyArr = np.append(keyArr,key)
                    leastSqX = np.array(leastSq.values())

# OPTIONS
                    scaleOption = 1
                    if(scaleOption == 1):
                        # Y normalized to be between X.min() and X.max()
                        nonZeroYNorm = (leastSqX.max()-leastSqX.min())*((nonZeroY - nonZeroY.min())/(nonZeroY.max()-nonZeroY.min())) + leastSqX.min()
                        leastSqXNorm = leastSqX
                    elif(scaleOption == 2):
                        # X and Y normalized to be between 0 and 1
                        nonZeroYNorm = (nonZeroY - nonZeroY.min())/(nonZeroY.max()-nonZeroY.min())
                        leastSqXNorm = (leastSqX - leastSqX.min())/(leastSqX.max()-leastSqX.min())

                    distance = []
                    distance = np.sqrt(np.sum([np.square(nonZeroYNorm),np.square(leastSqXNorm)],0))
                    minKey = keyArr[np.argmin(distance)]
                    minIndex = 0
                    while minIndex < len(Y):
                        if(Y[minIndex] == minKey+"("+str(calcNumberOfNonZeroCoeff(minKey))+")"):
                            break
                        minIndex += 1


                    logX = np.ma.log10(X)
                    if(pOrqN0 == "ppen"):
                        axarr[pdeg-1][qdeg-1].plot(logX, Y, '-rD', markevery=[minIndex])
                        axarr[pdeg-1][qdeg-1].set_title("p = "+str(pdeg)+"; q = "+str(qdeg))
                    else:
                        axarr[qdeg-1][pdeg-1].plot(logX, Y, '-rD', markevery=[minIndex])
                        axarr[qdeg-1][pdeg-1].set_title("p = "+str(pdeg)+"; q = "+str(qdeg))

            for ax in axarr.flat:
                ax.set(xlim=(-6,4))
                if(ax.is_first_col()):
                    ax.set_ylabel(pOrqN0[0] + ' Non Zeros', fontsize = 15)
                if(ax.is_last_row()):
                    ax.set_xlabel("$log_{10}\\left(\\left|\\left|f - \\frac{p^m}{q^n}\\right|\\right|_2^2\\right)$", fontsize = 15)

            f_out = dir_out+"/f"+str(fno)+noiseStr+"_n"+npoints+"_nz-"+pOrqN0+"_training.png"
            plt.savefig(f_out)


def flipBitsUpto(bitStr, upto):
    bitL = list(bitStr)
    index = 0
    while index < len(bitL):
        if(index < upto and bitL[index] == "1"):
            bitL[index] = "0"
        elif(index < upto and bitL[index] == "0"):
            bitL[index] = "1"
        else: break
        index += 1
    retStr = "".join(bitL)
    return retStr

#2D only 1100 = 1 + 2 + 3 = 6
def calcNumberOfNonZeroCoeff(bitStr):
    bitL = list(bitStr)
    index = 0
    coeffs = [2,3,4,5]
    sum = 1
    while index < len(bitL):
        sum += int(bitL[index]) * coeffs[index]
        index += 1
    return sum










#     X_test, Y_test = app.readData(f_test)
#     testSize = len(X_test[:,0])
#     # error that maps average error to m,n on x and y axis respectively
#     import numpy as np
#     # error_m_n_all = np.zeros(shape=(4,4))
#     error_m_n_1x = np.zeros(shape=(4,4))
#     error_m_n_2x = np.zeros(shape=(4,4))
#     error_m_n_1k = np.zeros(shape=(4,4))
#
#     for i in range(len(f_rapp[0])):
#         # print(f_rapp[0][i])
#         R = app.readApprentice(f_rapp[0][i])
#         if norm == 1: res = [abs(R(x)-Y_test[num]) for num, x in enumerate(X_test)]
#         if norm == 2: res = [(R(x)-Y_test[num])**2 for num, x in enumerate(X_test)]
#         m = R.m
#         n = R.n
#         addTerm = sum(res)/testSize
#         if R.trainingsize == R.M + R.N:
#             error_m_n_1x[m-1][n-1] = error_m_n_1x[m-1][n-1] + addTerm
#         elif R.trainingsize == 2*(R.M + R.N):
#             error_m_n_2x[m-1][n-1] = error_m_n_2x[m-1][n-1] + addTerm
#         elif R.trainingsize == 1000:
#             error_m_n_1k[m-1][n-1] = error_m_n_1k[m-1][n-1] + addTerm
#         else:
#             raise Exception("Something is wrong here. Incorrect training size used")
#         # error_m_n_all[m-1][n-1] = error_m_n_all[m-1][n-1] + addTerm
#
#     import matplotlib as mpl
#     import matplotlib.pyplot as plt
#     mpl.rc('text', usetex = True)
#     mpl.rc('font', family = 'serif', size=12)
#     mpl.style.use("ggplot")
#     cmapname   = 'viridis'
#     X,Y = np.meshgrid(range(1,5), range(1,5))
#
#     f, axarr = plt.subplots(3, sharex=True, sharey=True, figsize=(15,15))
#     f.suptitle("f"+str(fno)+": "+getFunctionLatex(fno), fontsize = 28)
#     markersize = 1000
#     vmin = -4
#     vmax = 2.5
#     v = np.linspace(-6, 3, 1, endpoint=True)
#     # sc1 = axarr[0,0].scatter(X,Y, marker = 's', s=markersize, c = np.ma.log10(error_m_n_all), cmap = cmapname, alpha = 1)
#     # axarr[0,0].set_title('All training size')
#     sc = axarr[0].scatter(X,Y, marker = 's', s=markersize, c = np.ma.log10(error_m_n_1x), cmap = cmapname, vmin=vmin, vmax=vmax, alpha = 1)
#     axarr[0].set_title('Training size = 1x', fontsize = 28)
#     sc = axarr[1].scatter(X,Y, marker = 's', s=markersize, c = np.ma.log10(error_m_n_2x), cmap = cmapname,  vmin=vmin, vmax=vmax, alpha = 1)
#     axarr[1].set_title('Training size = 2x', fontsize = 28)
#     sc = axarr[2].scatter(X,Y, marker = 's', s=markersize, c = np.ma.log10(error_m_n_1k), cmap = cmapname,  vmin=vmin, vmax=vmax, alpha = 1)
#     axarr[2].set_title('Training size = 1000', fontsize = 28)
#
#     for ax in axarr.flat:
#         ax.set(xlim=(0,5),ylim=(0,5))
#         ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
#         ax.tick_params(axis = 'both', which = 'minor', labelsize = 18)
#         ax.set_xlabel('$m$', fontsize = 22)
#         ax.set_ylabel('$n$', fontsize = 22)
#     for ax in axarr.flat:
#         ax.label_outer()
#     b=f.colorbar(sc,ax=axarr.ravel().tolist(), shrink=0.95)
#     b.set_label("Error = $log_{10}\\left(\\frac{\\left|\\left|f - \\frac{p^m}{q^n}\\right|\\right|_%i}{%i}\\right)$"%(norm,testSize), fontsize = 28)
#     # plt.show()
#     plt.savefig(f_out)
#
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
    # op.add_option("-t", dest="TEST", help="Test File name (default: %default)")
    op.add_option("-o", dest="OUTDIR", help="Output directory")
    # op.add_option("-n", dest="NORM", default=1, type=int, help="Error norm (default: %default)")
    # op.add_option("-p", dest="PLOT", default="residualMap", help="Plot Type: residualMap or errorPlot (default: %default)")
    op.add_option("-f", dest="FNO", default=1, type=int, help="Function no (default: %default)")
    # op.add_option("-n", dest="NOISE", default=0.0, type=float, help="Noise level(0.0,0.1,0.5) (default: %default)")
    op.add_option("-i", dest="INDIR", help="Input directory")
    opts, args = op.parse_args()

    plotPorQResidual(opts.INDIR, opts.OUTDIR, opts.FNO)
    # if opts.PLOT == "residualMap":
    #     plotResidualMap(args[0],  opts.TEST, opts.OUTFILE, opts.NORM, opts.FNO)
    # elif opts.PLOT == "errorPlot":
    #     plotError(opts.TEST, opts.OUTFILE, opts.NORM, opts.FNO, args)
    # else:
    #     raise Exception("plot type unknown")
