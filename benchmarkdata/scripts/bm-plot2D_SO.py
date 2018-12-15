import numpy as np
import apprentice as app

def plotPorQResidual(dir_in, dir_out, fno=1, datatype="train", f_test="", norm=1):
    noiseStr = ""
    if "noise_0.1" in dir_in:
        noiseStr = "_noise_0.1"
    elif "noise_0.5" in dir_in:
        noiseStr = "_noise_0.5"

    X_test = []
    Y_test = []
    testSize = 0
    if(datatype == "test"):
        X_test, Y_test = app.readData(f_test)
        testSize = len(X_test[:,0])

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
        error_m_n_1x = np.zeros(shape=(4,4))
        error_m_n_2x = np.zeros(shape=(4,4))
        error_m_n_1k = np.zeros(shape=(4,4))

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
            dataTypeStr = ""
            if(datatype == "train"):
                dataTypeStr = "Training LSQ"
            elif(datatype == "test"):
                dataTypeStr = "Testing LSQ"
            f.suptitle(dataTypeStr+". FixedPenalty = "+pOrq0 + ". noOfPoints = "+npoints+". f"+str(fno)+": "+getFunctionLatex(fno), fontsize = 28)

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
                        if(pOrqN0 == "ppen"):
                            yKey = flipBitsUpto(yKey,pdeg)
                        elif(pOrqN0 == "qpen"):
                            yKey = flipBitsUpto(yKey,qdeg)
                        if(datatype == "train"):
                            import json
                            if jsonfn:
                                with open(jsonfn, 'r') as fn:
                                    datastore = json.load(fn)
                            iterationInfo = datastore["iterationInfo"]
                            # print(yOrder[index])
                            leastSq[yKey] = iterationInfo[len(iterationInfo)-1]['LeastSqObj']
                        elif(datatype=="test"):
                            R = app.readApprentice(jsonfn)
                            if norm == 1: res = [abs(R(x)-Y_test[num]) for num, x in enumerate(X_test)]
                            if norm == 2: res = [(R(x)-Y_test[num])**2 for num, x in enumerate(X_test)]
                            leastSq[yKey] = sum(res)/testSize

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
                    leastSqX = np.array([])
                    for key in leastSq:
                        nonZeroY = np.append(nonZeroY,calcNumberOfNonZeroCoeff(key))
                        keyArr = np.append(keyArr,key)
                        leastSqX = np.append(leastSqX,leastSq[key])

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

                    if(npoints == "1x"):
                        error_m_n_1x[pdeg-1][qdeg-1] = X[minIndex]
                    if(npoints == "2x"):
                        error_m_n_2x[pdeg-1][qdeg-1] = X[minIndex]
                    if(npoints == "1k"):
                        error_m_n_1k[pdeg-1][qdeg-1] = X[minIndex]

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
                    if(datatype == "train"):
                        ax.set_xlabel("$log_{10}\\left(\\left|\\left|f - \\frac{p^m}{q^n}\\right|\\right|_2^2\\right)$", fontsize = 15)
                    elif(datatype == "test"):
                        ax.set_xlabel("$log_{10}\\left(\\frac{\\left|\\left|f - \\frac{p^m}{q^n}\\right|\\right|_%i}{%i}\\right)$"%(norm,testSize), fontsize = 15)

            f_out = ""
            if(datatype == "train"):
                f_out = dir_out+"/f"+str(fno)+noiseStr+"_n"+npoints+"_nz-"+pOrqN0+"_training.png"
            elif(datatype == "test"):
                f_out = dir_out+"/f"+str(fno)+noiseStr+"_n"+npoints+"_nz-"+pOrqN0+"_testing.png"
            plt.savefig(f_out)
        # print(np.c_[error_m_n_1x ,error_m_n_2x, error_m_n_1k])
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        mpl.rc('text', usetex = True)
        mpl.rc('font', family = 'serif', size=12)
        mpl.style.use("ggplot")
        cmapname   = 'viridis'
        X,Y = np.meshgrid(range(1,5), range(1,5))
        f, axarr = plt.subplots(3, sharex=True, sharey=True, figsize=(15,15))
        dataTypeStr = ""
        if(datatype == "train"):
            dataTypeStr = "Training LSQ"
        elif(datatype == "test"):
            dataTypeStr = "Testing LSQ"
        f.suptitle(dataTypeStr+". FixedPenalty = "+pOrq0+". f"+str(fno)+": "+getFunctionLatex(fno), fontsize = 28)
        markersize = 1000
        vmin = -6
        vmax = 4

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
        if(datatype == "train"):
            b.set_label("$log_{10}\\left(\\left|\\left|f - \\frac{p^m}{q^n}\\right|\\right|_2^2\\right)$", fontsize = 28)
        elif(datatype == "test"):
            b.set_label("$log_{10}\\left(\\frac{\\left|\\left|f - \\frac{p^m}{q^n}\\right|\\right|_%i}{%i}\\right)$"%(norm,testSize), fontsize = 28)


        f_out = ""
        if(datatype == "train"):
            f_out = dir_out+"/f"+str(fno)+noiseStr+"_nz-"+pOrqN0+"_training.png"
        if(datatype == "test"):
            f_out = dir_out+"/f"+str(fno)+noiseStr+"_nz-"+pOrqN0+"_testing.png"
        plt.savefig(f_out)

# #########################################################


def plotPandQResidual(dir_in, dir_out, fno=1, datatype="train", f_test="", norm=1):
    noiseStr = ""
    if "noise_0.1" in dir_in:
        noiseStr = "_noise_0.1"
    elif "noise_0.5" in dir_in:
        noiseStr = "_noise_0.5"

    X_test = []
    Y_test = []
    testSize = 0
    if(datatype == "test"):
        X_test, Y_test = app.readData(f_test)
        testSize = len(X_test[:,0])

    noPointsScale = ["1x", "2x", "1k"]

    # Only for (p,q) = (4,4) for now
    pdeg = 4
    qdeg = 4

    pyOrder = ["1111",   # 0     1
            "1111",     # 1     2.1
            "0111",     # 2     2.2
            "0111",     # 3     3
            "1111",     # 4     4.1
            "0011",     # 5     4.2
            "0111",     # 6     5.1
            "0011",     # 7     5.2
            "1111",     # 8     6.1
            "0001",     # 9     6.2
            "0011",     # 10    7
            "0111",     # 11    8.1
            "0001",     # 12    8.2
            "0011",     # 13    9.1
            "0001",     # 14    9.2
            "0000",     # 15    9.3
            "1111",     # 16    9.4
            "0000",     # 17    10.1
            "0111",     # 18    10.2
            "0001",     # 19    11
            "0011",     # 20    12.1
            "0000",     # 21    12.2
            "0001",     # 22    13.1
            "0000"]     # 23    13.2

    qyOrder = ["1111",   # 0     1
            "0111",     # 1     2.1
            "1111",     # 2     2.2
            "0111",     # 3     3
            "0011",     # 4     4.1
            "1111",     # 5     4.2
            "0011",     # 6     5.1
            "0111",     # 7     5.2
            "0001",     # 8     6.1
            "1111",     # 9     6.2
            "0011",     # 10    7
            "0001",     # 11    8.1
            "0111",     # 12    8.2
            "0001",     # 13    9.1
            "0011",     # 14    9.2
            "1111",     # 15    9.3
            "0000",     # 16    9.4
            "0111",     # 17    10.1
            "0000",     # 18    10.2
            "0001",     # 19    11
            "0000",     # 20    12.1
            "0011",     # 21    12.2
            "0000",     # 22    13.1
            "0001"]     # 23    13.2

    # print(np.c_[pyOrder,qyOrder])

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rc('text', usetex = True)
    mpl.rc('font', family = 'serif', size=12)
    mpl.style.use("ggplot")
    cmapname   = 'viridis'

    f, axarr = plt.subplots(3,sharex=True, figsize=(15,15))
    dataTypeStr = ""
    if(datatype == "train"):
        dataTypeStr = "Training LSQ"
    elif(datatype == "test"):
        dataTypeStr = "Testing LSQ"
    f.suptitle(dataTypeStr+". m = 4, n = 4. f"+str(fno)+": "+getFunctionLatex(fno), fontsize = 28)
    for npoints in noPointsScale:
        leastSq ={}
        index = 0
        while index < len(pyOrder):
            pyKey = pyOrder[index]
            qyKey = qyOrder[index]

            penStr = "ppen"+pyKey+"_"+"qpen"+qyKey
            jsonfn = dir_in+"/f"+str(fno)+noiseStr+"_p"+str(pdeg)+"_q"+str(qdeg)+"_n"+npoints+"_RA_SIP_LSQSO_Qxge1_Xsample_s10_"+penStr+".json"
            pyKey = flipBitsUpto(pyKey,pdeg)
            qyKey = flipBitsUpto(qyKey,qdeg)
            if(datatype == "train"):
                import json
                if jsonfn:
                    with open(jsonfn, 'r') as fn:
                        datastore = json.load(fn)
                iterationInfo = datastore["iterationInfo"]
                # print(yOrder[index])
                leastSq[(pyKey,qyKey)] = iterationInfo[len(iterationInfo)-1]['LeastSqObj']
            elif(datatype=="test"):
                R = app.readApprentice(jsonfn)
                if norm == 1: res = [abs(R(x)-Y_test[num]) for num, x in enumerate(X_test)]
                if norm == 2: res = [(R(x)-Y_test[num])**2 for num, x in enumerate(X_test)]
                leastSq[(pyKey,qyKey)] = sum(res)/testSize
            index += 1
        # print(leastSq)

        X = [];
        Y = [];
# OPTIONS
        plotOrder = 2
        if(plotOrder == 1):
            # IN the order of Yorder
            index = 0
            while index < len(pyOrder):
                pyKey = pyOrder[index]
                qyKey = qyOrder[index]
                pyKey = flipBitsUpto(pyKey,pdeg)
                qyKey = flipBitsUpto(qyKey,qdeg)
                if((pyKey,qyKey) in leastSq):
                    X.append(leastSq[(pyKey,qyKey)])
                    no_nz_coeff = (calcNumberOfNonZeroCoeff(pyKey)) \
                                    + (calcNumberOfNonZeroCoeff(qyKey))
                    Y.append(pyKey+"-"+qyKey+"("+str(no_nz_coeff)+")")
                index += 1
        elif(plotOrder == 2):
            # IN decreasing order of leastSq
            for key, value in sorted(leastSq.iteritems(), key=lambda (k,v): (v,k), reverse=True):
                index = 0
                no_nz_coeff = (calcNumberOfNonZeroCoeff(key[0])) \
                                + (calcNumberOfNonZeroCoeff(key[1]))
                Y.append(key[0]+"-"+key[1]+"("+str(no_nz_coeff)+")")
                X.append(value)

        # print(np.c_[X,Y])
        nonZeroY = np.array([])
        pkeyArr = np.array([])
        qkeyArr = np.array([])
        leastSqX = np.array([])
        for key in leastSq:
            no_nz_coeff = (calcNumberOfNonZeroCoeff(key[0])) \
                            + (calcNumberOfNonZeroCoeff(key[1]))
            nonZeroY = np.append(nonZeroY,no_nz_coeff)
            pkeyArr = np.append(pkeyArr,key[0])
            qkeyArr = np.append(qkeyArr,key[1])
            leastSqX = np.append(leastSqX,leastSq[key])
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
        pminKey = pkeyArr[np.argmin(distance)]
        qminKey = qkeyArr[np.argmin(distance)]
        minIndex = 0
        while minIndex < len(Y):
            no_nz_coeff = (calcNumberOfNonZeroCoeff(pminKey)) \
                            + (calcNumberOfNonZeroCoeff(qminKey))
            if(Y[minIndex] == pminKey+"-"+qminKey+"("+str(no_nz_coeff)+")"):
                break
            minIndex += 1

        # print(np.c_[X,Y])
        # print(np.c_[pkeyArr, qkeyArr, distance, Y, X])
        # print(pminKey,qminKey, minIndex, np.min(distance))
        # exit(1)
        logX = np.ma.log10(X)
        if(npoints == "1x"):
            axarr[0].plot(logX, Y, '-rD', markevery=[minIndex])
            axarr[0].set_title("no of points = 1x")
        if(npoints == "2x"):
            axarr[1].plot(logX, Y, '-rD', markevery=[minIndex])
            axarr[1].set_title("no of points = 2x")
        if(npoints == "1k"):
            axarr[2].plot(logX, Y, '-rD', markevery=[minIndex])
            axarr[2].set_title("no of points = 1000")

    for ax in axarr.flat:
        ax.set(xlim=(-6,4))
        if(ax.is_first_col()):
            ax.set_ylabel('p-q Non Zeros', fontsize = 15)
        if(ax.is_last_row()):
            if(datatype == "train"):
                ax.set_xlabel("$log_{10}\\left(\\left|\\left|f - \\frac{p^m}{q^n}\\right|\\right|_2^2\\right)$", fontsize = 15)
            elif(datatype == "test"):
                ax.set_xlabel("$log_{10}\\left(\\frac{\\left|\\left|f - \\frac{p^m}{q^n}\\right|\\right|_%i}{%i}\\right)$"%(norm,testSize), fontsize = 15)

    f_out = ""
    if(datatype == "train"):
        f_out = dir_out+"/f"+str(fno)+noiseStr+"_p"+str(pdeg)+"_q"+str(qdeg)+"_n"+npoints+"_nz-pandq_training.png"
    elif(datatype == "test"):
        f_out = dir_out+"/f"+str(fno)+noiseStr+"_p"+str(pdeg)+"_q"+str(qdeg)+"_n"+npoints+"_nz-pandq_testing.png"
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
    op.add_option("-t", dest="TEST", default="../f1_test.txt", help="Test File name (default: %default)")
    op.add_option("-o", dest="OUTDIR", help="Output directory")
    op.add_option("-n", dest="NORM", default=1, type=int, help="Error norm (default: %default)")
    op.add_option("-p", dest="PLOT", default="plotPorQResidual", help="Plot Type: plotPorQResidual or plotPandQResidual (default: %default)")
    op.add_option("-f", dest="FNO", default=1, type=int, help="Function no (default: %default)")
    op.add_option("-d", dest="DATATYPE", default="train", help="Data Type (train or test) (default: %default)")
    op.add_option("-i", dest="INDIR", help="Input directory")
    opts, args = op.parse_args()



    if opts.PLOT == "plotPorQResidual":
        plotPorQResidual(opts.INDIR, opts.OUTDIR, opts.FNO, opts.DATATYPE, opts.TEST, opts.NORM)
    elif opts.PLOT == "plotPandQResidual":
        plotPandQResidual(opts.INDIR, opts.OUTDIR, opts.FNO, opts.DATATYPE, opts.TEST, opts.NORM)
    else:
        raise Exception("plot type unknown")
