import numpy as np
from apprentice import RationalApproximationSIP, RationalApproximationONB, PolynomialApproximation
from apprentice import tools, readData
import matplotlib.ticker as mtick
import os

def getData(X_train, fn, noisepct):
    """
    TODO use eval or something to make this less noisy
    """
    from apprentice import testData
    if fn=="f1":
        Y_train = [testData.f1(x) for x in X_train]
    elif fn=="f2":
        Y_train = [testData.f2(x) for x in X_train]
    elif fn=="f3":
        Y_train = [testData.f3(x) for x in X_train]
    elif fn=="f4":
        Y_train = [testData.f4(x) for x in X_train]
    elif fn=="f5":
        Y_train = [testData.f5(x) for x in X_train]
    elif fn=="f6":
        Y_train = [testData.f6(x) for x in X_train]
    elif fn=="f7":
        Y_train = [testData.f7(x) for x in X_train]
    elif fn=="f8":
        Y_train = [testData.f8(x) for x in X_train]
    elif fn=="f9":
        Y_train = [testData.f9(x) for x in X_train]
    elif fn=="f10":
        Y_train = [testData.f10(x) for x in X_train]
    elif fn=="f12":
        Y_train = [testData.f12(x) for x in X_train]
    elif fn=="f13":
        Y_train = [testData.f13(x) for x in X_train]
    elif fn=="f14":
        Y_train = [testData.f14(x) for x in X_train]
    elif fn=="f15":
        Y_train = [testData.f15(x) for x in X_train]
    elif fn=="f16":
        Y_train = [testData.f16(x) for x in X_train]
    elif fn=="f17":
        Y_train = [testData.f17(x) for x in X_train]
    elif fn=="f18":
        Y_train = [testData.f18(x) for x in X_train]
    elif fn=="f19":
        Y_train = [testData.f19(x) for x in X_train]
    elif fn=="f20":
        Y_train = [testData.f20(x) for x in X_train]
    elif fn=="f21":
        Y_train = [testData.f21(x) for x in X_train]
    elif fn=="f22":
        Y_train = [testData.f22(x) for x in X_train]
    elif fn=="f23":
        Y_train = [testData.f23(x) for x in X_train]
    elif fn=="f24":
        Y_train = [testData.f24(x) for x in X_train]
    else:
        raise Exception("function {} not implemented, exiting".format(fn))
    return Y_train

def getbox(f):
    minbox = []
    maxbox = []
    if(f=="f7"):
        minbox  = [0,0]
        maxbox = [1,1]
    elif(f=="f10" or f=="f19"):
        minbox  = [-1,-1,-1,-1]
        maxbox = [1,1,1,1]
    elif(f=="f17"):
        minbox  = [80,5,90]
        maxbox  = [100,10,93]
    # elif(f=="f17"):
    #     minbox  = [-1,-1,-1]
    #     maxbox  = [1,1,1]
    elif(f=="f18"):
        minbox  = [-0.95,-0.95,-0.95,-0.95]
        maxbox  = [0.95,0.95,0.95,0.95]
    elif(f=="f20"):
        minbox  = [10**-6,10**-6,10**-6,10**-6]
        maxbox  = [4*np.pi,4*np.pi,4*np.pi,4*np.pi]
    elif(f=="f21"):
        minbox  = [10**-6,10**-6]
        maxbox  = [4*np.pi,4*np.pi]
    else:
        minbox  = [-1,-1]
        maxbox = [1,1]
    return minbox,maxbox

def getfarr():
    farr = ["f1","f2","f3","f4","f5","f7","f8","f9","f10","f12","f13","f14","f15","f16",
            "f17","f18","f19","f20","f21","f22"]
    # farr = ["f1","f2","f3","f4","f5","f7","f8","f9","f10","f12","f13","f14","f15","f16",
    #         "f17","f18","f19","f21","f22"]
    # farr = ["f20"]
    # farr = ["f1"]

    return farr

def getdim(fname):
    dim = {"f1":2,"f2":2,"f3":2,"f4":2,"f5":2,"f7":2,"f8":2,"f9":2,"f10":4,"f12":2,"f13":2,
            "f14":2,"f15":2,"f16":2,"f17":3,"f18":4,"f19":4,"f20":4,"f21":2,"f22":2}
    return dim[fname]

def getnoiseinfo(noise):
    noisearr = ["0","10-2","10-4","10-6"]
    noisestr = ["","_noisepct10-2","_noisepct10-4","_noisepct10-6"]
    noisepct = [0,10**-2,10**-4,10**-6]

    for i,n in enumerate(noisearr):
        if(n == noise):
            return noisestr[i],noisepct[i]

def knowmissing(filename):
    arr = [

    ]
    for a in arr:
        if(filename == a):
            return 1
    return 0

def ploterrorbars(fff, baseline=13.5,usejson=0):
    import matplotlib as mpl
    import json
    import apprentice
    if not os.path.exists('results/plots/'):
        os.makedirs('results/plots/',exist_ok = True)


    # mpl.use('pgf')
    # pgf_with_custom_preamble = {
    #     "text.usetex": True,    # use inline math for ticks
    #     "pgf.rcfonts": False,   # don't setup fonts from rc parameters
    #     "pgf.preamble": [
    #         "\\usepackage{amsmath}",         # load additional packages
    #     ]
    # }
    # mpl.rcParams.update(pgf_with_custom_preamble)


    # fff = getfarr()
    # pqqq = ['p4_q3','p2_q3','p3_q3','p3_q7','p2_q7','p3_q6','p2_q3','p3_q3']
    width = 0.15
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1,2,figsize=(15,10),sharey=True)
    data = {}
    noiselevels = ['0','10-6','10-2']
    # noiselevels = ['0']
    # allsamples = ['mc','lhs','so','sg']
    allsamples = ['lhs','splitlhs','sg']
    # allsamples = ['mc','lhs']
    # allsamples = ['sg']
    if(usejson == 0):
        for snum,sample in enumerate(allsamples):
            data[sample] = {}
            # first = sample
            for nnum,noise in enumerate(noiselevels):
                data[sample][noise] = {}

                second = noise

                noisestr,noisepct = getnoiseinfo(noise)

                for fnum,fname in enumerate(fff):
                    data[sample][noise][fname] = {}

                    # IF USING TESTFILE

                    # testfile = "../benchmarkdata/"+fname+"_test.txt"
                    # # testfile = "../benchmarkdata/"+fname+".txt"
                    # print(testfile)
                    # bottom_or_all = all
                    # try:
                    #     X, Y = readData(testfile)
                    # except:
                    #     DATA = tools.readH5(testfile, [0])
                    #     X, Y= DATA[0]
                    #
                    # if(bottom_or_all == "bottom"):
                    #     testset = [i for i in range(trainingsize,len(X_test))]
                    #     X_test = X[testset]
                    #     Y_test = Y[testset]
                    # else:
                    #     X_test = X
                    #     Y_test = Y


                    # IF USING POLEDATA FILES
                    dim = getdim(fname)
                    infile = "results/plots/poledata_corner"+str(dim)+"D.csv"
                    X_test_1 = np.loadtxt(infile, delimiter=',')
                    infile = "results/plots/poledata_inside"+str(dim)+"D.csv"
                    X_test_2 = np.loadtxt(infile, delimiter=',')
                    X_test = np.vstack([X_test_1,X_test_2])
                    minarr,maxarr = getbox(fname)
                    s = apprentice.Scaler(np.array(X_test, dtype=np.float64), a=minarr, b=maxarr)
                    X_test = s.scaledPoints
                    # print(np.shape(X_test_1),np.shape(X_test_2),np.shape(X_test))
                    Y_test = np.array(getData(X_test,fname,0))


                    # print(np.shape(np.array(Y_test)))
                    # exit(1)

                    ts = "2x"

                    datapa = []
                    datara = []
                    datarard = []
                    datarasip = []
                    for run in ["exp1","exp2","exp3","exp4","exp5"]:
                    # for run in ["./"]:
                        fndesc = "%s%s_%s_%s"%(fname,noisestr,sample,ts)
                        folder = "results/%s/%s"%(run,fndesc)
                        m = 5
                        n = 5
                        pq = "p%d_q%d"%(m,n)
                        print(run, fname,noisestr,sample,m,n)

                        rappsipfile = "%s/outrasip/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)
                        rappfile = "%s/outra/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)
                        rapprdfile = "%s/outrard/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)
                        pappfile = "%s/outpa/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)
                        if not os.path.exists(rappsipfile):
                            print("rappsipfile %s not found"%(rappsipfile))
                            if(knowmissing(rappsipfile)):
                                if(sample == "sg"):
                                    break
                                continue
                            exit(1)

                        if not os.path.exists(rappfile):
                            print("rappfile %s not found"%(rappfile))
                            if(knowmissing(rappfile)):
                                if(sample == "sg"):
                                    break
                                continue
                            exit(1)

                        if not os.path.exists(rapprdfile):
                            print("rappfile %s not found"%(rapprdfile))
                            if(knowmissing(rapprdfile)):
                                if(sample == "sg"):
                                    break
                                continue
                            exit(1)

                        if not os.path.exists(pappfile):
                            print("pappfile %s not found"%(pappfile))
                            if(knowmissing(pappfile)):
                                if(sample == "sg"):
                                    break
                                continue
                            exit(1)

                        rappsip = RationalApproximationSIP(rappsipfile)
                        Y_pred_rappsip = rappsip.predictOverArray(X_test)

                        rapp = RationalApproximationONB(fname=rappfile)
                        Y_pred_rapp = np.array([rapp(x) for x in X_test])

                        rapprd = RationalApproximationONB(fname=rapprdfile)
                        Y_pred_rapprd = np.array([rapprd(x) for x in X_test])

                        papp = PolynomialApproximation(fname=pappfile)
                        Y_pred_papp = np.array([papp(x) for x in X_test])

                        datapa.append(np.log10(np.sqrt(np.sum((Y_pred_papp-Y_test)**2))))
                        datara.append(np.log10(np.sqrt(np.sum((Y_pred_rapp-Y_test)**2))))
                        datarard.append(np.log10(np.sqrt(np.sum((Y_pred_rapprd-Y_test)**2))))
                        datarasip.append(np.log10(np.sqrt(np.sum((Y_pred_rappsip-Y_test)**2))))

                        if(sample == "sg"):
                            break

                    missingmean = -15
                    if(len(datapa) == 0):
                        data[sample][noise][fname]['pamean'] = missingmean
                        data[sample][noise][fname]['pasd'] = 0
                    else:
                        data[sample][noise][fname]['pamean'] = np.average(datapa)
                        data[sample][noise][fname]['pasd'] = np.std(datapa)


                    if(len(datara) == 0):
                        data[sample][noise][fname]['ramean'] = missingmean
                        data[sample][noise][fname]['rasd'] = 0
                    else:
                        data[sample][noise][fname]['ramean'] = np.average(datara)
                        data[sample][noise][fname]['rasd'] = np.std(datara)
                    if(len(datarard) == 0):
                        data[sample][noise][fname]['rardmean'] = missingmean
                        data[sample][noise][fname]['rardsd'] = 0
                    else:
                        data[sample][noise][fname]['rardmean'] = np.average(datarard)
                        data[sample][noise][fname]['rardsd'] = np.std(datarard)
                    if(len(datarasip) == 0):
                        data[sample][noise][fname]['rasipmean'] = missingmean
                        data[sample][noise][fname]['rasipsd'] = 0
                    else:
                        data[sample][noise][fname]['rasipmean'] = np.average(datarasip)
                        data[sample][noise][fname]['rasipsd'] = np.std(datarasip)

                    if(sample == "sg"):
                        data[sample][noise][fname]['pasd'] = 0
                        data[sample][noise][fname]['rasd'] = 0
                        data[sample][noise][fname]['rardsd'] = 0
                        data[sample][noise][fname]['rasipsd'] = 0



        outfile111 = "results/plots/Jerrors_"+fff[0]+".json"
        import json
        with open(outfile111, "w") as f:
            json.dump(data, f,indent=4, sort_keys=True)
        exit(0)
    # else:
        # import json
        # outfile111 = "results/plots/Jerrors.json"
        # if outfile111:
        #     with open(outfile111, 'r') as fn:
        #         data = json.load(fn)

    ecolor = 'black'
    # if(plottype == 'persample' or plottype == 'pernoiselevel'):
    # if(plottype == 'persample'):
        # minval = np.Infinity
    methodarr = ['ra','rard', 'rasip','pa']
    import matplotlib.pyplot as plt
    ffffff = plt.figure(0,figsize=(25, 20))
    # totalrow = 5
    # totalcol = 4
    totalrow = 2
    totalcol = 2
    baseline = baseline
    # color = ['#900C3F','#C70039','#FF5733','#FFC300']
    color = ['#FFC300','#FF5733','#900C3F']
    width = 0.2
    ecolor = 'black'
    plt.rc('ytick',labelsize=14)
    plt.rc('xtick',labelsize=14)
    props = dict(boxstyle='square', facecolor='wheat', alpha=0.5)
    X111 = np.arange(len(noiselevels)*len(methodarr))
    # color100 = ['#FFC300','#FF5733','#900C3F']
    # color1k = ['yellow','wheat','r']
    axarray = []

    for fnum, fname in enumerate(fff):
        import json
        outfile111 = "results/plots/Jerrors_"+fname+".json"
        if outfile111:
            with open(outfile111, 'r') as fn:
                data = json.load(fn)
        plotd = {}
        for snum, sample in enumerate(allsamples):
            plotd[sample] = {}
            plotd[sample]['mean'] = []
            plotd[sample]['sd'] = []
            for nnum, noise in enumerate(noiselevels):
                for method in methodarr:
                    meankey = method+'mean'
                    sdkey = method+'sd'

                    plotd[sample]['mean'].append(data[sample][noise][fname][meankey])
                    plotd[sample]['sd'].append(data[sample][noise][fname][sdkey])
        if(len(axarray)>0):
            ax = plt.subplot2grid((totalrow,totalcol), (int(fnum/totalcol),int(fnum%totalcol)),sharex=axarray[0],sharey=axarray[0])
            axarray.append(ax)
        else:
            ax = plt.subplot2grid((totalrow,totalcol), (int(fnum/totalcol),int(fnum%totalcol)))
            axarray.append(ax)
        ax.set_xlim(-.3,11.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.axvspan(-.3, 3.7, alpha=0.5, color='pink')
        plt.axvspan(3.7, 7.7, alpha=0.5, color='lightgrey')
        plt.axvspan(7.7, 11.7, alpha=0.5, color='cyan')
        plt.title(fname)
        # plt.text(1,3.35, "$\\epsilon = 0$", fontsize=10)
        # plt.text(4,3.35, "$\\epsilon = 10^{-6}$", fontsize=10)
        # plt.text(7,3.35, "$\\epsilon = 10^{-2}$", fontsize=10)
        labels = ['Latin hypercube sampling', 'Split latin hypercube sampling', 'Sparse grids']
        legendarr = ['$\\epsilon=0$','$\\epsilon=10^{-6}$','$\\epsilon=10^{-2}$']
        # plt.tight_layout()
        for snum, sample in enumerate(allsamples):
            if(sample == 'sg'):
                ax.bar(X111+snum*width, np.array(plotd[sample]['mean'])+baseline, width, color=color[snum], label=labels[snum])
            else:
                ax.bar(X111+snum*width, np.array(plotd[sample]['mean'])+baseline, width,color=color[snum], yerr=np.array(plotd[sample]['sd']),align='center',  ecolor=ecolor, capsize=3,label=labels[snum])

        if(fnum==0):
            l1 = ffffff.legend(loc='upper center',ncol=3,fontsize = 20)
        l2 = ffffff.legend(legendarr,loc='upper center', ncol=4,bbox_to_anchor=(0.435, 0.83), fontsize = 20,borderaxespad=0.,shadow=False)
        # ax.label_outer()
        # if(fnum==0):
        #     l222 = ffffff.legend(loc='upper center', ncol=4,bbox_to_anchor=(0.5, 0.92), fontsize = 20,borderaxespad=0.,shadow=False)

        ax.set_xticks(X111 + (len(allsamples)-1)*width / 2)
        xlab = [
            'Algorithm \\ref{ALG:MVVandQR} w/o DR',
            'Algorithm \\ref{ALG:MVVandQR}' ,
            'Algorithm \\ref{A:Polyak}',
            'Poly. Approx.',
            'Algorithm \\ref{ALG:MVVandQR} w/o DR',
            'Algorithm \\ref{ALG:MVVandQR}' ,
            'Algorithm \\ref{A:Polyak}',
            'Poly. Approx.',
            'Algorithm \\ref{ALG:MVVandQR} w/o DR',
            'Algorithm \\ref{ALG:MVVandQR}' ,
            'Algorithm \\ref{A:Polyak}',
            'Poly. Approx.'
        ]
        methodlabel = ['$r_1$','$r_2$','$r_3$','$r_4$']
        xlab1 = np.concatenate((methodlabel,methodlabel,methodlabel),axis=None)
        ax.set_xticklabels(xlab1,fontsize = 22)
        # ax.set_xlabel("Approach",fontsize=22)
        ax.set_ylabel("$\\log_{10}\\left[\\Delta_r\\right]$",fontsize=22)
        # ax.label_outer()
    # ffffff.text(0.08, 0.5, "$\\log_{10}\\left[\\Delta_r\\right]$", fontsize=22,va='center', rotation='vertical')



    # plt.show()
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: x-baseline))
    # plt.tight_layout()
    # plt.savefig("../../log/errors.png", bbox_extra_artists=(l1,l111,), bbox_inches='tight')
    ffffff.savefig('../../log/errors.png', bbox_extra_artists=(l1,l2,), bbox_inches='tight')
    # plt.savefig("../../log/errors.png")
    plt.clf()
    plt.close('all')








    exit(0)
    for snum,sample in enumerate(allsamples):
        import matplotlib.pyplot as plt
        plt.rc('ytick',labelsize=14)
        fig, axarr = plt.subplots(3, 1, sharey=True,figsize=(21,20))
        for nnum,noise in enumerate(noiselevels):
            pa = []
            ra = []
            rard = []
            rasip = []
            paerror = []
            raerror = []
            rarderror = []
            rasiperror = []
            for fnum,fname in enumerate(fff):
                pa.append(data[sample][noise][fname]['pamean'])
                paerror.append(data[sample][noise][fname]['pasd'])

                ra.append(data[sample][noise][fname]['ramean'])
                raerror.append(data[sample][noise][fname]['rasd'])

                rard.append(data[sample][noise][fname]['rardmean'])
                rarderror.append(data[sample][noise][fname]['rardsd'])

                rasip.append(data[sample][noise][fname]['rasipmean'])
                rasiperror.append(data[sample][noise][fname]['rasipsd'])

            p1 = axarr[nnum].bar(X111, np.array(pa)+baseline, width,color=color[0], yerr=np.array(paerror),align='center',  ecolor=ecolor, capsize=3)
            p2 = axarr[nnum].bar(X111+width, np.array(ra)+baseline, width,color=color[1],yerr=np.array(raerror),align='center',ecolor=ecolor, capsize=3)
            p3 = axarr[nnum].bar(X111+2*width, np.array(rard)+baseline, width,color=color[2],yerr=np.array(rarderror),align='center',ecolor=ecolor, capsize=3)
            p4 = axarr[nnum].bar(X111+3*width, np.array(rasip)+baseline, width,color=color[3],yerr=np.array(rasiperror),align='center', alpha=0.5, ecolor=ecolor, capsize=3)
            axarr[nnum].legend((p1[0], p2[0],p3[0],p4[0]), ('Polynomial Approx. ', 'Algorithm \\ref{ALG:MVVandQR}','Algorithm \\ref{ALG:MVVandQR} w/ DR' ,'Algorithm \\ref{A:Polyak}'),loc = 'upper right',fontsize = 15)

        for ax in axarr.flat:
            ax.set_xticks(X111 + 3*width / 2)
            xlab = []
            for f in fff:
                print(f)
                # xlab.append("\\ref{fn:%s}"%(f))
                xlab.append("%s"%(f))
            ax.set_xticklabels(xlab,fontsize = 14)
            ax.set_ylabel('$log_{10}(\\Delta_r)$',fontsize = 17)
            ax.label_outer()

        plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: x-baseline))
        plt.tight_layout()
        print(xlab)
        # plt.show()
        # plt.savefig("plots/Perrorbars.pgf", bbox_inches="tight")
        # outfile111 = "results/plots/Perrorbars_for_%s.pdf"%(sample)
        outfile111 = "results/plots/Perrorbars_for_%s.pgf"%(sample)
        plt.savefig(outfile111, bbox_inches="tight")
        plt.clf()
        plt.close('all')


    # elif(plottype == 'pernoiselevel'):

    # FOR FUTURE
    # approxqqq = ["Polynomial Approximation", 'RA (linear algebra) without degree reduction', 'RA (linear algebra) with degree reduction', 'Pole-free RA']
    # for nnum,noise in enumerate(noiselevels):
    #     import matplotlib.pyplot as plt
    #     plt.rc('ytick',labelsize=14)
    #     fig, axarr = plt.subplots(4, 1, sharey=True,figsize=(21,20))
    #     for anum,approx in enumerate(["pa","ra","rard","rasip"]):
    #         barobj = {}
    #         for snum,sample in enumerate(allsamples):
    #             mean = []
    #             sd = []
    #             for fnum,fname in enumerate(fff):
    #                 mean.append(data[sample][noise][fname][approx+"mean"])
    #                 sd.append(data[sample][noise][fname][approx+"sd"])
    #             barobj[snum] = axarr[anum].bar(X111+snum*width, np.array(mean)+baseline, width,color=color[snum], yerr=np.array(sd),align='center',  ecolor=ecolor, capsize=3)
    #
    #         axarr[anum].legend((barobj[0][0],barobj[1][0],barobj[2][0],barobj[3][0]),('Uniform Random','Latin Hypercube','Sobol Sequence', 'Sparse Grids'),loc = 'upper right',fontsize = 15)
    #         axarr[anum].set_title(" approx = "+approxqqq[anum],fontsize = 15)



        for nnum,noise in enumerate(noiselevels):
            import matplotlib.pyplot as plt
            plt.rc('ytick',labelsize=14)
            fig, axarr = plt.subplots(4, 1, sharey=True,figsize=(21,20))
            for anum,approx in enumerate(["pa","ra","rard","rasip"]):
                for snum,sample in enumerate(allsamples):
                    barobj = []
                    mean = []
                    sd = []
                    for fnum,fname in enumerate(fff):
                        mean.append(data[sample][noise][fname][approx+"mean"])
                        sd.append(data[sample][noise][fname][approx+"sd"])
                    barobj.append(axarr[anum].bar(X111+snum*width, np.array(mean)+baseline, width,color=color[snum], yerr=np.array(sd),align='center',  ecolor=ecolor, capsize=3))
                # axarr[anum].legend(barobj,('Polynomial Approx. ', 'Algorithm \\ref{ALG:MVVandQR}','Algorithm \\ref{A:Polyak}'),loc = 'upper right',fontsize = 15)
                axarr[anum].set_title(str(allsamples)+ " approx = "+approx)
            for ax in axarr.flat:
                ax.set_xticks(X111 + (len(allsamples)-1)*width / 2)
                xlab = []
                for f in fff:
                    print(f)
                    # xlab.append("\\ref{fn:%s}"%(f))
                    xlab.append("%s"%(f))
                ax.set_xticklabels(xlab,fontsize = 14)
                ax.set_ylabel('$log_{10}(\\Delta_r)$',fontsize = 17)
                ax.label_outer()

            plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: x-baseline))
            plt.tight_layout()
            print(xlab)
            # plt.show()
            # plt.savefig("plots/Perrorbars.pgf", bbox_inches="tight")
            # outfile111 = "results/plots/Perrorbars_for_%s.pdf"%(noise)
            outfile111 = "results/plots/Perrorbars_for_%s.pgf"%(noise)
            plt.savefig(outfile111, bbox_inches="tight")
            plt.clf()
            plt.close('all')


    # s = ""
    # for num,fname in enumerate(fff):
    #     s += "\\ref{fn:%s}"%(fname)
    #     s += "&%.3E&%.3E&%.3E&%.3E&%.3E&%.3E"%(pa[num],paerror[num],ra[num],raerror[num],rasip[num],rasiperror[num])
    #     s+="\\\\\hline\n"
    #
    # s += "\n"
    # for num,fname in enumerate(fff):
    #     s += "\\ref{fn:%s}"%(fname)
    #     s += "&%.3E&%.3E&%.3E&%.3E&%.3E&%.3E"%(pa1[num],paerror1[num],ra1[num],raerror1[num],rasip1[num],rasiperror1[num])
    #     s+="\\\\\hline\n"
    #
    # s += "\n"
    # for num,fname in enumerate(fff):
    #     s += "\\ref{fn:%s}"%(fname)
    #     s += "&%.3E&%.3E&%.3E&%.3E&%.3E&%.3E"%(pa2[num],paerror2[num],ra2[num],raerror2[num],rasip2[num],rasiperror2[num])
    #     s+="\\\\\hline\n"
    # print(s)


# for each \ep =0,10-1,10-3
# FNO     \Delta_r for PA     \Delta_r for 3.1        \Delta_r for 4.1
#             M SD                     M SD                   M SD


    # print(s)


if __name__ == "__main__":


 # python tablepoles.py f1,f2,f3,f4,f5,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f19,f20,f22  0,10-1 10,100,1000 2x  table
 # for fno in {1..5} {7..10} {12..20} 22; do  name="f"$fno; nohup python tablepoles.py $name 0,10-1 10,100,1000 2x  table> ../../debug/"tablepoles_"$name".log" 2>&1 & done
 # for fno in 3 5 9 13 14 18 19; do  name="f"$fno; nohup python tablepoles.py $name 0,10-1 10,100,1000 2x  latex> ../../debug/"tablepoles_latex_"$name".log" 2>&1 & done
    # import os, sys
    # if len(sys.argv) != 6:
    #     print("Usage: {} function noise thresholds ts testfilelist bottom_or_all table_or_latex_or_latexall_or_ploterror".format(sys.argv[0]))
    #     sys.exit(1)
    #
    # farr = sys.argv[1].split(',')
    # if len(farr) == 0:
    #     print("please specify comma saperated functions")
    #     sys.exit(1)
    #
    # noisearr = sys.argv[2].split(',')
    # if len(noisearr) == 0:
    #     print("please specify comma saperated noise levels")
    #     sys.exit(1)
    #
    # thresholdarr = sys.argv[3].split(',')
    # if len(thresholdarr) == 0:
    #     print("please specify comma saperated threshold levels")
    #     sys.exit(1)
    #
    # # testfilearr = sys.argv[5].split(',')
    # # if len(testfilearr) == 0:
    # #     print("please specify comma saperated testfile paths")
    # #     sys.exit(1)
    # #
    # # bottomallarr = sys.argv[6].split(',')
    # # if len(bottomallarr) == 0:
    # #     print("please specify comma saperated bottom or all options")
    # #     sys.exit(1)
    #
    #
    # # tablepoles(farr,noisearr, thresholdarr, testfilearr, bottomallarr,sys.argv[4],sys.argv[7])

     # for fno in {1..5} {7..10} {12..22}; do  name="f"$fno; nohup python ploterrorbars.py $name 14 0 > ../../log/"ploterrorbarjson_"$name".log" 2>&1 &  done
     # python ploterrorbars.py f1,f2,f3,f4,f5,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22 14 1
     # python ploterrorbars.py f4,f8,f18,f21 14 1
    import os, sys

    farr = sys.argv[1].split(',')
    if len(farr) == 0:
        print("please specify comma saperated functions")
        sys.exit(1)

    if len(sys.argv)==2:
        ploterrorbars(farr)
    elif len(sys.argv)==3:
        ploterrorbars(farr,float(sys.argv[2]))
    elif len(sys.argv)==4:
        ploterrorbars(farr,float(sys.argv[2]),int(sys.argv[3]))
    else:
        print("farr baseline (13.5), usejson(0 or 1) ")
###########
