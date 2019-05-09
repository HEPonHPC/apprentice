import numpy as np
from apprentice import RationalApproximationSIP, RationalApproximationONB, PolynomialApproximation
from apprentice import tools, readData
import matplotlib.ticker as mtick
import os

def getfarr():
    farr = ["f1","f2","f3","f4","f5","f7","f8","f9","f10","f12","f13","f14","f15","f16",
            "f17","f18","f19","f20","f21","f22"]
    # farr = ["f1","f2","f3","f4","f5","f7","f8","f9","f10","f12","f13","f14","f15","f16",
    #         "f17","f18","f19","f21","f22"]
    # farr = ["f20"]

    return farr

def getnoiseinfo(noise):
    noisearr = ["0","10-2","10-4","10-6"]
    noisestr = ["","_noisepct10-2","_noisepct10-4","_noisepct10-6"]
    noisepct = [0,10**-2,10**-4,10**-6]

    for i,n in enumerate(noisearr):
        if(n == noise):
            return noisestr[i],noisepct[i]

def knowmissing(filename):
    arr = [
        "results/exp1/f18_noisepct10-2_sg_2x/outrard/f18_noisepct10-2_sg_2x_p5_q5_ts2x.json",
        "results/exp1/f18_noisepct10-6_sg_2x/outrard/f18_noisepct10-6_sg_2x_p5_q5_ts2x.json"
    ]
    for a in arr:
        if(filename == a):
            return 1
    return 0

def ploterrorbars(baseline=13.5,plottype='persample',usejson=0):
    import matplotlib as mpl
    import json
    if not os.path.exists('results/plots/'):
        os.makedirs('results/plots/',exist_ok = True)


    mpl.use('pgf')
    pgf_with_custom_preamble = {
        "text.usetex": True,    # use inline math for ticks
        "pgf.rcfonts": False,   # don't setup fonts from rc parameters
        "pgf.preamble": [
            "\\usepackage{amsmath}",         # load additional packages
        ]
    }
    mpl.rcParams.update(pgf_with_custom_preamble)

    color = ['#900C3F','#C70039','#FF5733','#FFC300']

    fff = getfarr()
    # pqqq = ['p4_q3','p2_q3','p3_q3','p3_q7','p2_q7','p3_q6','p2_q3','p3_q3']
    width = 0.15
    X111 = np.arange(len(fff))
    width = 0.15
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1,2,figsize=(15,10),sharey=True)
    data = {}
    noiselevels = ['0','10-2','10-6']
    allsamples = ['mc','lhs','so','sg']
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
                    testfile = "../benchmarkdata/"+fname+"_test.txt"
                    # testfile = "../benchmarkdata/"+fname+".txt"
                    print(testfile)
                    bottom_or_all = all
                    try:
                        X, Y = readData(testfile)
                    except:
                        DATA = tools.readH5(testfile, [0])
                        X, Y= DATA[0]

                    if(bottom_or_all == "bottom"):
                        testset = [i for i in range(trainingsize,len(X_test))]
                        X_test = X[testset]
                        Y_test = Y[testset]
                    else:
                        X_test = X
                        Y_test = Y
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

        outfile111 = "results/plots/Jerrors.json"
        import json
        with open(outfile111, "w") as f:
            json.dump(data, f,indent=4, sort_keys=True)
    else:
        import json
        outfile111 = "results/plots/Jerrors.json"
        if outfile111:
            with open(outfile111, 'r') as fn:
                data = json.load(fn)

    ecolor = 'black'
    if(plottype == 'persample' or plottype == 'pernoiselevel'):
    # if(plottype == 'persample'):
        # minval = np.Infinity

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
    import os, sys
    if len(sys.argv)==2:
        ploterrorbars(float(sys.argv[1]))
    elif len(sys.argv)==3:
        ploterrorbars(float(sys.argv[1]),sys.argv[2])
    elif len(sys.argv)==4:
        ploterrorbars(float(sys.argv[1]),sys.argv[2],int(sys.argv[3]))
    else:
        print("baseline (13.5), plottype (persample or pernoiselevel), usejson(0 or 1) ")
###########
