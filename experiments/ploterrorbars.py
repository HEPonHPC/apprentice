import numpy as np
from apprentice import RationalApproximationSIP, RationalApproximationONB, PolynomialApproximation
from apprentice import tools, readData
import matplotlib.ticker as mtick
import os


def ploterrorbars(baseline=0):
    import matplotlib as mpl
    import json


    # mpl.use('pgf')
    # pgf_with_custom_preamble = {
    #     "text.usetex": True,    # use inline math for ticks
    #     "pgf.rcfonts": False,   # don't setup fonts from rc parameters
    #     "pgf.preamble": [
    #         "\\usepackage{amsmath}",         # load additional packages
    #     ]
    # }
    # mpl.rcParams.update(pgf_with_custom_preamble)


    fff = ['f1','f2','f3','f4','f5','f7','f8','f9','f10','f12','f13','f14','f15','f16','f17','f18','f19','f20','f21','f22']
    # fff = ['f1','f2','f3','f4','f5','f7','f8','f9','f10','f12','f13','f14','f15','f16','f17','f18','f19','f21','f22']
    # fff = ['f12','f13','f14','f15','f16']
    # fff = ['f1','f2','f3','f4']
    # fff = ['f1']
    # fff = ['f20']
    # pqqq = ['p4_q3','p2_q3','p3_q3','p3_q7','p2_q7','p3_q6','p2_q3','p3_q3']
    width = 0.15
    X111 = np.arange(len(fff))
    width = 0.15
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1,2,figsize=(15,10),sharey=True)
    for snum,sample in enumerate(['mc','lhs','sc','sg']):
    # for snum,sample in enumerate(['lhs']):


        import matplotlib.pyplot as plt
        plt.rc('ytick',labelsize=14)
        fig, axarr = plt.subplots(3, 1, sharey=True,figsize=(21,20))

        ecolor = 'black'
        minval = np.Infinity



        first = sample
        for nnum,noise in enumerate(['0','10-1','10-3']):

            plotindex = nnum
            second = noise


            noisestr = ""
            noisepct = 0
            if(noise!="0"):
                noisestr = "_noisepct"+noise
            pa = []
            ra = []
            rasip = []
            paerror = []
            raerror = []
            rasiperror = []

            for fnum,fname in enumerate(fff):
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
                datarasip = []
                for run in ["exp1","exp2","exp3","exp4","exp5"]:
                # for run in ["./"]:
                    fndesc = "%s%s_%s_%s"%(fname,noisestr,sample,ts)
                    folder = "results/%s/%s"%(run,fndesc)
                    m = 4
                    n = 3
                    pq = "p%d_q%d"%(m,n)
                    print(run, fname,noisestr,sample,m,n)

                    rappsipfile = "%s/outrasip/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)
                    rappfile = "%s/outra/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)
                    pappfile = "%s/outpa/%s_%s_ts%s.json"%(folder,fndesc,pq,ts)
                    if not os.path.exists(rappsipfile):
                        print("rappsipfile %s not found"%(rappsipfile))
                        exit(1)

                    if not os.path.exists(rappfile):
                        print("rappfile %s not found"%(rappfile))
                        exit(1)

                    if not os.path.exists(pappfile):
                        print("pappfile %s not found"%(pappfile))
                        exit(1)

                    rappsip = RationalApproximationSIP(rappsipfile)
                    Y_pred_rappsip = rappsip.predictOverArray(X_test)
                    rapp = RationalApproximationONB(fname=rappfile)
                    Y_pred_rapp = np.array([rapp(x) for x in X_test])
                    papp = PolynomialApproximation(fname=pappfile)
                    Y_pred_papp = np.array([papp(x) for x in X_test])

                    datapa.append(np.log10(np.sqrt(np.sum((Y_pred_papp-Y_test)**2))))
                    datara.append(np.log10(np.sqrt(np.sum((Y_pred_rapp-Y_test)**2))))
                    datarasip.append(np.log10(np.sqrt(np.sum((Y_pred_rappsip-Y_test)**2))))

                    if(sample == "sc" or sample == "sg"):
                        break

                pa.append(np.average(datapa))
                paerror.append(np.std(datapa))

                ra.append(np.average(datara))
                raerror.append(np.std(datara))

                rasip.append(np.average(datarasip))
                rasiperror.append(np.std(datarasip))

            minval = min(minval,min(pa))
            minval = min(minval,min(ra))
            minval = min(minval,min(rasip))
            # print(pa)
            # print(ra)
            # print(rasip)


            # print(len(pa),len(ra),len(rasip))

            # print(min(pa),min(ra),min(rasip))
            # print(np.c_[pa,np.log10(pa)])

            # baseline = 20
            p1 = axarr[plotindex].bar(X111, np.array(pa)+baseline, width,color='#C70039', yerr=np.array(paerror),align='center',  ecolor=ecolor, capsize=3)
            p2 = axarr[plotindex].bar(X111+width, np.array(ra)+baseline, width,color='#FF5733',yerr=np.array(raerror),align='center',ecolor=ecolor, capsize=3)
            p3 = axarr[plotindex].bar(X111+2*width, np.array(rasip)+baseline, width,color='#FFC300',yerr=np.array(rasiperror),align='center', alpha=0.5, ecolor=ecolor, capsize=3)
            axarr[plotindex].legend((p1[0], p2[0],p3[0]), ('Polynomial Approx. ', 'Algorithm \\ref{ALG:MVVandQR}','Algorithm \\ref{A:Polyak}'),loc = 'lower left',fontsize = 15)
            # ax2.legend((p1[0], p2[0],p3[0]), ('Polynomial Approx. ', 'Algorithm \\ref{ALG:MVVandQR}','Algorithm \\ref{A:Polyak}'),loc = 'lower left',fontsize = 15)
            # ax3.legend((p1[0], p2[0],p3[0]), ('Polynomial Approx. ', 'Algorithm \\ref{ALG:MVVandQR}','Algorithm \\ref{A:Polyak}'),loc = 'lower left',fontsize = 15)

            axarr[plotindex].set_xticks(X111 + 2*width / 2)
            # ax2.set_xticks(X111 + 2*width / 2)
            # ax3.set_xticks(X111 + 2*width / 2)
            xlab = []
            for f in fff:
                print(f)
                xlab.append("\\ref{fn:%s}"%(f))
            print(xlab)

            axarr[plotindex].set_xticklabels(xlab,fontsize = 14)
            # ax2.set_xticklabels(xlab,fontsize = 14)
            # ax3.set_xticklabels(xlab,fontsize = 14)

            # ax3.set_xlabel('Function No.',fontsize = 17)
            # ax2.set_xlabel('Function No.',fontsize = 17)
            axarr[plotindex].set_ylabel('$log_{10}(\\Delta_r)$',fontsize = 17)
            # ax2.set_ylabel('$log_{10}(\\Delta_r)$',fontsize = 17)
            # ax3.set_ylabel('$log_{10}(\\Delta_r)$',fontsize = 17)

        # p1 = axarr[plotindex].bar(X111, np.log10(pa), width,color='#900C3F')
        # p2 = axarr[plotindex].bar(X111+width, np.log10(ra), width,color='#FF5733')
        # p3 = axarr[plotindex].bar(X111+2*width, np.log10(rasip), width,color='#FFC300')

        # p1 = ax2.bar(X111, np.log10(pa1), width,color='#C70039', yerr=np.log10(paerror1),align='center',  ecolor=ecolor, capsize=3)
        # p2 = ax2.bar(X111+width, np.log10(ra1), width,color='#FF5733',yerr=np.log10(raerror1),align='center',ecolor=ecolor, capsize=3)
        # p3 = ax2.bar(X111+2*width, np.log10(rasip1), width,color='#FFC300',yerr=np.log10(rasiperror1),align='center', alpha=0.5, ecolor=ecolor, capsize=3)

        # p1 = ax2.bar(X111, np.log10(pa1), width,color='#900C3F')
        # p2 = ax2.bar(X111+width, np.log10(ra1), width,color='#FF5733')
        # p3 = ax2.bar(X111+2*width, np.log10(rasip1), width,color='#FFC300')

        # p1= ax3.errorbar(X111, np.log10(pa1), np.log10(paerror2), linestyle='None', marker='o',ecolor=ecolor,color='#900C3F',capsize=3)
        # p1= ax3.errorbar(X111+width, np.log10(ra1), np.log10(raerror2), linestyle='None', marker='o',ecolor=ecolor,color='#900C3F',capsize=3)
        # p1= ax3.errorbar(X111+2*width, np.log10(rasip1), np.log10(rasiperror2), linestyle='None', marker='o',ecolor=ecolor,color='#900C3F',capsize=3)

        # p1 = ax3.bar(X111, np.log10(pa2), width,color='#C70039', yerr=np.log10(paerror2),align='center',  ecolor=ecolor, capsize=3)
        # p2 = ax3.bar(X111+width, np.log10(ra2), width,color='#FF5733',yerr=np.log10(raerror2),align='center',ecolor=ecolor, capsize=3)
        # p3 = ax3.bar(X111+2*width, np.log10(rasip2), width,color='#FFC300',yerr=np.log10(rasiperror2),align='center', alpha=0.5, ecolor=ecolor, capsize=3)

        # p1 = ax3.bar(X111, np.log10(pa2), width,color='#900C3F')
        # p2 = ax3.bar(X111+width, np.log10(ra2), width,color='#FF5733')
        # p3 = ax3.bar(X111+2*width, np.log10(rasip2), width,color='#FFC300')





        # axarr[plotindex].set_ylim((-10,10))
        # ax2.set_ylabel('$\\Delta_r$',fontsize = 17)

            # ax.set_ylim([-9,4])
        print('minval = %f'%minval)
        plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: x-baseline))
        plt.tight_layout()
        # plt.show()
        # plt.savefig("plots/Perrorbars.pgf", bbox_inches="tight")
        if not os.path.exists("results/plots"):
            os.makedirs("results/plots", exist_ok = True)
        outfile111 = "results/plots/Perrorbars_for_%s.pdf"%(first)
        plt.savefig(outfile111, bbox_inches="tight")

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
    if len(sys.argv)==1:
        ploterrorbars()
    else:
        ploterrorbars(float(sys.argv[1]))
###########
