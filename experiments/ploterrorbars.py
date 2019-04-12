import numpy as np
from apprentice import RationalApproximationSIP, RationalApproximationONB, PolynomialApproximation
from apprentice import tools, readData
import os


def ploterrorbars():
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

    width = 0.15
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1,2,figsize=(15,10),sharey=True)
    pa = []
    ra = []
    rasip = []
    paerror = []
    raerror = []
    rasiperror = []

    pa1 = []
    ra1 = []
    rasip1 = []
    paerror1 = []
    raerror1 = []
    rasiperror1 = []

    pa2 = []
    ra2 = []
    rasip2 = []
    paerror2 = []
    raerror2 = []
    rasiperror2 = []

    fff = ['f1','f2','f3','f4','f5','f7','f8','f9','f10','f12','f13','f14','f15','f16','f17','f18','f19','f20','f21','f22']
    # fff = ['f1','f2','f3','f4','f5','f7','f8','f9','f10']
    # pqqq = ['p4_q3','p2_q3','p3_q3','p3_q7','p2_q7','p3_q6','p2_q3','p3_q3']
    width = 0.15
    X111 = np.arange(len(fff))

    for num,fname in enumerate(fff):
        # pq = pqqq[num]
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

        noisestr = ""
        datapa = []
        datara = []
        datarasip = []
        for run in ["./","../experiments2/","../experiments3/","../experiments4/","../experiments5/"]:
            folder = "%s%s%s_%s"%(run,fname,noisestr,ts)
            if(run == "./"):
                optjsonfile = folder+"/plots/Joptdeg_"+fname+noisestr+"_jsdump_opt6.json"
                if not os.path.exists(optjsonfile):
                    print("optjsonfile: " + optjsonfile+ " not found")
                    exit(1)
                if optjsonfile:
                    with open(optjsonfile, 'r') as fn:
                        optjsondatastore = json.load(fn)

                optm1 = optjsondatastore['optdeg']['m']
                optn1 = optjsondatastore['optdeg']['n']
            pq = "p%d_q%d"%(optm1,optn1)
            print(fname,noisestr,optm1,optn1)
            rappsipfile = "%s/out/%s%s_%s_%s_ts%s.json"%(folder,fname,noisestr,ts,pq,ts)
            rappfile = "%s/outra/%s%s_%s_%s_ts%s.json"%(folder,fname,noisestr,ts,pq,ts)
            pappfile = "%s/outpa/%s%s_%s_%s_ts%s.json"%(folder,fname,noisestr,ts,pq,ts)
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

            datapa.append(np.sqrt(np.sum((Y_pred_papp-Y_test)**2)))
            datara.append(np.sqrt(np.sum((Y_pred_rapp-Y_test)**2)))
            datarasip.append(np.sqrt(np.sum((Y_pred_rappsip-Y_test)**2)))

        pa.append(np.average(datapa))
        paerror.append(np.std(datapa))

        ra.append(np.average(datara))
        raerror.append(np.std(datara))

        rasip.append(np.average(datarasip))
        rasiperror.append(np.std(datarasip))




        noisestr = "_noisepct10-1"
        datapa = []
        datara = []
        datarasip = []
        for run in ["./","../experiments2/","../experiments3/","../experiments4/","../experiments5/"]:
            folder = "%s%s%s_%s"%(run,fname,noisestr,ts)
            if(run == "./"):
                optjsonfile = folder+"/plots/Joptdeg_"+fname+noisestr+"_jsdump_opt6.json"
                if not os.path.exists(optjsonfile):
                    print("optjsonfile: " + optjsonfile+ " not found")
                    exit(1)
                if optjsonfile:
                    with open(optjsonfile, 'r') as fn:
                        optjsondatastore = json.load(fn)
                optm2 = optjsondatastore['optdeg']['m']
                optn2 = optjsondatastore['optdeg']['n']

            if(optn2==0):
                optm2=optn1
                optn2=optn1
            pq = "p%d_q%d"%(optm2,optn2)
            rappsipfile = "%s/out/%s%s_%s_%s_ts%s.json"%(folder,fname,noisestr,ts,pq,ts)
            rappfile = "%s/outra/%s%s_%s_%s_ts%s.json"%(folder,fname,noisestr,ts,pq,ts)
            pappfile = "%s/outpa/%s%s_%s_%s_ts%s.json"%(folder,fname,noisestr,ts,pq,ts)

            print(fname,noisestr,optm2,optn2)
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

            datapa.append(np.sqrt(np.sum((Y_pred_papp-Y_test)**2)))
            datara.append(np.sqrt(np.sum((Y_pred_rapp-Y_test)**2)))
            datarasip.append(np.sqrt(np.sum((Y_pred_rappsip-Y_test)**2)))

        pa1.append(np.average(datapa))
        paerror1.append(np.std(datapa))

        ra1.append(np.average(datara))
        raerror1.append(np.std(datara))

        rasip1.append(np.average(datarasip))
        rasiperror1.append(np.std(datarasip))

        noisestr = "_noisepct10-3"
        datapa = []
        datara = []
        datarasip = []
        for run in ["./","../experiments2/","../experiments3/","../experiments4/","../experiments5/"]:
            folder = "%s%s%s_%s"%(run,fname,noisestr,ts)
            if(run == "./"):
                optjsonfile = folder+"/plots/Joptdeg_"+fname+noisestr+"_jsdump_opt6.json"
                if not os.path.exists(optjsonfile):
                    print("optjsonfile: " + optjsonfile+ " not found")
                    exit(1)
                if optjsonfile:
                    with open(optjsonfile, 'r') as fn:
                        optjsondatastore = json.load(fn)
                optm2 = optjsondatastore['optdeg']['m']
                optn2 = optjsondatastore['optdeg']['n']

            if(optn2==0):
                optm2=optn1
                optn2=optn1
            pq = "p%d_q%d"%(optm2,optn2)
            rappsipfile = "%s/out/%s%s_%s_%s_ts%s.json"%(folder,fname,noisestr,ts,pq,ts)
            rappfile = "%s/outra/%s%s_%s_%s_ts%s.json"%(folder,fname,noisestr,ts,pq,ts)
            pappfile = "%s/outpa/%s%s_%s_%s_ts%s.json"%(folder,fname,noisestr,ts,pq,ts)

            print(fname,noisestr,optm2,optn2)
            print(fname,noisestr,optm2,optn2)
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

            datapa.append(np.sqrt(np.sum((Y_pred_papp-Y_test)**2)))
            datara.append(np.sqrt(np.sum((Y_pred_rapp-Y_test)**2)))
            datarasip.append(np.sqrt(np.sum((Y_pred_rappsip-Y_test)**2)))
        pa2.append(np.average(datapa))
        paerror2.append(np.std(datapa))

        ra2.append(np.average(datara))
        raerror2.append(np.std(datara))

        rasip2.append(np.average(datarasip))
        rasiperror2.append(np.std(datarasip))

    print(len(pa),len(ra),len(rasip))

    print(min(pa),min(ra),min(rasip))
    print(np.c_[pa,np.log10(pa)])
    import matplotlib.pyplot as plt
    plt.rc('ytick',labelsize=14)
    fig, (ax1, ax2,ax3) = plt.subplots(3, 1, sharey=True,figsize=(21,20))

    # p1 = ax1.bar(X111, np.log2(pa), width,color='#900C3F', yerr=paerror,align='center',  ecolor='black', capsize=5)
    # p2 = ax1.bar(X111+width, np.log2(ra), width,color='#FF5733',yerr=raerror,align='center',ecolor='black', capsize=5)
    # p3 = ax1.bar(X111+2*width, np.log2(rasip), width,color='#FFC300',yerr=rasiperror,align='center', alpha=0.5, ecolor='black', capsize=5)

    p1 = ax1.bar(X111, np.log10(pa), width,color='#900C3F')
    p2 = ax1.bar(X111+width, np.log10(ra), width,color='#FF5733')
    p3 = ax1.bar(X111+2*width, np.log10(rasip), width,color='#FFC300')

    p1 = ax2.bar(X111, np.log10(pa1), width,color='#900C3F')
    p2 = ax2.bar(X111+width, np.log10(ra1), width,color='#FF5733')
    p3 = ax2.bar(X111+2*width, np.log10(rasip1), width,color='#FFC300')

    p1 = ax2.bar(X111, np.log10(pa2), width,color='#900C3F')
    p2 = ax2.bar(X111+width, np.log10(ra2), width,color='#FF5733')
    p3 = ax2.bar(X111+2*width, np.log10(rasip2), width,color='#FFC300')

    ax1.legend((p1[0], p2[0],p3[0]), ('Polynomial Approx. ', 'Algorithm \\ref{ALG:MVVandQR}','Algorithm \\ref{A:Polyak}'),loc = 'lower left',fontsize = 15)
    ax2.legend((p1[0], p2[0],p3[0]), ('Polynomial Approx. ', 'Algorithm \\ref{ALG:MVVandQR}','Algorithm \\ref{A:Polyak}'),loc = 'lower left',fontsize = 15)
    ax3.legend((p1[0], p2[0],p3[0]), ('Polynomial Approx. ', 'Algorithm \\ref{ALG:MVVandQR}','Algorithm \\ref{A:Polyak}'),loc = 'lower left',fontsize = 15)

    ax1.set_xticks(X111 + 2*width / 2)
    ax2.set_xticks(X111 + 2*width / 2)
    ax3.set_xticks(X111 + 2*width / 2)
    xlab = []
    for f in fff:
        print(f)
        xlab.append("\\ref{fn:%s}"%(f))
    print(xlab)

    ax1.set_xticklabels(xlab,fontsize = 14)
    ax2.set_xticklabels(xlab,fontsize = 14)
    ax3.set_xticklabels(xlab,fontsize = 14)

    ax3.set_xlabel('Function No.',fontsize = 17)
    # ax2.set_xlabel('Function No.',fontsize = 17)
    ax1.set_ylabel('$log_{10}(\\Delta_r)$',fontsize = 17)
    ax2.set_ylabel('$log_{10}(\\Delta_r)$',fontsize = 17)
    ax3.set_ylabel('$log_{10}(\\Delta_r)$',fontsize = 17)
    # ax1.set_ylim((-10,10))
    # ax2.set_ylabel('$\\Delta_r$',fontsize = 17)

    # ax.set_ylim([-9,4])
    plt.tight_layout()
    plt.show()
    # plt.savefig("plots/Perrorbars.pgf", bbox_inches="tight")






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
    ploterrorbars()
###########
