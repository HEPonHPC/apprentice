import numpy as np
from apprentice import RationalApproximationSIP, RationalApproximationONB, PolynomialApproximation
from apprentice import tools, readData
import os


def ploterrorbars():
    import matplotlib as mpl
    import json
    mpl.use('pgf')
    pgf_with_custom_preamble = {
        "text.usetex": True,    # use inline math for ticks
        "pgf.rcfonts": False,   # don't setup fonts from rc parameters
        "pgf.preamble": [
            "\\usepackage{amsmath}",         # load additional packages
        ]
    }
    mpl.rcParams.update(pgf_with_custom_preamble)

    width = 0.15
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1,2,figsize=(15,10),sharey=True)
    pa = []
    ra = []
    rasip = []

    pa1 = []
    ra1 = []
    rasip1 = []

    fff = ['f3','f5','f8','f13','f14','f18','f19','f22']
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

        folder = "%s%s_%s"%(fname,noisestr,ts)
        optjsonfile = folder+"/plots/Joptdeg_"+fname+noisestr+"_jsdump_opt6.json"
        if not os.path.exists(optjsonfile):
            print("optjsonfile: " + optjsonfile+ " not found")
            exit(1)
        if optjsonfile:
            with open(optjsonfile, 'r') as fn:
                optjsondatastore = json.load(fn)

        optm = optjsondatastore['optdeg']['m']
        optn = optjsondatastore['optdeg']['n']
        pq = "p%d_q%d"%(optm,optn)
        print(fname,noisestr,optm,optn)
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

        pa.append(np.sqrt(np.sum((Y_pred_papp-Y_test)**2)))
        ra.append(np.sqrt(np.sum((Y_pred_rapp-Y_test)**2)))
        rasip.append(np.sqrt(np.sum((Y_pred_rappsip-Y_test)**2)))

        noisestr = "_noisepct10-1"
        folder = "%s%s_%s"%(fname,noisestr,ts)
        optjsonfile = folder+"/plots/Joptdeg_"+fname+noisestr+"_jsdump_opt6.json"
        if not os.path.exists(optjsonfile):
            print("optjsonfile: " + optjsonfile+ " not found")
            exit(1)
        if optjsonfile:
            with open(optjsonfile, 'r') as fn:
                optjsondatastore = json.load(fn)

        optm = optjsondatastore['optdeg']['m']
        optn = optjsondatastore['optdeg']['n']
        pq = "p%d_q%d"%(optm,optn)
        rappsipfile = "%s/out/%s%s_%s_%s_ts%s.json"%(folder,fname,noisestr,ts,pq,ts)
        rappfile = "%s/outra/%s%s_%s_%s_ts%s.json"%(folder,fname,noisestr,ts,pq,ts)
        pappfile = "%s/outpa/%s%s_%s_%s_ts%s.json"%(folder,fname,noisestr,ts,pq,ts)
        print(fname,noisestr,optm,optn)
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

        pa1.append(np.sqrt(np.sum((Y_pred_papp-Y_test)**2)))
        ra1.append(np.sqrt(np.sum((Y_pred_rapp-Y_test)**2)))
        rasip1.append(np.sqrt(np.sum((Y_pred_rappsip-Y_test)**2)))
    print(len(pa),len(ra),len(rasip))

    print(min(pa),min(ra),min(rasip))
    print(np.c_[pa,np.log10(pa)])
    import matplotlib.pyplot as plt
    plt.rc('ytick',labelsize=14)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(21,6))

    p1 = ax1.bar(X111, np.log10(pa), width,color='#900C3F')
    p2 = ax1.bar(X111+width, np.log10(ra), width,color='#FF5733')
    p3 = ax1.bar(X111+2*width, np.log10(rasip), width,color='#FFC300')

    p1 = ax2.bar(X111, np.log10(pa1), width,color='#900C3F')
    p2 = ax2.bar(X111+width, np.log10(ra1), width,color='#FF5733')
    p3 = ax2.bar(X111+2*width, np.log10(rasip1), width,color='#FFC300')

    ax1.legend((p1[0], p2[0],p3[0]), ('Polynomial Approx. ', 'Algorithm \\ref{ALG:MVVandQR}','Algorithm \\ref{A:Polyak}'),loc = 'lower left',fontsize = 15)
    ax2.legend((p1[0], p2[0],p3[0]), ('Polynomial Approx. ', 'Algorithm \\ref{ALG:MVVandQR}','Algorithm \\ref{A:Polyak}'),loc = 'lower left',fontsize = 15)

    ax1.set_xticks(X111 + 2*width / 2)
    ax2.set_xticks(X111 + 2*width / 2)
    xlab = []
    for f in fff:
        print(f)
        xlab.append("\\ref{fn:%s}"%(f))
    print(xlab)

    ax1.set_xticklabels(xlab,fontsize = 14)
    ax2.set_xticklabels(xlab,fontsize = 14)

    ax1.set_xlabel('Function No.',fontsize = 17)
    ax2.set_xlabel('Function No.',fontsize = 17)
    ax1.set_ylabel('$log_{10}(\\Delta_r)$',fontsize = 17)
    # ax2.set_ylabel('$\\Delta_r$',fontsize = 17)

    # ax.set_ylim([-9,4])
    plt.tight_layout()
    # plt.show()
    plt.savefig("plots/Perrorbars.pgf", bbox_inches="tight")






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
