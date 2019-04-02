import numpy as np
from apprentice import RationalApproximationSIP, PolynomialApproximation
from sklearn.model_selection import KFold
from apprentice import tools, readData
import os
from mpl_toolkits.mplot3d import Axes3D
import os, sys

def getactualdegree(f):
    deg = ""
    m = 0
    n = 0
    if(f=="f1"):
        deg = "(-,4)"
        m=0
        n=4
    elif(f=="f4"):
        deg = "(-,-)"
        m=0
        n=0
    elif(f=="f7"):
        deg = "(3,3)"
        m=3
        n=3
    elif(f=="f8"):
        deg = "(2,2)"
        m=2
        n=2
    elif(f=="f9"):
        deg = "(4,4)"
        m=4
        n=4
    elif(f=="f10"):
        deg = "(2,2)"
        m=2
        n=2
    elif(f=="f12"):
        deg = "(2,3)"
        m=2
        n=3
    elif(f=="f13"):
        deg = "(3,2)"
        m=3
        n=2
    elif(f=="f14"):
        deg = "(4,4)"
        m=4
        n=4
    elif(f=="f15"):
        deg = "(3,4)"
        m=3
        n=4
    elif(f=="f16"):
        deg = "(4,3)"
        m=4
        n=3
    elif(f=="f22"):
        deg = "(2,0)"
        m=2
        n=0
    else:
        deg = "N/A"
        m=0
        n=0
    return deg,m,n

def getParetoOrderStr(data, pareto,orders):
    ret = ""
    print(pareto)
    for pnum,p in enumerate(pareto):
        for dnum,d in enumerate(data):
            if(p[0] == d[0] and p[1] == d[1]):
                if(pnum != 0 and pnum%5 == 0):
                    ret+=",\\\\"
                elif(ret!=""):
                    ret+=", "
                ret+="(%d, %d)"%(orders[dnum][0],orders[dnum][1])
    if(len(pareto)>5):
        ret = "\\makecell{"+ret+"}"
    return ret


def printables(farr, ts):
    import json
    noise = ["0","_noisepct10-6","_noisepct10-3","_noisepct10-1"]

    data={}
    for ns in noise:
        data[ns] = ""

    for f in farr:
        for ns in noise:
            print(noise)
            print(ns)
            data[ns]+="\\ref{fn:%s}&"%(f)
            noisestr = ""
            if(ns != "0"):
                noisestr = ns
            folder = "%s%s_%s"%(f,noisestr,ts)
            if not os.path.exists(folder):
                print("Folder '{}' not found.".format(folder))
                sys.exit(1)

            optdegjson = "%s/plots/Joptdeg_%s%s_jsdump_opt6.json"%(folder,f,noisestr)

            if not os.path.exists(optdegjson):
                print("Optimal degree file '{}' not found.".format(optdegjson))
                sys.exit(1)

            if optdegjson:
                with open(optdegjson, 'r') as fn:
                    optdegds = json.load(fn)

            print(optdegjson)
            dat = optdegds['data']
            ptf = optdegds['pareto']
            ord = optdegds['orders']
            data[ns]+="%s&"%(getParetoOrderStr(dat,ptf,ord))
            deg,m,n = getactualdegree(f)
            data[ns]+="%s&"%(deg)
            data[ns]+="%s&"%(optdegds['optdeg']['str'])
            lowestl2index = np.inf
            lowestl2 = np.inf
            for num, d in enumerate(dat):
                if d[1] < lowestl2:
                    lowestl2index = num
                    lowestl2 = d[1]
            data[ns]+="(%d, %d)\\\\\\hline\n"%(ord[lowestl2index][0],ord[lowestl2index][1])

    for ns in noise:
        print ("%s\n"%(ns))
        print(data[ns])



def plotoptdegreecomparison(farr, ts):

    import json
    dimarr = np.array([])
    real = np.array([])
    optdegnonoise = np.array([])
    optdeg10_1noise = np.array([])
    optdeg10_3noise = np.array([])
    optdeg10_6noise = np.array([])

    realstr = np.array([])
    optdegnonoisestr = np.array([])
    optdeg10_1noisestr = np.array([])
    optdeg10_3noisestr = np.array([])
    optdeg10_6noisestr = np.array([])

    for f in farr:
        """
        nonoise
        """
        folder = "%s_%s"%(f,ts)
        if not os.path.exists(folder):
            print("Folder '{}' not found.".format(folder))
            sys.exit(1)

        optdegjson = "%s/plots/Joptdeg_%s_jsdump_opt6.json"%(folder,f)

        if not os.path.exists(optdegjson):
            print("Optimal degree file '{}' not found.".format(optdegjson))
            sys.exit(1)

        if optdegjson:
            with open(optdegjson, 'r') as fn:
                optdegds = json.load(fn)

        dim = optdegds['dim']
        dimarr = np.append(dimarr,dim)
        m = optdegds['optdeg']['m']
        n = optdegds['optdeg']['n']
        degstr = optdegds['optdeg']['str']

        ncoeffs = tools.numCoeffsPoly(dim,m) + tools.numCoeffsPoly(dim, n)
        optdegnonoise = np.append(optdegnonoise,ncoeffs)
        optdegnonoisestr = np.append(optdegnonoisestr,degstr)

        """
        Real
        """
        deg,m,n = getactualdegree(f)
        ncoeffs = tools.numCoeffsPoly(dim,m) + tools.numCoeffsPoly(dim, n)
        real  = np.append(real,ncoeffs)
        realstr = np.append(realstr,deg)

        """
        10-6 noise
        """
        noisestr = "noisepct10-6"
        folder = "%s_%s_%s"%(f,noisestr,ts)
        if not os.path.exists(folder):
            print("Folder '{}' not found.".format(folder))
            sys.exit(1)

        optdegjson = "%s/plots/Joptdeg_%s_%s_jsdump_opt6.json"%(folder,f,noisestr)

        if not os.path.exists(optdegjson):
            print("Optimal degree file '{}' not found.".format(optdegjson))
            sys.exit(1)

        if optdegjson:
            with open(optdegjson, 'r') as fn:
                optdegds = json.load(fn)

        dim = optdegds['dim']
        m = optdegds['optdeg']['m']
        n = optdegds['optdeg']['n']
        degstr = optdegds['optdeg']['str']

        ncoeffs = tools.numCoeffsPoly(dim,m) + tools.numCoeffsPoly(dim, n)
        optdeg10_6noise = np.append(optdeg10_6noise,ncoeffs)
        optdeg10_6noisestr = np.append(optdeg10_6noisestr,degstr)

        """
        10-3 noise
        """
        noisestr = "noisepct10-3"
        folder = "%s_%s_%s"%(f,noisestr,ts)
        if not os.path.exists(folder):
            print("Folder '{}' not found.".format(folder))
            sys.exit(1)

        optdegjson = "%s/plots/Joptdeg_%s_%s_jsdump_opt6.json"%(folder,f,noisestr)

        if not os.path.exists(optdegjson):
            print("Optimal degree file '{}' not found.".format(optdegjson))
            sys.exit(1)

        if optdegjson:
            with open(optdegjson, 'r') as fn:
                optdegds = json.load(fn)

        dim = optdegds['dim']
        m = optdegds['optdeg']['m']
        n = optdegds['optdeg']['n']
        degstr = optdegds['optdeg']['str']

        ncoeffs = tools.numCoeffsPoly(dim,m) + tools.numCoeffsPoly(dim, n)
        optdeg10_3noise = np.append(optdeg10_3noise,ncoeffs)
        optdeg10_3noisestr = np.append(optdeg10_3noisestr,degstr)

        """
        10-1 noise
        """
        noisestr = "noisepct10-1"
        folder = "%s_%s_%s"%(f,noisestr,ts)
        if not os.path.exists(folder):
            print("Folder '{}' not found.".format(folder))
            sys.exit(1)

        optdegjson = "%s/plots/Joptdeg_%s_%s_jsdump_opt6.json"%(folder,f,noisestr)

        if not os.path.exists(optdegjson):
            print("Optimal degree file '{}' not found.".format(optdegjson))
            sys.exit(1)

        if optdegjson:
            with open(optdegjson, 'r') as fn:
                optdegds = json.load(fn)

        dim = optdegds['dim']
        m = optdegds['optdeg']['m']
        n = optdegds['optdeg']['n']
        degstr = optdegds['optdeg']['str']

        ncoeffs  = tools.numCoeffsPoly(dim,m) + tools.numCoeffsPoly(dim, n)
        optdeg10_1noise = np.append(optdeg10_1noise,ncoeffs)
        optdeg10_1noisestr = np.append(optdeg10_1noisestr,degstr)
    # '#E6E9ED'"#900C3F", '#C70039', '#FF5733', '#FFC300'

    act = np.zeros(len(real))
    noise0 = np.zeros(len(real))
    noise1 = np.zeros(len(real))
    noise3 = np.zeros(len(real))
    noise6 = np.zeros(len(real))


    index = 0
    yaxislabels = [""]
    # sdimarr = np.argsort(dimarr)
    # currdim = -1
    data = []
    strarr = []
    indarr = []
    for currind,f in enumerate(farr):
        indarr.append(currind)
        data.append(real[currind])
        data.append(optdegnonoise[currind])
        data.append(optdeg10_6noise[currind])
        data.append(optdeg10_3noise[currind])
        data.append(optdeg10_1noise[currind])
        strarr.append(realstr[currind])
        strarr.append(optdegnonoisestr[currind])
        strarr.append(optdeg10_6noisestr[currind])
        strarr.append(optdeg10_3noisestr[currind])
        strarr.append(optdeg10_1noisestr[currind])
    print(data)
    sdata = np.argsort(data)
    currdeg = ""
    for s in sdata:
        if(currdeg != strarr[s]):
            index += 1
            currdeg = strarr[s]
            yaxislabels.append(strarr[s])
        data[s] = index

        i=0
        j=0
        while i<len(data):
            act[indarr[j]] = data[i]
            i+=1
            noise0[indarr[j]] = data[i]
            i+=1
            noise6[indarr[j]] = data[i]
            i+=1
            noise3[indarr[j]] = data[i]
            i+=1
            noise1[indarr[j]] = data[i]
            i+=1
            j+=1


    print(yaxislabels)
    print(act)
    print(noise0)
    print(noise6)
    print(noise3)
    print(noise1)

    import matplotlib as mpl
    mpl.use('pgf')
    pgf_with_custom_preamble = {
        "text.usetex": True,    # use inline math for ticks
        "pgf.rcfonts": False,   # don't setup fonts from rc parameters
        "pgf.preamble": [
            "\\usepackage{amsmath}",         # load additional packages
        ]
    }
    mpl.rcParams.update(pgf_with_custom_preamble)
    import matplotlib.pyplot as plt

    width = 0.15
    fig, ax = plt.subplots(figsize=(15,10))

    X = np.arange(len(farr))
    # p1 = ax.bar(X, np.log10(real), width, color='gray')
    # p2 = ax.bar(X+width, np.log10(optdegnonoise), width, color='#900C3F')
    # p3 = ax.bar(X+2*width, np.log10(optdeg10_6noise), width, color='#C70039')
    # p4 = ax.bar(X+3*width, np.log10(optdeg10_3noise), width, color='#FF5733')
    # p5 = ax.bar(X+4*width, np.log10(optdeg10_1noise), width, color='#FFC300')

    p1 = ax.bar(X, act, width, color='gray')
    p2 = ax.bar(X+width, noise0, width, color='#900C3F')
    p3 = ax.bar(X+2*width, noise6, width, color='#C70039')
    p4 = ax.bar(X+3*width, noise3, width, color='#FF5733')
    p5 = ax.bar(X+4*width, noise1, width, color='#FFC300')

    # for num,p in enumerate(p1.patches):
    #     h = p.get_height()
    #     x = p.get_x()+p.get_width()/2.
    #     print(h,x)
    #     ax.annotate("%s"%(realstr[num]), xy=(x,h), xytext=(0,4), rotation=90, fontsize=8,
    #                textcoords="offset points", ha="center", va="bottom")
    #
    # for num,p in enumerate(p2.patches):
    #     h = p.get_height()
    #     x = p.get_x()+p.get_width()/2.
    #     print(h,x)
    #     ax.annotate("%s"%(optdegnonoisestr[num]), xy=(x,h), xytext=(0,4), rotation=90, fontsize=8,
    #                textcoords="offset points", ha="center", va="bottom")
    #
    # for num,p in enumerate(p3.patches):
    #     h = p.get_height()
    #     x = p.get_x()+p.get_width()/2.
    #     print(h,x)
    #     ax.annotate("%s"%(optdeg10_6noisestr[num]), xy=(x,h), xytext=(0,4), rotation=90, fontsize=8,
    #                textcoords="offset points", ha="center", va="bottom")
    #
    # for num,p in enumerate(p4.patches):
    #     h = p.get_height()
    #     x = p.get_x()+p.get_width()/2.
    #     print(h,x)
    #     ax.annotate("%s"%(optdeg10_3noisestr[num]), xy=(x,h), xytext=(0,4), rotation=90, fontsize=8,
    #                textcoords="offset points", ha="center", va="bottom")

    # for num,p in enumerate(p5.patches):
    #     h = p.get_height()
    #     x = p.get_x()+p.get_width()/2.
    #     print(h,x)
    #     ax.annotate("%s"%(optdeg10_1noisestr[num]), xy=(x,h), xytext=(0,4), rotation=90, fontsize=8,
    #                textcoords="offset points", ha="center", va="bottom")
    ax.legend((p1[0], p2[0],p3[0],p4[0],p5[0]), ('Actual orders','$\\epsilon=0$', '$\\epsilon=10^{-6}$','$\\epsilon=10^{-3}$', '$\\epsilon=10^{-1}$'))
    # ax.set_title('Comparing optimal degree obtained for different types of training data with \'%s\' points'%(ts))
    ax.set_xticks(X + 4*width / 2)
    ax.set_yticks(range(0,index+1))
    xlab = []
    for f in farr:
        xlab.append("\\ref{fn:%s}"%(f))
    ax.set_xticklabels(xlab)
    ax.set_yticklabels(yaxislabels)
    ax.set_xlabel('Function No.')
    ax.set_ylabel('Order of numerator and denominator polynomials')
    # ax.autoscale_view()
    # plt.show()
    if not os.path.exists("plots"):
        os.mkdir('plots')

    outfilepng = "plots/Popdegcmp_%s_%s_ts%s_optimaldegreecomparison.png"%(farr[0],farr[len(farr)-1],ts)

    # plt.show()
    # plt.savefig(outfilepng)
    plt.savefig("plots/Poptdegcomparebarplots.pgf", bbox_inches="tight")

# python plot2Dsurface.py f21_2x/out/f21_2x_p12_q12_ts2x.json ../benchmarkdata/f21_test.txt f21_2x f21_2x all

def plotoptdegreecompsubplots(farr, noisearr,ts):
    import json
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.text as text
    # mpl.use('pgf')
    pgf_with_custom_preamble = {
        "text.usetex": True,    # use inline math for ticks
        "pgf.rcfonts": False,   # don't setup fonts from rc parameters
        "pgf.preamble": [
            "\\usepackage{amsmath}",         # load additional packages
        ]
    }
    mpl.rcParams.update(pgf_with_custom_preamble)
    lx="$\\log_{10}(\\eta_r)$"
    ly="$\\log_{10}(\\Delta_{r})$"
    logy=True
    logx=True
    f, axarr = plt.subplots(2,2, figsize=(15,8))
    f.subplots_adjust(hspace=0.3)
    # f.subplots_adjust(wspace=0.3)
    paretotxt = ""
    # f.subplots_adjust(wspace=-.5)
    for num, fname in enumerate(farr):
        row = int(num/2)
        col = num%2
        noise = noisearr[num]
        noisestr = ""
        if(noise!="0"):
            noisestr = "_noisepct"+noise
        folder = "%s%s_%s"%(fname,noisestr,ts)
        optjsonfile = folder+"/plots/Joptdeg_"+fname+noisestr+"_jsdump_opt6.json"

        if not os.path.exists(optjsonfile):
            print("optjsonfile: " + optjsonfile+ " not found")
            exit(1)

        if optjsonfile:
            with open(optjsonfile, 'r') as fn:
                optjsondatastore = json.load(fn)

        pareto = optjsondatastore['pareto']
        txt = optjsondatastore['text']
        data = optjsondatastore['data']
        orders = optjsondatastore['orders']

        axarr[row][col].set_xlabel(lx)
        axarr[row][col].set_ylabel(ly)
        if logx: axarr[row][col].set_xscale("log")
        if logy: axarr[row][col].set_yscale("log")

        c = []
        # paretopoint=[]
        for num, (m,n) in enumerate(orders):
            if n==0:
                c.append("b")
            else:
                c.append("r")

        marker,size = [],[]
        cornerindex = -1

        for num, t in enumerate(txt):
            if(t==optjsondatastore['optdeg']['str']):
                cornerindex = num
            if(t!=""):
                marker.append('o')
                size.append(50)
            else:
                marker.append('x')
                size.append(15)

        lowestl2index = np.inf
        lowestl2 = np.inf
        largestl2 = 0
        for num, d in enumerate(data):
            if d[1] < lowestl2:
                lowestl2index = num
                lowestl2 = d[1]
            if(d[1] > largestl2):
                largestl2 = d[1]



        for num, d in enumerate(data):
            if(num==cornerindex):
                axarr[row][col].scatter(d[0], d[1], marker = '*', c = "peru"  ,s=444, alpha = 1)
            if(num==lowestl2index):
                axarr[row][col].scatter(d[0], d[1], marker = 'x', c = "purple"  ,s=222, alpha = 1)
            axarr[row][col].scatter(d[0], d[1],c=c[num],marker=marker[num],s=size[num])

        axarr[row][col].set_title("\\textbf{Function No. \\ref{fn:%s}}"%(fname),fontweight='bold')
        paretotxt +="\\ref{fn:%s}&"%(fname)
        for num, t in enumerate(txt):
            # axarr[row][col].text(data[num][0]-data[num][0]/(num+1), data[num][1], t, fontsize=8,verticalalignment='center')
            if(t!=""):
                paretotxt += "%s"%(t)
                if(num != len(txt)-1):
                    paretotxt += ", "
        paretotxt += "\\\\hline\n"

    print(paretotxt)


    for ax in axarr.flat:
        ax.tick_params(axis = 'both', which = 'major')
        ax.tick_params(axis = 'both', which = 'minor')
        # ax.label_outer()


    # plt.show()
    plt.savefig("plots/Poptdegcomparesubplots.pgf", bbox_inches="tight")


if __name__ == "__main__":

    if len(sys.argv)!=5:
        print("Usage: {} functions noise ts strategy=[barplot,subplot,table]".format(sys.argv[0]))
        sys.exit(1)

    farr = sys.argv[1].split(',')
    if len(farr) == 0:
        print("please specify comma saperated functions")
        sys.exit(1)

    noisearr = sys.argv[2].split(',')
    if len(noisearr) == 0:
        print("please specify comma saperated noise. If \"barplot or table\", arg not used, add junk val")
        sys.exit(1)

    strategy = sys.argv[4]
    if(strategy == 'table'):
        printables(farr, sys.argv[3])
    elif(strategy == 'barplot'):
        plotoptdegreecomparison(farr, sys.argv[3])
    elif(strategy == 'subplot'):
        if(len(farr)!=len(noisearr)):
            print("functions and noise should have same vals")
            sys.exit(1)
        plotoptdegreecompsubplots(farr,noisearr, sys.argv[3])
###########
