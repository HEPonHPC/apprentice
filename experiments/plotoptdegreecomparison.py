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
    if(f=="f7"):
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
    else:
        deg = "N/A"
        m=0
        n=0
    return deg,m,n



def plotoptdegreecomparison(farr, ts):

    import json
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

        optdegjson = "%s/plots/Joptdeg_%s_jsdump.json"%(folder,f)

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

        optdegjson = "%s/plots/Joptdeg_%s_%s_jsdump.json"%(folder,f,noisestr)

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

        optdegjson = "%s/plots/Joptdeg_%s_%s_jsdump.json"%(folder,f,noisestr)

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

        optdegjson = "%s/plots/Joptdeg_%s_%s_jsdump.json"%(folder,f,noisestr)

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
        optdeg10_1noise = np.append(optdeg10_1noise,ncoeffs)
        optdeg10_1noisestr = np.append(optdeg10_1noisestr,degstr)
    # '#E6E9ED'"#900C3F", '#C70039', '#FF5733', '#FFC300'
    import matplotlib.pyplot as plt
    width = 0.15
    fig, ax = plt.subplots(figsize=(15,10))

    X = np.arange(len(farr))
    p1 = ax.bar(X, np.log10(real), width, color='gray')
    p2 = ax.bar(X+width, np.log10(optdegnonoise), width, color='#900C3F')
    p3 = ax.bar(X+2*width, np.log10(optdeg10_6noise), width, color='#C70039')
    p4 = ax.bar(X+3*width, np.log10(optdeg10_3noise), width, color='#FF5733')
    p5 = ax.bar(X+4*width, np.log10(optdeg10_1noise), width, color='#FFC300')

    for num,p in enumerate(p1.patches):
        h = p.get_height()
        x = p.get_x()+p.get_width()/2.
        print(h,x)
        ax.annotate("%s"%(realstr[num]), xy=(x,h), xytext=(0,4), rotation=90, fontsize=8,
                   textcoords="offset points", ha="center", va="bottom")

    for num,p in enumerate(p2.patches):
        h = p.get_height()
        x = p.get_x()+p.get_width()/2.
        print(h,x)
        ax.annotate("%s"%(optdegnonoisestr[num]), xy=(x,h), xytext=(0,4), rotation=90, fontsize=8,
                   textcoords="offset points", ha="center", va="bottom")

    for num,p in enumerate(p3.patches):
        h = p.get_height()
        x = p.get_x()+p.get_width()/2.
        print(h,x)
        ax.annotate("%s"%(optdeg10_6noisestr[num]), xy=(x,h), xytext=(0,4), rotation=90, fontsize=8,
                   textcoords="offset points", ha="center", va="bottom")

    for num,p in enumerate(p4.patches):
        h = p.get_height()
        x = p.get_x()+p.get_width()/2.
        print(h,x)
        ax.annotate("%s"%(optdeg10_3noisestr[num]), xy=(x,h), xytext=(0,4), rotation=90, fontsize=8,
                   textcoords="offset points", ha="center", va="bottom")

    for num,p in enumerate(p5.patches):
        h = p.get_height()
        x = p.get_x()+p.get_width()/2.
        print(h,x)
        ax.annotate("%s"%(optdeg10_1noisestr[num]), xy=(x,h), xytext=(0,4), rotation=90, fontsize=8,
                   textcoords="offset points", ha="center", va="bottom")
    ax.legend((p1[0], p2[0],p3[0],p4[0],p5[0]), ('Actual','No noise', 'e = $10^{-6}$','e = $10^{-3}$', 'e = $10^{-1}$'))
    ax.set_title('Comparing optimal degree obtained for different types training data with \'%s\' training data'%(ts))
    ax.set_xticks(X + 4*width / 2)
    ax.set_xticklabels(farr)
    ax.set_xlabel('Functions')
    ax.set_ylabel('$log_{10}(N_{coeff})$')
    # ax.autoscale_view()
    # plt.show()
    if not os.path.exists("plots"):
        os.mkdir('plots')

    outfilepng = "plots/Popdegcmp_%s_%s_ts%s_optimaldegreecomparison.png"%(farr[0],farr[len(farr)-1],ts)

    plt.savefig(outfilepng)

# python plot2Dsurface.py f21_2x/out/f21_2x_p12_q12_ts2x.json ../benchmarkdata/f21_test.txt f21_2x f21_2x all

if __name__ == "__main__":

    if len(sys.argv)!=3:
        print("Usage: {} functions ts".format(sys.argv[0]))
        sys.exit(1)
    farr = sys.argv[1].split(',')
    if len(farr) == 0:
        print("please specify comma saperated functions")
        sys.exit(1)
    plotoptdegreecomparison(farr, sys.argv[2])
###########
