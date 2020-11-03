import numpy as np
import os,json
import apprentice as app
from apprentice.appset import AppSet
def plotresults(args):
    assert (os.path.isfile(args.DATA))
    datajsonfile = os.path.join(args.INDIR,"Xaxis_data.json")
    XaxisDict = {}
    type1 = os.path.basename(args.INDIR)
    if os.path.exists(datajsonfile):
        with open(datajsonfile,'r') as f:
            XaxisDict = json.load(f)
    else:
        DATA, binids, pnames, rankIdx, xmin, xmax = app.io.readInputDataH5(args.DATA, args.WEIGHTS)
        for MN in ["3,0","3,1"]:
            XaxisDict[MN] = None
            folder = os.path.join(args.INDIR,MN)
            nseeds = 0
            for file in os.listdir(folder):
                if "testdata" in file:
                    nseeds += 1
                    with open(os.path.join(folder,file), 'r') as f:
                        tdata = json.load(f)
                    seed = tdata['seed']
                    teindex = tdata['teindex']

                    appfile = os.path.join(folder, "val_{}.json".format(seed))
                    AS = AppSet(appfile)
                    minbound = [a + 0.01 * a for a in AS._bounds[:, 0]]
                    maxbound = [a - 0.01 * a for a in AS._bounds[:, 1]]
                    if XaxisDict[MN] is None:
                        XaxisDict[MN] = np.zeros(len(AS._binids),dtype=np.float)


                    for num, (XD, YD, ED) in enumerate(DATA):
                        thisBinId = binids[num]
                        if thisBinId not in AS._binids:
                            continue

                        XtefromData = XD[teindex]
                        YtefromData = YD[teindex]
                        EtefromData = ED[teindex]
                        Xte = []
                        Yte = []
                        Ete = []
                        for X, Y, E in zip(XtefromData, YtefromData, EtefromData):
                            add = True
                            for xno, x in enumerate(X):
                                if x < minbound[xno] or x > maxbound[xno]:
                                    add = False
                                    break
                            if add:
                                Xte.append(X)
                                Yte.append(Y)
                                Ete.append(E)
                            #     for xno, x in enumerate(X):
                            #         print(x >= minbound[xno], minbound[xno],x, maxbound[xno], x <= maxbound[xno])
                            # print(len(XtefromData),len(Xte))
                        binno = np.where(AS._binids == thisBinId)
                        Y = np.array([AS.vals(x,sel=binno[0])[0] for x in Xte])
                        MSE = np.mean((Y-Yte)**2)
                        XaxisDict[MN][binno[0][0]] += MSE
            XaxisDict[MN] /= nseeds
            XaxisDict[MN] = XaxisDict[MN].tolist()

        with open (datajsonfile,'w') as f:
            json.dump(XaxisDict,f,indent=4)

    # Plotting
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rc('text', usetex=False)
    mpl.rc('font', family='serif', size=12)
    mpl.style.use("ggplot")
    colors = [
        (0.1216, 0.4667, 0.7059),
        (0.9961, 0.4980, 0.0588),
        (0.1804, 0.6235, 0.1686),
        (0.8392, 0.1490, 0.1569),
        (0.5804, 0.4039, 0.7412),
        (0.5490, 0.3373, 0.2980),
        'yellow'
    ]
    marker = ['-', '--', '-.', ':']
    for mno,MN in enumerate(["3,0", "3,1"]):
        Yaxis = np.arange(1 / len(XaxisDict[MN]), 1, 1 / len(XaxisDict[MN]))
        if len(Yaxis) !=len(XaxisDict[MN]):
            Yaxis = np.append(Yaxis, 1)
        ylabel = 'Fraction of bins (# {})'.format(len(Yaxis))
        label = "(m,n) = ({})".format(MN)
        Xaxis  = np.sort(XaxisDict[MN])

        plt.step(Xaxis, Yaxis, where='mid', label=label, linewidth=1.2,
                 c=colors[mno % len(colors)],
                 linestyle="%s" % (marker[mno % len(marker)]))
        size = 20
        plt.xticks(fontsize=size - 6)
        plt.yticks(fontsize=size - 6)
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plt.xlabel('Mean Squared Error', fontsize=size)
        plt.ylabel(ylabel, fontsize=size)
        plt.legend(fontsize=size - 4)
        plt.title(type1,fontsize=size)
    os.makedirs(args.OUTDIR, exist_ok=True)
    file = os.path.join(args.OUTDIR, '{}_approximation_mse_perf.pdf'.format(args.OFILEPREFIX))
    plt.savefig(file)
    plt.close("all")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test and plot cross validation result',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--indir", dest="INDIR", type=str, default=None,
                        help="Input directory with cross validation results")
    parser.add_argument("-o", "--outtdir", dest="OUTDIR", type=str, default=None,
                        help="Output Dir for the plot file")
    parser.add_argument("-x", "--ofileprefix", dest="OFILEPREFIX", type=str, default="",
                        help="Output file prefix (No spaces)")
    parser.add_argument("-d", "--data", dest="DATA", type=str, default="",
                        help="H5 Simulation data")
    parser.add_argument("-w", dest="WEIGHTS", default=None,
                        help="Obervable file (default: %default)")


    args = parser.parse_args()
    plotresults(args)
