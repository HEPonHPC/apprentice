import numpy as np
import os,json
import apprentice as app
from apprentice.appset import AppSet,TuningObjective2
def docdfplots(args):
    assert (os.path.isfile(args.DATA))
    datajsonfile = os.path.join(args.INDIR,"Yaxis_data.json")
    XaxisDict = {}
    type1 = os.path.basename(args.INDIR)
    if os.path.exists(datajsonfile):
        with open(datajsonfile,'r') as f:
            XaxisDict = json.load(f)
    else:
        ASmaster = None
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
                    if ASmaster is None:
                        ASmaster = AS
                    minbound = [a + 0.0 * a for a in AS._bounds[:, 0]]
                    maxbound = [a - 0.0 * a for a in AS._bounds[:, 1]]
                    if XaxisDict[MN] is None:
                        XaxisDict[MN] = np.zeros(len(AS._binids),dtype=np.float)


                    for num, (XD, YD, ED) in enumerate(DATA):
                        thisBinId = binids[num]
                        if thisBinId not in AS._binids:
                            continue
                        if len(teindex) != len(XD):
                            oldteindex = teindex
                            teindex = teindex[:len(XD)]
                            XtefromData = XD[teindex]
                            YtefromData = YD[teindex]
                            EtefromData = ED[teindex]
                            teindex = oldteindex
                        else:
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
            XaxisDict['hnames'] = sorted(list(set(ASmaster._hnames)))
            XaxisDict['binids'] = ASmaster._binids.tolist()

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

    hnames = XaxisDict['hnames']
    binids = XaxisDict['binids']
    catrng, names_lab, names_fn = readcategoryfile(args.CATEGORY, hnames)
    width = 0.55
    size = 20
    type1 = os.path.basename(args.INDIR)

    def obsBins(hname):
        return [i for i, item in enumerate(binids) if item.startswith(hname)]

    for mno, MN in enumerate(["3,0", "3,1"]):
        import matplotlib.pyplot as plt
        odir = os.path.join(args.OUTDIR, "MSE_{}Cat{}".format(type1, len(names_lab)), MN)
        os.makedirs(odir, exist_ok=True)
        for ano, arr in enumerate(catrng):
            Yaxis = []
            for i in arr:
                hname = hnames[i]
                sel = obsBins(hname)
                for i in sel:
                    Yaxis.append(XaxisDict[MN][i])
            Xaxis = np.arange(len(Yaxis))
            fig, ax = plt.subplots(figsize=(30, 8))
            ax.bar(Xaxis, Yaxis, width, color='blue')
            ax.set_xlabel('Bins', fontsize=24)
            ax.set_ylabel('Mean Squared Error', fontsize=24)
            ax.set_title("{} ({})".format(names_lab[ano],MN), fontsize=size)
            plt.xticks(fontsize=size - 6)
            plt.yticks(fontsize=size - 6)
            plt.yscale('log')
            plt.ylim(10 ** -9, 10 ** 3)
            plt.savefig(os.path.join(odir, "_{}_{}_{}_.pdf".format(args.OFILEPREFIX, type1, names_fn[ano])))
            plt.close('all')


def readcategoryfile(categoryfile,hnames):
    if categoryfile is None:
        rng, names_lab, names_fn = [np.arange(len(hnames))], ["all"], ["all"]
    else:
        rng, names_lab, names_fn = ([],[],[])
        import json
        import re
        with open(categoryfile,'r') as f:
            catds = json.load(f)
        for key in catds:
            names_lab.append(key)
            names_fn.append(re.sub(r"\s+", "", key))
            currrng = []
            for obs in catds[key]:
                currrng.append(hnames.index(obs))
            rng.append(currrng)
    return rng, names_lab, names_fn

def plotBinwiseDenomSignificance(args):
    allDenoms = []
    hnames = None
    binids = []
    type1 = None
    size = 20
    assert (os.path.isdir(args.INDIR))
    folder = args.INDIR
    nseeds = 0
    for file in os.listdir(folder):
        if "testdata" in file:
            nseeds += 1
            with open(os.path.join(folder, file), 'r') as f:
                tdata = json.load(f)
            seed = tdata['seed']
            appfile = os.path.join(folder, "val_{}.json".format(seed))
            AS = AppSet(appfile)
            if len(allDenoms) == 0:
                type1 = args.INDIR.split('/')[-2]
                for bin in AS._binids:
                    allDenoms.append([])
                    binids.append(bin)
                hnames = sorted(list(set(AS._hnames)))
            for bno, bin in enumerate(AS._binids):
                allDenoms[bno].append(AS._QC[bno])
    if nseeds == 0:
        type1 = os.path.basename(args.INDIR)
        approxfile = args.INDIR + "/approximation.json"
        errapproxfile = args.INDIR + "/errapproximation.json"
        expdatafile = args.INDIR + "/experimental_data.json"
        weightfile = args.INDIR + "/weights"
        from apprentice.appset import TuningObjective2
        IO = TuningObjective2(weightfile, expdatafile, approxfile, errapproxfile,
                              filter_hypothesis=False, filter_envelope=False)
        AS = IO._AS
        for bin in IO._binids:
            allDenoms.append([])
            binids.append(bin)
        hnames = sorted(list(set(IO._hnames)))
        for bno, bin in enumerate(IO._binids):
            allDenoms[bno].append(AS._QC[bno])

    significanceArr = []
    for binno, bindenom in enumerate(allDenoms):
        assert(len(bindenom) != 0)
        significance = 0
        for dno,denom in enumerate(bindenom):
            significance += sum(abs(denom[1:]))/abs(denom[0])
        significance /= len(bindenom)
        significanceArr.append(significance)

    import matplotlib.pyplot as plt
    # import matplotlib as mpl
    # mpl.rc('text', usetex=False)
    # mpl.rc('font', family='serif', size=12)
    # mpl.style.use("ggplot")

    def obsBins(hname):
        return [i for i, item in enumerate(binids) if item.startswith(hname)]

    catrng, names_lab, names_fn = readcategoryfile(args.CATEGORY, hnames)

    width = 0.55
    odir = os.path.join(args.OUTDIR,"{}Cat{}".format(type1,len(names_lab)),"n{}".format(len(allDenoms[0])))
    os.makedirs(odir,exist_ok=True)
    for ano, arr in enumerate(catrng):
        Yaxis = []
        for i in arr:
            hname = hnames[i]
            sel = obsBins(hname)
            for i in sel:
                Yaxis.append(significanceArr[i])
        Xaxis = np.arange(len(Yaxis))
        fig, ax = plt.subplots(figsize=(30, 8))
        ax.bar(Xaxis, Yaxis, width, color='blue')

        ax.set_xlabel('Bins', fontsize=24)
        ax.set_ylabel('$r(p)=\\frac{{n(p)}}{{d(p)}}, \\quad d(p)=a^Tp+b, \\quad y=\\frac{{||a||}}{{|b|}}$', fontsize=24)
        ax.set_title(names_lab[ano],fontsize=size)
        plt.xticks(fontsize=size - 6)
        plt.yticks(fontsize=size - 6)
        xlab = []
        for i in range(len(Xaxis)):
            j = i + 1
            if j == 1:
                xlab.append("1")
                continue
            if j % 100 == 0:
                xlab.append(str(j))
            else:
                xlab.append("")
        # plt.xticks(Xaxis, xlab, fontsize=24)
        plt.yscale('log')
        plt.ylim(10 ** -4, 10 ** 0)
        plt.savefig(os.path.join(odir,"_{}_{}_{}.pdf".format(args.OFILEPREFIX,type1,names_fn[ano])))

def plotBinwiseDenomRange(args):
    allDenoms = []
    bounds = None
    hnames = None
    binids = []
    type1 = None
    size = 20
    assert (os.path.isdir(args.INDIR))
    folder = args.INDIR
    nseeds = 0
    for file in os.listdir(folder):
        if "testdata" in file:
            nseeds += 1
            with open(os.path.join(folder, file), 'r') as f:
                tdata = json.load(f)
            seed = tdata['seed']
            appfile = os.path.join(folder, "val_{}.json".format(seed))
            AS = AppSet(appfile)
            if len(allDenoms) == 0:
                type1 = args.INDIR.split('/')[-2]
                for bin in AS._binids:
                    allDenoms.append([])
                    binids.append(bin)
                bounds = AS._bounds
                hnames = sorted(list(set(AS._hnames)))
            for bno, bin in enumerate(AS._binids):
                allDenoms[bno].append(AS._QC[bno])
    if nseeds == 0:
        type1 = os.path.basename(args.INDIR)
        approxfile = args.INDIR + "/approximation.json"
        errapproxfile = args.INDIR + "/errapproximation.json"
        expdatafile = args.INDIR + "/experimental_data.json"
        weightfile = args.INDIR + "/weights"
        from apprentice.appset import TuningObjective2
        IO = TuningObjective2(weightfile, expdatafile, approxfile, errapproxfile,
                              filter_hypothesis=False, filter_envelope=False)
        AS = IO._AS
        for bin in IO._binids:
            allDenoms.append([])
            binids.append(bin)
        bounds = AS._bounds
        hnames = sorted(list(set(IO._hnames)))
        for bno, bin in enumerate(IO._binids):
            allDenoms[bno].append(AS._QC[bno])

    rangeArr = []
    # print(bounds)
    for binno, bindenom in enumerate(allDenoms):
        assert(len(bindenom) != 0)
        range = 0.
        for dno,denom in enumerate(bindenom):
            a = denom[1:1+len(bounds)]
            a = a[::-1]
            minp = np.zeros(len(a),dtype=np.float)
            maxp = np.zeros(len(a),dtype=np.float)
            for ano,ai in enumerate(a):
                currb = bounds[ano]
                if ai<0:
                    minp[ano] = currb[1]
                    maxp[ano] = currb[0]
                else:
                    minp[ano] = currb[0]
                    maxp[ano] = currb[1]
            range += np.sum(a*maxp) - np.sum(a*minp)
        range /= len(bindenom)
        rangeArr.append(range)

    import matplotlib.pyplot as plt
    # import matplotlib as mpl
    # mpl.rc('text', usetex=False)
    # mpl.rc('font', family='serif', size=12)
    # mpl.style.use("ggplot")

    def obsBins(hname):
        return [i for i, item in enumerate(binids) if item.startswith(hname)]

    catrng, names_lab, names_fn = readcategoryfile(args.CATEGORY, hnames)

    width = 0.55
    odir = os.path.join(args.OUTDIR,"{}Cat{}".format(type1,len(names_lab)),"n{}".format(len(allDenoms[0])))
    os.makedirs(odir,exist_ok=True)
    for ano, arr in enumerate(catrng):
        Yaxis = []
        for i in arr:
            hname = hnames[i]
            sel = obsBins(hname)
            for i in sel:
                Yaxis.append(rangeArr[i])
        Xaxis = np.arange(len(Yaxis))
        fig, ax = plt.subplots(figsize=(30, 8))
        # print(np.shape(Xaxis),np.shape(Yaxis))
        ax.bar(Xaxis, Yaxis, width, color='blue')

        ax.set_xlabel('Bins', fontsize=24)
        ax.set_ylabel('$r(p)=\\frac{{n(p)}}{{d(p)}}, \\quad y = $range$(d(p))$', fontsize=24)
        ax.set_title(names_lab[ano],fontsize=size)
        plt.xticks(fontsize=size - 6)
        plt.yticks(fontsize=size - 6)
        xlab = []
        # for i in range(len(Xaxis)):
        #     j = i + 1
        #     if j == 1:
        #         xlab.append("1")
        #         continue
        #     if j % 100 == 0:
        #         xlab.append(str(j))
        #     else:
        #         xlab.append("")
        # plt.xticks(Xaxis, xlab, fontsize=24)
        plt.yscale('log')
        plt.ylim(10 ** -6, 10 ** 0)
        plt.savefig(os.path.join(odir,"_{}_{}_{}.pdf".format(args.OFILEPREFIX,type1,names_fn[ano])))

def plotBinwisePolyRationalComparison(args):
    def numer(RAAS, x, sel=slice(None, None, None), set_cache=True, maxorder=None):
        import apprentice
        if set_cache: RAAS.setRecurrence(x)
        if maxorder is None:
            MM=RAAS._maxrec * RAAS._PC[sel]
        else:
            nc = apprentice.tools.numCoeffsPoly(RAAS.dim, 2)
            MM=RAAS._maxrec[:nc] * RAAS._PC[sel][:,:nc]
        vals = np.sum(MM, axis=1)
        return vals

    def denom(RAAS, x, sel=slice(None, None, None), set_cache=True, maxorder=None):
        if set_cache: RAAS.setRecurrence(x)
        if RAAS._hasRationals:
            den = np.sum(RAAS._maxrec * RAAS._QC[sel], axis=1)
            return den
        return None

    assert (os.path.isdir(args.INDIR))
    assert (os.path.isfile(args.DATA))
    metrics = ["p-r_sqr by fiterr",
               "p-r_sqr by chi2",
               "p-r_sqr by fiterr+chi2",
               ]
    pmrsqrlab = "\\frac{{1}}{{N_{te}}} \\sum_{i=1}^{N_{te}} \\left(P_b(p_i)-R_b(p_i)\\right)^2"
    pfiterrlab = "\\frac{{1}}{{|P_{ens}|}} \\sum_{p \in P_{ens}} (P_b(p)-MC_b(p))^2"
    rfiterrlab = "\\frac{{1}}{{|P_{ens}|}} \\sum_{p \in P_{ens}} (R_b(p)-MC_b(p))^2"
    pchi2 = "\\frac{{1}}{{N_{te}}} \\sum_{i=1}^{N_{te}}\\frac{{(P_b(p_i)-D_b)^2}}{{\Delta P_b(p_i)^2 + \Delta D_b^2}}"
    rchi2 = "\\frac{{1}}{{N_{te}}} \\sum_{i=1}^{N_{te}}\\frac{{(R_b(p_i)-D_b)^2}}{{\Delta R_b(p_i)^2 + \Delta D_b^2}}"
    metricylab = [
        '$\\frac{{%s}}{{%s + %s}}$' % (pmrsqrlab,pfiterrlab,rfiterrlab),
        '$\\frac{{%s}}{{%s + %s}}$' % (pmrsqrlab,pchi2,rchi2),
        '$\\frac{{%s}}{{%s + %s + %s + %s}}$' % (pmrsqrlab,pfiterrlab,rfiterrlab,pchi2,rchi2)
    ]
    ylimlow = [10**-4,10**-18,10**-19]
    ylimhigh = [10**2,10**-5,10**-5]

    binwisePolyRationalComparison_fn = os.path.join(args.OUTDIR, "binwisePolyRationalComparison_data.json")
    metricData = None
    if not os.path.exists(binwisePolyRationalComparison_fn):
        DATA, dbinids, dpnames, drankIdx, dxmin, dxmax = app.io.readInputDataH5(args.DATA, args.WEIGHTS)
        polydir = args.INDIR
        radir = args.INDIR + "-RA"

        TuningObjectives = []
        for dir in [polydir, radir]:
            apprfile = os.path.join(dir, "approximation.json")
            errapprfile = os.path.join(dir, "errapproximation.json")
            datafile = os.path.join(dir, "experimental_data.json")
            wtfile = os.path.join(dir, "weights")
            TuningObjectives.append(TuningObjective2(wtfile, datafile, apprfile,
                                                     errapprfile,
                                                     filter_envelope=False,
                                                     filter_hypothesis=False
                                                      ))
        allhnames = sorted(list(set(TuningObjectives[0]._AS._hnames)))
        allbinids = TuningObjectives[0]._AS._binids
        metricData = {}
        for m in metrics:
            metricData[m] = None

        np.random.seed(873268)
        Xperdim = ()
        npoints = 1000
        for d in range(TuningObjectives[0].dim):
            Xperdim = Xperdim + (
                np.random.rand(npoints, ) * (
                        TuningObjectives[0]._AS._bounds[d, 1] - TuningObjectives[0]._AS._bounds[d, 0]) +
                TuningObjectives[0]._AS._bounds[d, 0],)
        Xte = np.column_stack(Xperdim)

        ###########
        # Fiterror
        pfiterror = np.zeros(len(allbinids), dtype=np.float)
        rfiterror = np.zeros(len(allbinids), dtype=np.float)

        for num, (XD, YD, ED) in enumerate(DATA):
            thisBinId = dbinids[num]
            if thisBinId not in allbinids:
                continue
            binno = allbinids.index(thisBinId)

            pY = np.array([TuningObjectives[0]._AS.vals(x, sel=[binno])[0] for x in XD])
            rY = np.array([TuningObjectives[1]._AS.vals(x, sel=[binno])[0] for x in XD])

            pMSE = np.mean((pY - YD) ** 2)
            rMSE = np.mean((rY - YD) ** 2)

            pfiterror[binno] += pMSE
            rfiterror[binno] += rMSE

        ###########
        # p - r sqr
        pmr_sqr = np.zeros(len(allbinids), dtype=np.float)
        for x in Xte:
            PY = TuningObjectives[0]._AS.vals(x)
            RY = TuningObjectives[1]._AS.vals(x)
            pmr_sqr += (RY - PY) ** 2
        pmr_sqr /= len(Xte)

        ###########
        # Chi2
        pchi2 = np.zeros(len(allbinids), dtype=np.float)
        rchi2 = np.zeros(len(allbinids), dtype=np.float)
        for x in Xte:
            pc = TuningObjectives[0].objective(x,unbiased=True)
            rc = TuningObjectives[1].objective(x,unbiased=True)
            pchi2 += pc
            rchi2 += rc
        pchi2 /= len(Xte)
        rchi2 /= len(Xte)

        for mno, m in enumerate(metrics):
            if metricData[m] is None:
                metricData[m] = np.zeros(len(allbinids), dtype=np.float)
            if m == "p-r_sqr by fiterr":
                metricData[m] = pmr_sqr/(pfiterror + rfiterror)
            elif m == "p-r_sqr by chi2":
                metricData[m] = pmr_sqr / (pchi2 + rchi2)
            elif m == "p-r_sqr by fiterr+chi2":
                metricData[m] = pmr_sqr / (pfiterror + rfiterror + pchi2 + rchi2)
        for key in metricData:
            metricData[key] = metricData[key].tolist()
        metricData['hnames'] = allhnames
        metricData['binids'] = allbinids
        with open(binwisePolyRationalComparison_fn,'w') as f:
            json.dump(metricData,f,indent=4)
    else:
        with open(binwisePolyRationalComparison_fn,'r') as f:
            metricData = json.load(f)

    allhnames = metricData['hnames']
    allbinids = metricData['binids']

    def obsBins(hname):
        return [i for i, item in enumerate(allbinids) if item.startswith(hname)]

    catrng, names_lab, names_fn = readcategoryfile(args.CATEGORY, allhnames)

    width = 0.55
    size = 20
    import matplotlib.pyplot as plt
    type1 = os.path.basename(args.INDIR)
    for m,mlab,yl,yh in zip(metrics,metricylab,ylimlow,ylimhigh):
        odir = os.path.join(args.OUTDIR, "{}Cat{}".format(type1, len(names_lab)), m)
        os.makedirs(odir, exist_ok=True)
        for ano, arr in enumerate(catrng):
            Yaxis = []
            for i in arr:
                hname = allhnames[i]
                sel = obsBins(hname)
                for i in sel:
                    Yaxis.append(metricData[m][i])
            Xaxis = np.arange(len(Yaxis))
            fig, ax = plt.subplots(figsize=(30, 15))
            ax.bar(Xaxis, Yaxis, width, color='blue')

            ax.set_xlabel('Bins b', fontsize=24)
            ax.set_ylabel(mlab, fontsize=24)
            ax.set_title(names_lab[ano], fontsize=size)
            plt.xticks(fontsize=size - 6)
            plt.yticks(fontsize=size - 6)
            plt.yscale('log')
            plt.ylim(yl, yh)
            # plt.show()
            plt.savefig(os.path.join(odir, "_{}_{}_{}_.pdf".format(args.OFILEPREFIX, type1, names_fn[ano])))
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
    parser.add_argument("-c", "--category", dest="CATEGORY", type=str, default=None,
                        help="Filename of the Category file. "
                             "If None, categories ignored.")
    parser.add_argument("--denomsignificance", dest="DSIGN", default=False, action="store_true",
                        help="Plot denominator significance")

    args = parser.parse_args()
    if not args.DSIGN:
        """
         data=Sherpa; python approxCrossValidationTest.py -i ../../log/ApproximationsCrossValidation/$data -o ../../log/ApproximationsCrossValidation/$data/plots -d ../../log/SimulationData/$data-h5/*.h5
         data=A14; python approxCrossValidationTest.py -i ../../log/ApproximationsCrossValidation/$data -o ../../log/ApproximationsCrossValidation/$data/plots -d ../../log/SimulationData/$data-h5/*.h5 -c ../../pyoo/data/A14Categories/A14Cat_10.json
        """
        docdfplots(args)
    else:
        """
        data=A14; python approxCrossValidationTest.py -i ../../pyoo/data/$data-RA -o ../../log/ApproximationsCrossValidation/$data/plots --denomsignificance -c ../../pyoo/data/A14Categories/A14Cat_10.json 
        data=A14; python approxCrossValidationTest.py -i ../../log/ApproximationsCrossValidation/$data/3,1 -o ../../log/ApproximationsCrossValidation/$data/plots --denomsignificance -c ../../pyoo/data/A14Categories/A14Cat_10.json
        
        data=Sherpa; python approxCrossValidationTest.py -i ../../log/ApproximationsCrossValidation/$data/3,1 -o ../../log/ApproximationsCrossValidation/$data/plots --denomsignificance   
        """
        # plotBinwiseDenomSignificance(args)

        """
        data=A14; python approxCrossValidationTest.py -i ../../pyoo/data/$data-RA -o ../../log/ApproximationsCrossValidation/$data/plots/range --denomsignificance -c ../../pyoo/data/A14Categories/A14Cat_10.json 
        data=A14; python approxCrossValidationTest.py -i ../../log/ApproximationsCrossValidation/$data/3,1 -o ../../log/ApproximationsCrossValidation/$data/plots/range --denomsignificance -c ../../pyoo/data/A14Categories/A14Cat_10.json

        data=Sherpa; python approxCrossValidationTest.py -i ../../log/ApproximationsCrossValidation/$data/3,1 -o ../../log/ApproximationsCrossValidation/$data/plots/range --denomsignificance   
                """
        # plotBinwiseDenomRange(args)

        """
        data=A14; python approxCrossValidationTest.py -i ../../pyoo/data/$data -o ../../log/ApproximationsCrossValidation/$data/plots/polyrationalcomparison --denomsignificance -d ../../log/SimulationData/$data-h5/*.h5 -c ../../pyoo/data/A14Categories/A14Cat_10.json
        data=Sherpa; python approxCrossValidationTest.py -i ../../pyoo/data/$data -o ../../log/ApproximationsCrossValidation/$data/plots/polyrationalcomparison --denomsignificance -d ../../log/SimulationData/$data-h5/*.h5
        """
        plotBinwisePolyRationalComparison(args)