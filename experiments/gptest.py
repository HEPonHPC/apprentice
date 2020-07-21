import argparse
import pandas as pd
import numpy as np
import json
import os,sys

def predict(GP,multitestfiles,RAFOLD,OUTDIR):
    Nsample = 1000
    seed = 326323
    allMC = []
    allDeltaMC = []
    X = None
    for fno, file in enumerate(multitestfiles):
        dataperfile = pd.read_csv(file, header=None)
        Dperfile = dataperfile.values
        if fno == 0: X = Dperfile[:, :-2]
        allMC.append(Dperfile[:, -2].tolist())
        allDeltaMC.append(Dperfile[:, -1].tolist())

    distrCompare = {'ks': {}, 'kl': {}}

    bestparamfileForRA = GP.printRAmetrics(RAFOLD)
    print("\n\n\n\n")

    # RESULTS
    print("################ RESULTS START HERE")
    with open(GP.bestparamfile, 'r') as f:
        ds = json.load(f)

    if 'buildtype' in ds:
        buildtype = ds['buildtype']
    else:
        print("Buildtype not in ds not implemented")
        sys.exit(1)

    if buildtype != "gp":
        Ymean, Ysd = GP.predictHeteroscedastic(X)
    else:
        Ymean, Ysd = GP.predictHomoscedastic(X)

    allchi2metric = []
    allmeanmsemetric = []
    allsdmsemetric = []
    for j, (mu, sd) in enumerate(zip(Ymean, Ysd)):
        MCatp = [allMC[i][j] for i in range(len(allMC))]
        allchi2metric.append(((mu - np.mean(MCatp)) / sd) ** 2)
        allmeanmsemetric.append((mu - np.mean(MCatp)) ** 2)
        allsdmsemetric.append((sd - np.std(MCatp)) ** 2)

    chi2metric = np.mean(allchi2metric)
    meanmsemetric = np.mean(allmeanmsemetric)
    sdmsemetric = np.mean(allsdmsemetric)

    with open(GP.bestparamfile, 'r') as f:
        ds = json.load(f)
    bestkernel = ds['kernel']
    print("Best Kernel is {}".format(bestkernel))
    print("with meanmsemetric %.2E" % (meanmsemetric))
    print("with sdmsemetric %.2E" % (sdmsemetric))
    print("with chi2metric %.2E" % (chi2metric))

    np.random.seed(seed)

    distrCompare['ks']['MCvs{}'.format(buildtype)] = []
    distrCompare['kl']['MCvs{}'.format(buildtype)] = []
    for j,(mu,sd) in enumerate(zip(Ymean,Ysd)):
        MCatp = [allMC[i][j] for i in range(len(allMC))]
        distrCompare['ks']['MCvs{}'.format(buildtype)].append(
            computeKSstatistic(MCatp, np.random.normal(mu, sd, Nsample))
        )
        distrCompare['kl']['MCvs{}'.format(buildtype)].append(
            computeKLdivergence(MCatp, np.random.normal(mu, sd, Nsample))
        )

    ############################################
    # print(X)
    # print(Ymean)
    # print(Ysd)
    os.makedirs(OUTDIR,exist_ok=True)
    datatdump = np.column_stack((X,Ymean,Ysd))
    np.savetxt(os.path.join(OUTDIR,"{}.csv".format(ds["obsname"])), datatdump, delimiter=',')
    ############################################

    if bestparamfileForRA is not None:
        import apprentice
        OUTDIRRA = os.path.dirname(bestparamfileForRA)
        with open(bestparamfileForRA, 'r') as f:
            ds = json.load(f)
        seed = ds['seed']
        Moutfile = os.path.join(OUTDIRRA, 'RA', "{}_MCRA_S{}.json".format(GP.obsname.replace('/', '_'),
                                                                        seed))
        DeltaMoutfile = os.path.join(OUTDIRRA, 'RA', "{}_DeltaMCRA_S{}.json".format(GP.obsname.replace('/', '_'),
                                                                        seed))

        meanappset = apprentice.appset.AppSet(Moutfile, binids=[GP.obsname])
        if len(meanappset._binids) != 1 or \
                meanappset._binids[0] != GP.obsname:
            print("Something went wrong.\n"
                  "RA Fold Mean function could not be created.")
            exit(1)
        meanerrappset = apprentice.appset.AppSet(DeltaMoutfile, binids=[GP.obsname])
        if len(meanerrappset._binids) != 1 or \
                meanerrappset._binids[0] != GP.obsname:
            print("Something went wrong.\n"
                  "RA Fold Error mean function could not be created.")
            exit(1)

        Mte = np.array([meanappset.vals(x)[0] for x in X])
        DeltaMte = np.array([meanerrappset.vals(x)[0] for x in X])

    else:
        Mte = np.array([GP.approxmeancountval(x) for x in X])
        DeltaMte = np.array([GP.errapproxmeancountval(x) for x in X])

    allchi2metricRA = []
    allmeanmsemetricRA = []
    allsdmsemetricRA = []
    for j, (mu, sd) in enumerate(zip(Mte, DeltaMte)):
        MCatp = [allMC[i][j] for i in range(len(allMC))]
        allchi2metricRA.append(((mu - np.mean(MCatp)) / sd) ** 2)
        allmeanmsemetricRA.append((mu - np.mean(MCatp)) ** 2)
        allsdmsemetricRA.append((sd - np.std(MCatp)) ** 2)
    chi2metricRA = np.mean(allchi2metricRA)
    meanmsemetricRA = np.mean(allmeanmsemetricRA)
    sdmsemetricRA = np.mean(allsdmsemetricRA)

    print("RAMEAN (meanmsemetric_RA) is %.2E" % (meanmsemetricRA))
    print("RAMEAN (sdmsemetric_RA) is %.2E" % (sdmsemetricRA))
    print("RAMEAN (chi2metric_RA) is %.2E" % (chi2metricRA))

    np.random.seed(seed)
    distrCompare['ks']['MCvsRA'] = []
    distrCompare['kl']['MCvsRA'] = []
    for j, (mu, sd) in enumerate(zip(Mte, DeltaMte)):
        MCatp = [allMC[i][j] for i in range(len(allMC))]
        distrCompare['ks']['MCvsRA'].append(
            computeKSstatistic(MCatp, np.random.normal(mu, sd, Nsample))
        )
        distrCompare['kl']['MCvsRA'].append(
            computeKLdivergence(MCatp, np.random.normal(mu, sd, Nsample))
        )

    np.random.seed(seed)
    distrCompare['ks']['RAvs{}'.format(buildtype)] = \
        [computeKSstatistic(np.random.normal(mu1, sd1, Nsample),
                            np.random.normal(mu2, sd2, Nsample))
                            for (mu1, mu2, sd1, sd2) in
                            zip(Mte, Ymean,DeltaMte,Ysd)]
    np.random.seed(seed)
    distrCompare['kl']['RAvs{}'.format(buildtype)] = \
        [computeKLdivergence(np.random.normal(mu1, sd1, Nsample),
                            np.random.normal(mu2, sd2, Nsample))
                             for (mu1, mu2, sd1, sd2) in
                             zip(Mte, Ymean, DeltaMte, Ysd)]

    ############################################
    # Print best metrics into a json file
    ############################################
    bestmetricdata = {
        'RA':{
            'meanmsemetric' : meanmsemetricRA,
            'chi2metric': chi2metricRA,
            'sdmsemetric': sdmsemetricRA
        },
        buildtype:{
            'meanmsemetric': meanmsemetric,
            'chi2metric': chi2metric,
            'sdmsemetric': sdmsemetric,
            'bestkernel':bestkernel
        },
        'distrCompare': distrCompare

    }
    bestmetricfile = os.path.join(OUTDIR,"{}_bestmetrics.json".format(ds["obsname"]))
    with open(bestmetricfile, 'w') as f:
        json.dump(bestmetricdata, f, indent=4)
    ############################################

def computeKSstatistic(D1,D2):
    # n1, n2 = len(D1), len(D2)
    # mu1, mu2 = np.mean(D1), np.mean(D2)
    # sd1, sd2 = np.std(D1), np.std(D2)
    # return (mu1 - mu2) / np.sqrt((sd1 ** 2 / n1) + (sd2 ** 2 / n2))
    from scipy import stats
    return stats.ks_2samp(D1, D2)

def computeKLdivergence(D1,D2):
    from scipy.stats import entropy
    n1,n2 = len(D1),len(D2)
    if n1>n2:
        return entropy(D1[:len(D2)], D2)
    else:
        return entropy(D1, D2[:len(D1)])

class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baysian Optimal Experimental Design for Model Fitting',
                                     formatter_class=SaneFormatter)
    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument("-a", "--approx", dest="APPROX", type=str, default=None,required=True,
                        help="Polynomial/Rational approximation file over MC mean counts\n"
                             "REQUIRED argument")
    requiredNamed.add_argument("-d", "--datafile", dest="DATAFILE", type=str, default=None, required=True,
                        help="MC Data file where the first n-2 columns are X, (n-1)st "
                             "column is MC and nth column is DeltaMC\n"
                             "REQUIRED argument")

    parser.add_argument("-p", "--paramfile", dest="PARAMFILES", type=str, default=[], nargs='+',
                        help="Parameter and Xinfo JSON file (s).")
    parser.add_argument("-t", "--testfiles", dest="TESTFILES", type=str, nargs='+', default=None,
                        help="Multiple test parameter files, MC, \Delta MC values in CSV")
    parser.add_argument("-m", "--metric", dest="METRIC", type=str, default="meanmsemetric",
                        choices=["meanmsemetric", "sdmsemetric", "chi2metric"],
                        help="Metric based on which to select the best GP parameters on.")

    parser.add_argument("-o", "--outdir", dest="OUTDIR", type=str, default=None,
                        help="Output Directory")
    parser.add_argument("--dorafold", dest="DORAFOLD", default=False, action="store_true",
                        help="results of the rational approx from the best fold will be printed"
                        )

    parser.add_argument("-v", "--debug", dest="DEBUG", action="store_true", default=False,
                        help="Turn on some debug messages")



    args = parser.parse_args()
    print(args)
    from apprentice import GaussianProcess
    GP = GaussianProcess(
        DATAFILE=args.DATAFILE,
        BUILDTYPE='savedparams',
        APPROX=args.APPROX,
        PARAMFILES=args.PARAMFILES,
        METRIC = args.METRIC,
        DEBUG = args.DEBUG
    )

    predict(GP, args.TESTFILES, args.DORAFOLD,args.OUTDIR)





