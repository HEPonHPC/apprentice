import argparse
import pandas as pd
import numpy as np
import json
import os,sys

def predict(GP,multitestfiles,RAFOLD,OUTDIR):
    Nsample = 30
    seed = 326323
    allMC = []
    allDeltaMC = []
    X = None

    dc_keys = ['ks2','ks','ad2','ad','kl']
    dc_fns = [computeKS2Sample,computeKS,computeAD2Sample,computeAD,computeKLdivergence]
    distrCompare = {}
    for key in dc_keys:
        distrCompare[key] = {}

    if len(multitestfiles) >1 : raise Exception("Multiple test files not compatible")

    for fno, file in enumerate(multitestfiles):
        dataperfile = pd.read_csv(file, header=None)
        Dperfile = dataperfile.values
        if fno == 0: X = Dperfile[:, :-2]
        allMC.append(Dperfile[:, -2].tolist())
        allDeltaMC.append(Dperfile[:, -1].tolist())

    bestparamfileForRA = GP.printRAmetrics(RAFOLD)
    print("\n\n\n\n")

    # RESULTS
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
    # for j, (mu, sd) in enumerate(zip(Ymean, Ysd)):
    #     MCatp = [allMC[i][j] for i in range(len(allMC))]
    #     allchi2metric.append(((mu - np.mean(MCatp)) / sd) ** 2)
    #     allmeanmsemetric.append((mu - np.mean(MCatp)) ** 2)
    #     allsdmsemetric.append((sd - np.std(MCatp)) ** 2)


    for j, (mu, sd) in enumerate(zip(Ymean, Ysd)):
        MCatp = allMC[0][j]
        DeltaMCatp = allDeltaMC[0][j]
        allchi2metric.append(((mu - MCatp) / sd) ** 2)
        allmeanmsemetric.append((mu - MCatp) ** 2)
        allsdmsemetric.append((sd - DeltaMCatp) ** 2)


    chi2metric = np.mean(allchi2metric)
    meanmsemetric = np.mean(allmeanmsemetric)
    sdmsemetric = np.mean(allsdmsemetric)

    print("#########################")
    for kno,key in enumerate(dc_keys):
        distrCompare[key]['MCvs{}'.format(buildtype)] = []
        distrCompare[key]['MCvs{}'.format(buildtype)] = []
        for j,(mu,sd) in enumerate(zip(Ymean,Ysd)):
            MCatp = allMC[0][j]
            DeltaMCatp = allDeltaMC[0][j]
            data = np.random.normal(MCatp, DeltaMCatp, Nsample)
            distrCompare[key]['MCvs{}'.format(buildtype)].append(
                dc_fns[kno](data,mu,sd,seed)
            )
    print("#########################\n\n")

    print("################ RESULTS START HERE")
    with open(GP.bestparamfile, 'r') as f:
        ds = json.load(f)
    bestkernel = ds['kernel']
    print("Best Kernel is {}".format(bestkernel))
    print("with meanmsemetric %.2E" % (meanmsemetric))
    print("with sdmsemetric %.2E" % (sdmsemetric))
    print("with chi2metric %.2E" % (chi2metric))

    ############################################
    # print(X)
    # print(Ymean)
    # print(Ysd)
    os.makedirs(OUTDIR, exist_ok=True)
    datatdump = np.column_stack((X, Ymean, Ysd))
    np.savetxt(os.path.join(OUTDIR, "{}.csv".format(ds["obsname"])), datatdump, delimiter=',')
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
    # for j, (mu, sd) in enumerate(zip(Mte, DeltaMte)):
    #     MCatp = [allMC[i][j] for i in range(len(allMC))]
    #     allchi2metricRA.append(((mu - np.mean(MCatp)) / sd) ** 2)
    #     allmeanmsemetricRA.append((mu - np.mean(MCatp)) ** 2)
    #     allsdmsemetricRA.append((sd - np.std(MCatp)) ** 2)

    for j, (mu, sd) in enumerate(zip(Mte, DeltaMte)):
        MCatp = allMC[0][j]
        DeltaMCatp = allDeltaMC[0][j]
        allchi2metricRA.append(((mu - MCatp) / sd) ** 2)
        allmeanmsemetricRA.append((mu - MCatp) ** 2)
        allsdmsemetricRA.append((sd - DeltaMCatp) ** 2)
    chi2metricRA = np.mean(allchi2metricRA)
    meanmsemetricRA = np.mean(allmeanmsemetricRA)
    sdmsemetricRA = np.mean(allsdmsemetricRA)

    print("RAMEAN (meanmsemetric_RA) is %.2E" % (meanmsemetricRA))
    print("RAMEAN (sdmsemetric_RA) is %.2E" % (sdmsemetricRA))
    print("RAMEAN (chi2metric_RA) is %.2E" % (chi2metricRA))

    print("\n\n#########################")
    for kno,key in enumerate(dc_keys):
        distrCompare[key]['MCvsRA'] = []
        distrCompare[key]['MCvsRA'] = []
        for j, (mu, sd) in enumerate(zip(Mte, DeltaMte)):
            MCatp = allMC[0][j]
            DeltaMCatp = allDeltaMC[0][j]
            data = np.random.normal(MCatp, DeltaMCatp, Nsample)
            distrCompare[key]['MCvsRA'].append(
                dc_fns[kno](data,mu,sd,seed)
            )
    print("#########################")

    # np.random.seed(seed)
    # distrCompare['ks']['RAvs{}'.format(buildtype)] = \
    #     [computeKSstatistic(np.random.normal(mu1, sd1, Nsample),
    #                         np.random.normal(mu2, sd2, Nsample))
    #                         for (mu1, mu2, sd1, sd2) in
    #                         zip(Mte, Ymean,DeltaMte,Ysd)]
    # np.random.seed(seed)
    # distrCompare['kl']['RAvs{}'.format(buildtype)] = \
    #     [computeKLdivergence(np.random.normal(mu1, sd1, Nsample),
    #                         np.random.normal(mu2, sd2, Nsample))
    #                          for (mu1, mu2, sd1, sd2) in
    #                          zip(Mte, Ymean, DeltaMte, Ysd)]

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
        'distrCompare': distrCompare,
        "Nsample":Nsample

    }
    bestmetricfile = os.path.join(OUTDIR,"{}_bestmetrics.json".format(ds["obsname"]))
    with open(bestmetricfile, 'w') as f:
        json.dump(bestmetricdata, f, indent=4)
    ############################################

    import scipy.stats as stats
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    plotoutdir = os.path.join(OUTDIR,'plots','QQplot')
    os.makedirs(plotoutdir,exist_ok=True)
    for j, (gpmu, gpsd, ramu, rasd) in enumerate(zip(Ymean, Ysd, Mte, DeltaMte)):
        MCatp = allMC[0][j]
        DeltaMCatp = allDeltaMC[0][j]
        MCdata = np.random.normal(MCatp, DeltaMCatp, 1000)
        RAdata = np.random.normal(ramu, rasd, 1000)
        GPdata = np.random.normal(gpmu, gpsd, 1000)
        fig = plt.figure()
        plt.style.use('seaborn')
        ax = fig.add_subplot(1, 1, 1)
        sm.qqplot_2samples(MCdata, RAdata, line='45',ax=ax)
        ax.get_lines()[0].set_markerfacecolor('blue')
        ax.get_lines()[0].set_label('RA')
        sm.qqplot_2samples(MCdata, GPdata, line='45',ax=ax)
        ax.get_lines()[2].set_markerfacecolor('green')
        ax.get_lines()[2].set_label('GP')

        ax.set_xlabel('MC')
        ax.set_ylabel('')

        plt.legend(loc='best')
        fig.tight_layout()
        plotfilename = os.path.join(plotoutdir, "qqplot_{}.pdf".format(j))
        plt.savefig(plotfilename)
        # plt.show()
        plt.close('all')




def computeKS(data,mu,sd,seed):
    np.random.seed(seed)
    from skgof import ks_test
    from scipy.stats import norm
    res = ks_test(data, norm(loc=mu,scale=sd))
    return [res.statistic,res.pvalue]


def computeAD(data,mu,sd,seed):
    np.random.seed(seed)
    from skgof import ad_test
    from scipy.stats import norm, anderson
    res = ad_test(data, norm(loc=mu,scale=sd))
    res2 = anderson(data, 'norm')
    return [res.statistic, res.pvalue,res2.critical_values.tolist()]

def computeAD2Sample(data,mu,sd,seed):
    Nsample = len(data)
    np.random.seed(seed)
    otherdata = np.random.normal(mu, sd, Nsample)
    from scipy import stats
    res = stats.anderson_ksamp((data, otherdata))
    return [res.statistic, res.significance_level,res.critical_values.tolist()]

def computeKS2Sample(data,mu,sd,seed):
    Nsample = len(data)
    np.random.seed(seed)
    otherdata = np.random.normal(mu,sd,Nsample)
    from scipy import stats
    return stats.ks_2samp(data, otherdata)

def computeKLdivergence(data,mu,sd,seed):
    Nsample = len(data)
    np.random.seed(seed)
    otherdata = np.random.normal(mu, sd, Nsample)
    from scipy.stats import entropy
    return entropy(data, otherdata)

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





