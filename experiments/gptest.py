import argparse
import pandas as pd
import numpy as np
import json
import os,sys

def predict(GP,testfile,RAFOLD,OUTDIR):
    data = pd.read_csv(testfile, header=None)
    D = data.values
    X = D[:, :-2]
    MC = D[:, -2]
    DeltaMC = D[:, -1]

    bestparamfileForRA = GP.printRAmetrics(RAFOLD)
    print("\n\n\n\n")

    # RESULTS
    print("################ RESULTS START HERE")
    with open(GP.bestparamfile, 'r') as f:
        ds = json.load(f)
    if 'buildtype' not in ds or ds['buildtype'] != "gp":
        Ymean, Ysd = GP.predictHeteroscedastic(X)
    else:
        Ymean, Ysd = GP.predictHomoscedastic(X)
    buildtype = ""
    if 'buildtype' in ds:
        buildtype = ds['buildtype']
    else:
        print("Buildtype not in ds not implemented")
        sys.exit(1)
    chi2metric = np.mean(((Ymean - MC) / Ysd) ** 2)
    meanmsemetric = np.mean((Ymean - MC) ** 2)
    sdmsemetric = np.mean((Ysd - DeltaMC) ** 2)
    with open(GP.bestparamfile, 'r') as f:
        ds = json.load(f)
    print("Best Kernel is {}".format(ds['kernel']))
    print("with meanmsemetric %.2E" % (meanmsemetric))
    print("with sdmsemetric %.2E" % (sdmsemetric))
    print("with chi2metric %.2E" % (chi2metric))

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
    meanmsemetricRA = np.mean((Mte - MC) ** 2)
    sdmsemetricRA = np.mean((DeltaMte - DeltaMC) ** 2)
    chi2metricRA = np.mean(((Mte - MC) / DeltaMC) ** 2)
    print("RAMEAN (meanmsemetric_RA) is %.2E" % (meanmsemetricRA))
    print("RAMEAN (sdmsemetric_RA) is %.2E" % (sdmsemetricRA))
    print("RAMEAN (chi2metric_RA) is %.2E" % (chi2metricRA))

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
            'sdmsemetric': sdmsemetric
        }
    }
    bestmetricfile = os.path.join(OUTDIR,"{}_bestmetrics.json".format(ds["obsname"]))
    with open(bestmetricfile, 'w') as f:
        json.dump(bestmetricdata, f, indent=4)
    ############################################

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
    parser.add_argument("-t", "--testfile", dest="TESTFILE", type=str,
                        help="Test parameters, MC, \Delta MC values in CSV")
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

    predict(GP, args.TESTFILE, args.DORAFOLD,args.OUTDIR)





