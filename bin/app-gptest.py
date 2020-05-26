import os,sys
import matplotlib.pyplot as plt
import numpy as np
import argparse

def testmodel(GP,model,paramsavefile):
    import json
    with open(paramsavefile, 'r') as f:
        ds = json.load(f)

    if ds['keepout'] == 0:
        print("keepout is set to 0. No test data available. Quitting Now!")
        exit(1)
    seed = ds['seed']
    np.random.seed(seed)
    Xtrindex = ds['Xtrindex']
    Xteindex = np.in1d(np.arange(GP.nens), Xtrindex)
    Xte = GP.X[~Xteindex, :]
    ntest = len(Xte)
    MCte = GP.MC[~Xteindex]
    DeltaMCte = GP.DeltaMC[~Xteindex]
    ybar, vy = model.predict(Xte)
    predmean = np.array([y[0] for y in ybar])
    predvar = np.array([y[0] for y in vy])
    predsd = np.sqrt(predvar)
    print(predsd)
    M = np.array([GP.approxmeancountval(x) for x in Xte])
    predmean += M
    KLarr = []
    JSarr = []
    from scipy.stats import entropy
    from scipy.spatial.distance import jensenshannon
    for pm, psd, mcm, mcsd in zip(predmean, predsd, MCte, DeltaMCte):
        mcsample = np.random.normal(mcm, mcsd, 100)
        predsample = np.random.normal(pm, psd, 100)
        KLarr.append(entropy(mcsample, predsample))
        JSarr.append(jensenshannon(mcsample, predsample))

    mse1 = np.mean((predmean - MCte) ** 2)
    mse2 = np.mean((M - MCte) ** 2)
    print("MSE1 predmean {}".format(mse1))
    print("MSE2 ramean {}".format(mse2))
    Ns = ds['Ns']
    # Xterepeat = np.repeat(self.X[~Xteindex, :], [Ns] * ntest, axis=0)
    MCterepeat = np.repeat(GP.MC[~Xteindex], Ns)
    DeltaMCterepeat = np.repeat(GP.DeltaMC[~Xteindex], Ns)
    ykj = np.random.normal(MCterepeat, DeltaMCterepeat)
    predmeanrepeat = np.repeat(predmean, Ns)
    predsdrepeat = np.repeat(predsd, Ns)
    skj = np.random.normal(predmeanrepeat, predsdrepeat)
    mse3 = np.mean((ykj - skj) ** 2)
    print("MSE1 distr sample mean3 {}".format(mse3))
    mse3ink = []
    for k in range(ntest):
        mse3ink.append(np.mean((ykj[k * Ns:k * Ns + Ns] - skj[k * Ns:k * Ns + Ns]) ** 2))
    print("MSE1 distr sample mean3 test {}".format(np.mean(mse3ink)))
    print("MSE predvar {}".format(np.mean((predvar - DeltaMCte) ** 2)))

    print("\nKL Divergence:")
    kl = np.mean(KLarr)
    print(kl)
    print("\nJS Dist:")
    jsd = np.mean(JSarr)
    print(jsd)

    # import matplotlib.pyplot as plt
    # plt.plot(MCte, predmean, ls='', marker='.')
    # plt.plot(np.linspace(min(MCte), max(MCte), 100),
    #          np.linspace(min(MCte), max(MCte), 100))
    # plt.fill_between(np.linspace(min(MCte), max(MCte), ntest),
    #                  np.linspace(min(MCte),max(MCte),ntest) + 2 * predsd,
    #                  np.linspace(min(MCte),max(MCte),ntest) - 2 * predsd,
    #                  color='gray', alpha=.5)
    # plt.xlabel('MC mean count')
    # plt.ylabel('predicted mean count')
    # plt.show()

    return mse1,mse2,mse3,kl,jsd

def getGPmodelFromSavedParamFile(GP,paramfile):
    oldparamfile = GP.paramsavefile
    GP.paramsavefile = paramfile
    model = GP.buildGPmodelFromSavedParam()
    GP.paramsavefile = oldparamfile
    return model

def printresults(GP,paramarr):
    msepredmeandict, meanradict, meansampledict,kldict,jsddict = ({},{},{},{},{})
    for pno, pfile in enumerate(paramarr):
        model = getGPmodelFromSavedParamFile(GP,pfile)
        msepredmean, meanra, meansample,kl,jsd =  testmodel(GP,model,pfile)
        msepredmeandict[pfile] =  msepredmean
        meanradict[pfile] =  meanra
        meansampledict[pfile] =  meansample
        kldict[pfile] =  kl
        jsddict[pfile] =  jsd

    print("mse predmean")
    for k, v in sorted(msepredmeandict.items(), key=lambda item: item[1]):
        print("%.8E %s"%(v,os.path.basename(k)))

    print("kl divergence")
    for k, v in sorted(kldict.items(), key=lambda item: item[1]):
        print("%.8E %s"%(v,os.path.basename(k)))

    print("js distance")
    for k, v in sorted(jsddict.items(), key=lambda item: item[1]):
        print("%.8E %s"%(v,os.path.basename(k)))

    print("mse sample")
    for k, v in sorted(meansampledict.items(), key=lambda item: item[1]):
        print("%.8E %s"%(v,os.path.basename(k)))

    print("mse ra")
    for k, v in sorted(meanradict.items(), key=lambda item: item[1]):
        print("%.8E %s"%(v,os.path.basename(k)))



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
    requiredNamed.add_argument("-t", "--testtype", dest="BUILDTYPE", type=str, default="savedparam", required=True,
                        choices=["savedparam"],
                        help="Build type\n"
                             "REQUIRED argument")
    requiredNamed.add_argument("-d", "--datafile", dest="DATAFILE", type=str, default=None, required=True,
                        help="MC Data file where the first n-2 columns are X, (n-1)st "
                             "column is MC and nth column is DeltaMC\n"
                             "REQUIRED argument")

    requiredNamed.add_argument("-p", "--paramfile", dest="PARAMFILES", type=str, default=[], nargs='+', required=True,
                               help="Parameter and Xinfo JSON file.\n"
                                    "REQUIRED only build type (\"-b\", \"--buildtype\") is \"savedparam\"")


    args = parser.parse_args()
    print(args)
    from apprentice.GP import GaussianProcess
    GP = GaussianProcess(
        DATAFILE=args.DATAFILE,
        BUILDTYPE=args.BUILDTYPE,
        APPROX=args.APPROX,
        PARAMFILE=args.PARAMFILES[0]
    )
    printresults(GP,args.PARAMFILES)



