import os
import matplotlib.pyplot as plt
import GPy
import numpy as np
import pandas as pd
import sys
import datetime
from timeit import default_timer as timer
import argparse


class GaussianProcess():
    def __init__(self, *args, **kwargs):
        datafile = kwargs['DATAFILE']
        data = pd.read_csv(datafile,header=None)
        D = data.values
        self.X = D[:, :-2]
        self.MC = D[:, -2]
        self.DeltaMC = D[:, -1]
        self.nens,self.nparam = self.X.shape
        self.buildtype = kwargs['BUILDTYPE']

        if self.buildtype=="data":
            self.kernel = kwargs['KERNEL']
            self.obsname = kwargs['OBS']
            # self.nprocess = kwargs['NPROCESS']
            self.nrestart = kwargs['NRESTART']
            self.keepout = kwargs['KEEPOUT']/100
            os.makedirs(kwargs['OUTDIR'],exist_ok=True)
            self.outfile = os.path.join(kwargs['OUTDIR'], "{}_K{}.json".format(self.obsname.replace('/', '_'), self.kernel))
        elif self.buildtype=="savedparam":
            self.paramsavefile = kwargs['PARAMFILE']
            import json
            with open(self.paramsavefile, 'r') as f:
                ds = json.load(f)
            self.obsname = ds['obsname']


        import apprentice
        self.meanappset = apprentice.appset.AppSet(kwargs['APPROX'], binids=self.obsname)
        if len(self.meanappset._binids)!=1 or \
            self.meanappset._binids[0] != self.obsname:
            print("Something went wrong.\n"
                  "Mean function could not be created.\n"
                  "This code does not support multi output GP")
            exit(1)
        if self.buildtype == 'data':
            self.model = self.buildGPmodelFromData()
        elif self.buildtype == 'savedparam':
            self.model = self.buildGPmodelFromSavedParam()

    def approxmeancountval(self, x):
        return self.meanappset.vals(x)[0]

    def approxmeancountgrad(self, x):
        return self.meanappset.grads(x)

    def getKernel(self, kernelStr, polyorder):
        kernelObj = None
        availablekernels = ["sqe","ratquad","matern32","matern52","poly"]
        if kernelStr == "sqe":
            kernelObj = GPy.kern.RBF(input_dim=self.nparam, ARD=True)
        elif kernelStr == "ratquad":
            kernelObj = GPy.kern.RatQuad(input_dim=self.nparam, ARD=True)
        elif kernelStr == "matern32":
            kernelObj = GPy.kern.Matern32(input_dim=self.nparam, ARD=True)
        elif kernelStr == "matern52":
            kernelObj = GPy.kern.Matern52(input_dim=self.nparam, ARD=True)
        elif kernelStr == "poly":
            kernelObj = GPy.kern.Poly(input_dim=self.nparam, order=polyorder)
        elif kernelStr == "or":
            kernelObj = self.getKernel(availablekernels[0],polyorder)
            for i in range(1,len(availablekernels)):
                kernelObj += self.getKernel(availablekernels[i],polyorder)
        else:
            print("Kernel {} unknown. Quitting now!".format(kernelStr))
            exit(1)
        return kernelObj


    def buildGPmodelFromData(self):
        import json
        Ntr = int((1-self.keepout) *self.nens)
        Ns = 25
        seed = 992739462
        np.random.seed(seed)
        Xtrindex = np.random.choice(np.arange(self.nens), Ntr, replace=False)
        Xtr = np.repeat(self.X[Xtrindex, :], [Ns] * len(Xtrindex), axis=0)
        MCtr = np.repeat(self.MC[Xtrindex], Ns)
        DeltaMCtr = np.repeat(self.DeltaMC[Xtrindex], Ns)

        # Get Ns samples of each of the Ntr training distribution
        Ytr = np.random.normal(MCtr, DeltaMCtr)
        M = [self.approxmeancountval(x) for x in Xtr]
        # Y-M (Training labels)
        Ytrmm = Ytr - M
        Ytrmm2D = np.array([Ytrmm]).transpose()

        # Homoscedastic noise (for now) that we will find during parameter tuning
        lik = GPy.likelihoods.Gaussian()

        polyorder = None
        if self.kernel in ['poly','or']:
            polyorder = 3

        kernelObj = self.getKernel(self.kernel,polyorder)

        # 0 mean GP to model f
        model = GPy.core.GP(Xtr,
                            Ytrmm2D,
                            kernel=kernelObj,
                            likelihood=lik
                            )

        print(model.likelihood.variance)
        print(model.kern.parameters)
        print(model.kern.param_array)
        print(model.param_array.tolist())
        start = timer()
        print("##############################")
        print(datetime.datetime.now())
        print("##############################")
        sys.stdout.flush()

        # if self.nprocess > 1:
        #     print("Something is wrong with parallel runs. FIX required\nQuitting for now")
        #     sys.exit(1)
        #     model.optimize_restarts(robust=True,
        #                             parallel=True,
        #                             # messages=True,
        #                             num_processes=self.nprocess,
        #                             num_restarts=self.nrestart
        #                             )
        #
        # else:

        model.optimize()
        model.optimize_restarts(num_restarts=self.nrestart,
                                robust=True)
        print(timer() - start)
        print(model.likelihood.variance)
        print(model.kern.param_array)
        print(model.param_array.tolist())
        print(model)
        print("##############################")
        print(datetime.datetime.now())
        print("##############################")
        sys.stdout.flush()

        data = {
            "savedmodelparams":model.param_array.tolist(),
            "Ntr": Ntr,
            "Ns": Ns,
            'seed': seed,
            'kernel':self.kernel,
            "keepout":self.keepout*100,
            "obsname":self.obsname,
            "Xtrindex": Xtrindex.tolist(),
            "Ytrmm": Ytrmm.tolist()
        }
        if self.kernel in ['poly','or']:
            data['polyorder'] = polyorder
        with open(self.outfile, 'w') as f:
            json.dump(data, f, indent=4)

        return model

    def  buildGPmodelFromSavedParam(self):
        import json
        with open(self.paramsavefile, 'r') as f:
            ds = json.load(f)
        Ns = ds['Ns']
        seed = ds['seed']
        np.random.seed(seed)
        Xtrindex = ds['Xtrindex']
        Xtr = np.repeat(self.X[Xtrindex, :], [Ns] * len(Xtrindex), axis=0)
        Ytrmm = ds['Ytrmm']
        Ytrmm2D = np.array([Ytrmm]).transpose()
        kernel = ds['kernel']

        # Homoscedastic noise (for now) that we will find during parameter tuning
        lik = GPy.likelihoods.Gaussian()
        polyorder = None
        if kernel in ['poly', 'or']:
            polyorder = ds['polyorder']
        kernelObj = self.getKernel(kernel,polyorder)

        # 0 mean GP to model f
        model = GPy.core.GP(Xtr,
                            Ytrmm2D,
                            kernel=kernelObj,
                            likelihood=lik,
                            )
        model.update_model(False)
        model.initialize_parameter()
        model[:] = ds['savedmodelparams']
        model.update_model(True)
        return model

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
    requiredNamed.add_argument("-b", "--buildtype", dest="BUILDTYPE", type=str, default="data", required=True,
                        choices=["data","savedparam"],
                        help="Build type\n"
                             "REQUIRED argument")
    requiredNamed.add_argument("-d", "--datafile", dest="DATAFILE", type=str, default=None, required=True,
                        help="MC Data file where the first n-2 columns are X, (n-1)st "
                             "column is MC and nth column is DeltaMC\n"
                             "REQUIRED argument")

    parser.add_argument("--obsname", dest="OBS", type=str, default=None,
                        help="Observable Name\n"
                        "REQUIRED only build type (\"-b\", \"--buildtype\") is \"data\"")
    parser.add_argument("--keepout", dest="KEEPOUT", type=float, default=20.,
                        help="Percentage in \[0,100\] of the data to be left out as test data. \n"
                             "Train on the (100-keepout) percent data and then test on the rest. \n"
                             "REQUIRED only build type (\"-b\", \"--buildtype\") is \"data\"")
    parser.add_argument("-o", "--outdir", dest="OUTDIR", type=str, default=None,
                        help="Output Directory \n"
                             "REQUIRED only build type (\"-b\", \"--buildtype\") is \"data\"")
    # parser.add_argument("--nprocess", dest="NPROCESS", type=int, default=1,
    #                     help="Number of processes to use in optimization. "
    #                          "If >1, parallel version of optmimize used \n"
    #                          "REQUIRED only build type (\"-b\", \"--buildtype\") is \"data\"")
    parser.add_argument("--nrestart", dest="NRESTART", type=int, default=1,
                        help="Number of optimization restarts (multistart)\n"
                             "REQUIRED only build type (\"-b\", \"--buildtype\") is \"data\"")
    parser.add_argument("-k", "--kernel", dest="KERNEL", type=str, default="sqe",
                               choices=["matern32", "matern52","sqe","ratquad","poly","or"],
                               help="Kernel to use (ARD will be set to True for all (where applicable)\n"
                                    "REQUIRED only build type (\"-b\", \"--buildtype\") is \"data\"")


    parser.add_argument("-p", "--paramfile", dest="PARAMFILE", type=str, default=None,
                        help="Parameter and Xinfo JSON file.\n"
                             "REQUIRED only build type (\"-b\", \"--buildtype\") is \"savedparam\"")



    args = parser.parse_args()
    print(args)
    GP = GaussianProcess(
        DATAFILE=args.DATAFILE,
        BUILDTYPE=args.BUILDTYPE,
        KERNEL=args.KERNEL,
        OBS=args.OBS,
        # NPROCESS=args.NPROCESS,
        NRESTART=args.NRESTART,
        KEEPOUT=args.KEEPOUT,
        OUTDIR=args.OUTDIR,
        APPROX=args.APPROX,
        PARAMFILE=args.PARAMFILE
    )



