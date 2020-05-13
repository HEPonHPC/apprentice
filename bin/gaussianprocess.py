import numpy as np
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
    def __init__(self, *args, scargs, **kwargs):
        infile = scargs.INFILE
        data = pd.read_csv(infile,header=None)
        D = data.values
        self.X = D[:, :-2]
        self.MC = D[:, -2]
        self.DeltaMC = D[:, -1]
        self.nens,self.nparam = self.X.shape
        import apprentice
        self.meanappset = apprentice.appset.AppSet(scargs.APPROX,binids=scargs.OBS)
        self.outdir = scargs.OUTDIR
        self.obsname =scargs.OBS
        self.nprocess = scargs.NPROCESS
        self.nrestart = scargs.NRESTART
        if len(self.meanappset._binids)!=1 and \
            self.meanappset._binids[0] != scargs.OBS:
            print("Something went wrong.\n"
                  "Mean function could not be created.\n"
                  "This code does not support multi output GP")
            exit(1)

    def approxmeancountval(self, x):
        return self.meanappset.vals(x)[0]

    def approxmeancountgrad(self, x):
        return self.meanappset.grads(x)

    def buildGPmodel(self):
        Ntr = 10
        Ns = 25
        seed = 992739462
        np.random.seed(seed)
        Xtrindex = np.random.choice(np.arange(self.nens), Ntr, replace=False)
        Xtr = np.repeat(self.X[Xtrindex, :],[Ns]*len(Xtrindex),axis=0)
        MCtr = np.repeat(self.MC[Xtrindex],Ns)
        DeltaMCtr = np.repeat(self.DeltaMC[Xtrindex], Ns)

        # Get Ns samples of each of the Ntr training distribution
        Ytr = np.random.normal(MCtr,DeltaMCtr)
        M = [self.approxmeancountval(x) for x in Xtr]
        # Y-M (Training labels)
        Ytrmm = Ytr-M
        Ytrmm2D = np.array([Ytrmm]).transpose()

        # Homoscedastic noise (for now) that we will find during parameter tuning
        lik = GPy.likelihoods.Gaussian()
        kernel = GPy.kern.RBF(input_dim=self.nparam,ARD=True)

        # 0 mean GP to model f
        model = GPy.core.GP(Xtr,
                        Ytrmm2D,
                        kernel=kernel,
                        likelihood=lik
        )

        print(model.likelihood.variance)
        print(model.kern.parameters)
        start=timer()
        print("##############################")
        print(datetime.datetime.now())
        print("##############################")
        sys.stdout.flush()

        if self.nprocess >1:
            model.optimize_restarts(num_restarts=self.nrestart,
                                    robust=True,
                                    parallel=True,
                                    num_processes=self.nprocess)
        else:
            model.optimize_restarts(num_restarts=self.nrestart,
                                    robust=True)
        print(timer()-start)
        print(model.likelihood.variance)
        print(model.kern.parameters)
        print(model)
        print("##############################")
        print(datetime.datetime.now())
        print("##############################")
        sys.stdout.flush()
        dir = os.path.join(self.outdir,"ParamSave")
        os.makedirs(dir,exist_ok=True)
        savefn = os.path.join(dir,"{}.npy".format(self.obsname.replace("/","_")))
        np.save(savefn, model.param_array)
        dir = os.path.join(self.outdir, "XinfoSave")
        os.makedirs(dir, exist_ok=True)
        savefn = os.path.join(dir, "{}.json".format(self.obsname.replace("/", "_")))
        data ={"Ntr":Ntr,"Ns":Ns,'seed':seed,"Xtrindex":Xtrindex.tolist(),"Ytrmm":Ytrmm.tolist()}
        import json
        with open(savefn, 'w') as f:
            json.dump(data, f, indent=4)
        return model

        # Xteindex = np.in1d(np.arange(self.nens), Xtrindex)
        # Xte = self.X[~Xteindex, :]
        # MCte = self.MC[~Xteindex]
        # DeltaMCte =self.DeltaMC[~Xteindex]
        # Ntest = len(MCte)

class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baysian Optimal Experimental Design for Model Fitting',
                                     formatter_class=SaneFormatter)
    parser.add_argument("-i", "--infile", dest="INFILE", type=str, default=None,
                        help="In file where the first n-2 columns are X, (n-1)st "
                             "column is MC and nth column is DeltaMC")
    parser.add_argument("-a", "--approx", dest="APPROX", type=str, default=None,
                        help="Approximation file")
    parser.add_argument("--obsname", dest="OBS", type=str, default=None,
                        help="Observable Name")
    parser.add_argument("-o", "--outtdir", dest="OUTDIR", type=str, default=None,
                        help="Output Dir")
    parser.add_argument("--nprocess", dest="NPROCESS", type=int, default=1,
                        help="Number of processes to use in optimization. "
                             "If >1, parallel version of optmimize used")
    parser.add_argument("--nrestart", dest="NRESTART", type=int, default=1,
                        help="Number of optimization restarts (multistart)")




    args = parser.parse_args()
    print(args)
    GP = GaussianProcess(scargs=args)
    GP.buildGPmodel()


