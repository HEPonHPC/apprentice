import os
import matplotlib.pyplot as plt
import GPy
import numpy as np
import pandas as pd
import sys
import datetime
from timeit import default_timer as timer
import argparse
from mpi4py import MPI

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
        self._debug = kwargs['DEBUG']

        if self.buildtype=="data":
            self.MPITUNE = kwargs['MPITUNE']
            self.SEED  = kwargs['SEED']
            self.STOPCOND = kwargs['STOPCOND']
            self.kernel = kwargs['KERNEL']
            self.obsname = kwargs['OBS']
            self.nrestart = kwargs['NRESTART']
            self.keepout = kwargs['KEEPOUT']/100
            os.makedirs(kwargs['OUTDIR'],exist_ok=True)
            self.outfile = os.path.join(kwargs['OUTDIR'], "{}_K{}_S{}.json".format(self.obsname.replace('/', '_'),
                                                                                self.kernel,self.SEED))
        elif self.buildtype=="savedparams":
            self.paramsavefiles = kwargs['PARAMFILES']
            import json
            with open(self.paramsavefiles[0], 'r') as f:
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
            self.modely,self.modelz = self.buildGPmodelFromData()
        elif self.buildtype == 'savedparams':
            self.modely,self.modelz = self.buildGPmodelFromSavedParam()

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

    def mpitune(self,model,num_restarts,useMPI=False,robust=True):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        if not useMPI:
            model.optimize_restarts(num_restarts=num_restarts,
                                    robust=robust,verbose=self._debug)
            return model.param_array
        else:
            import apprentice
            allWork = apprentice.tools.chunkIt([i for i in range(num_restarts)], size)
            rankWork = comm.scatter(allWork, root=0)
            import time
            import sys
            import datetime
            _paramarray = np.zeros(num_restarts, dtype=object)
            _F = np.zeros(num_restarts)
            t0 = time.time()
            for ii in rankWork:
                try:
                    if ii > 0:
                        for iii in range(ii):
                            model.randomize()
                    model.optimize_restarts(num_restarts=1,robust=robust,verbose=self._debug)
                except Exception as e:
                    if robust:
                        print(("Warning - optimization restart on rank {} failed".format(ii)))
                    else:
                        raise e
                _paramarray[ii] = model.param_array
                _F[ii] = model.objective_function()

                if rank == 0 and self._debug:
                    print("[{}] {}/{}".format(rank, ii, len(rankWork)))
                    now = time.time()
                    tel = now - t0
                    ttg = tel * (len(rankWork) - ii) / (ii + 1)
                    eta = now + ttg
                    eta = datetime.datetime.fromtimestamp(now + ttg)
                    sys.stdout.write(
                        "[{}] {}/{} (elapsed: {:.1f}s, to go: {:.1f}s, ETA: {})\r".format(
                            rank, ii + 1, len(rankWork), tel, ttg, eta.strftime('%Y-%m-%d %H:%M:%S')))
                    sys.stdout.flush()
            a = comm.gather(_paramarray[rankWork])
            b = comm.gather(_F[rankWork])
            myreturnvalue = None
            if rank == 0:
                allWork = apprentice.tools.chunkIt([i for i in range(num_restarts)], size)
                for r in range(size): _paramarray[allWork[r]] = a[r]
                for r in range(size): _F[allWork[r]] = b[r]
                myreturnvalue = _paramarray[np.argmin(_F)]
                if self._debug:
                    print("Objective values from all parallel runs:")
                    print(_F)
                    # print(_paramarray)
                    sys.stdout.flush()
            myreturnvalue = comm.bcast(myreturnvalue, root=0)
            return myreturnvalue

    def buildGPmodelFromData(self):
        import json
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        Ntr = int((1-self.keepout) *self.nens)
        Ns = 25

        np.random.seed(self.SEED)

        Xtrindex = np.random.choice(np.arange(self.nens), Ntr, replace=False)
        Xtr = np.repeat(self.X[Xtrindex, :], [Ns] * len(Xtrindex), axis=0)

        MCtr = np.repeat(self.MC[Xtrindex], Ns)

        DeltaMCtr = np.repeat(self.DeltaMC[Xtrindex], Ns)
        DeltaMCtr2D = np.array([DeltaMCtr]).transpose()

        Xteindex = np.in1d(np.arange(self.nens), Xtrindex)
        Xte = self.X[~Xteindex, :]
        ntest = len(Xte)
        MCte = self.MC[~Xteindex]
        DeltaMCte = self.DeltaMC[~Xteindex]
        Mte = np.array([self.approxmeancountval(x) for x in Xte])

        # Get Ns samples of each of the Ntr training distribution
        Ytr = np.random.normal(MCtr, DeltaMCtr)
        Mtr = np.array([self.approxmeancountval(x) for x in Xtr])
        # Y-M (Training labels)
        Ytrmm = Ytr - Mtr
        Ytrmm2D = np.array([Ytrmm]).transpose()

        # Homoscedastic noise (for now) that we will find during parameter tuning
        # lik = GPy.likelihoods.Gaussian()
        start = timer()
        if rank == 0:
            print("##############################")
            print(datetime.datetime.now())
            print("##############################")
            sys.stdout.flush()

        polyorder = None
        if self.kernel in ['poly','or']:
            polyorder = 3
        if rank == 0:
            database = {
                # "savedmodelparams": model.param_array.tolist(),
                # "param_names_flat": model.parameter_names_flat().tolist(),
                # "param_names": model.parameter_names(),
                "Ntr": Ntr,
                "Ns": Ns,
                'seed': self.SEED,
                'kernel': self.kernel,
                "keepout": self.keepout * 100,
                "obsname": self.obsname,
                "Xtrindex": Xtrindex.tolist(),
                "Ytrmm": Ytrmm.tolist(),
                'modely':{},
                'modelz':{},
                'log':{'stopcond':self.STOPCOND}
            }
            if self.kernel in ['poly', 'or']:
                database['polyorder'] = polyorder

        kernelObjZero = self.getKernel(self.kernel, polyorder)
        modelzero = GPy.models.GPHeteroscedasticRegression(Xtr,
                                                       Ytrmm2D,
                                                       kernel=kernelObjZero
                                                    )
        modelzero['.*het_Gauss.variance'] =  DeltaMCtr2D
        modelzero.het_Gauss.variance.fix()
        modelzero[:] = self.mpitune(modelzero,num_restarts=self.nrestart,
                                 useMPI=self.MPITUNE,robust=True)
        modelzero.update_model(True)
        if self._debug and rank == 0:
            print(modelzero.objective_function())

        currenthetobjective = modelzero.objective_function()
        oldhetobjective = np.infty
        modely = modelzero
        iteration = 0
        bestobjectivey = np.infty
        bestmodely = None
        bestmodelz = None
        while (currenthetobjective-oldhetobjective)**2 > self.STOPCOND:
            Ybar = modely._raw_predict(Xtr)[0]
            Ymean = np.array([y[0] for y in Ybar])
            Ymean += Mtr
            Vtr = (Ytr - Ymean)**2
            Vtr = Vtr.reshape(Ntr, Ns).sum(axis=1) / Ns
            Ztr = [np.log(v) for v in Vtr]
            Ztr2D = np.array([Ztr]).transpose()
            kernelObjz = self.getKernel(self.kernel, polyorder)
            modelz = GPy.models.GPRegression(
                                self.X[Xtrindex, :],
                                Ztr2D,
                                kernel=kernelObjz,
                                normalizer = True
                                )
            modelz[:] = self.mpitune(modelz, num_restarts=self.nrestart,
                                  useMPI=self.MPITUNE, robust=True)
            modelz.update_model(True)
            if self._debug and rank == 0:
                 print(modelz.objective_function())

            Zbar,Zv = modelz.predict(Xtr)
            Zmean = np.array([z[0] for z in Zbar])
            Zvar = np.array([z[0] for z in Zv])
            Vmean = np.exp(Zmean + (Zvar / 2))
            Vmean2D = np.array([Vmean]).transpose()

            kernelObjy = self.getKernel(self.kernel, polyorder)
            modely = GPy.models.GPHeteroscedasticRegression(Xtr,
                                                            Ytrmm2D,
                                                            kernel=kernelObjy
                                                            )
            modely['.*het_Gauss.variance'] = Vmean2D
            modely.het_Gauss.variance.fix()
            modely[:] = self.mpitune(modely, num_restarts=self.nrestart,
                                  useMPI=self.MPITUNE, robust=True)
            modely.update_model(True)
            if self._debug and rank == 0:
                 print(modely.objective_function())

            oldhetobjective = currenthetobjective
            currenthetobjective = modely.objective_function()
            if bestobjectivey > modely.objective_function():
                bestobjectivey = modely.objective_function()
                bestmodely = modely
                bestmodelz = modelz

            if rank == 0:
                if iteration==0:
                    database['modely']['param_names_flat'] = modely.parameter_names_flat().tolist()
                    database['modelz']['param_names_flat'] = modelz.parameter_names_flat().tolist()
                    database['modely']["param_names"] = modely.parameter_names()
                    database['modelz']["param_names"] = modelz.parameter_names()
                    database['modely']['savedmodelparams'] = []
                    database['modelz']['savedmodelparams'] = []
                    database['modely']['objective'] = []
                    database['modely']['metrics'] = {'msemetric': [], 'chi2metric': []}
                    database['modelz']['objective'] = []
                    database['modelz']['Ztr'] = []
                database['modely']['savedmodelparams'].append(modely.param_array.tolist())
                database['modelz']['savedmodelparams'].append(modelz.param_array.tolist())
                database['modely']['objective'].append(modely.objective_function())
                database['modelz']['objective'].append(modelz.objective_function())
                database['modelz']['Ztr'].append(Ztr)
                (msemetric, chi2metric) = self.getMetrics(Xte, MCte, Mte, modely, modelz)
                database['modely']['metrics']['msemetric'].append(msemetric)
                database['modely']['metrics']['chi2metric'].append(chi2metric)
                database['log']['iterations_done'] = iteration
                database['log']['timetaken'] = timer()-start
                with open(self.outfile, 'w') as f:
                    json.dump(database, f, indent=4)
                print('On iteration {}, OBJDIFF is {}\n'.format(iteration,(currenthetobjective-oldhetobjective)**2))
                sys.stdout.flush()
            iteration += 1

        if rank == 0:
            print("##############################")
            print(datetime.datetime.now())
            print("##############################")
            sys.stdout.flush()
        return bestmodely,bestmodelz

    def  buildGPmodelFromSavedParam(self):
        import json
        bestmetricval = np.infty
        bestmodely = None
        bestmodelz = None
        metricdataForPrint = {}
        iterationdataForPrint = {}
        print("Total No. of files = {}".format(len(self.paramsavefiles)))
        for pno, pfile in enumerate(self.paramsavefiles):
            with open(pfile, 'r') as f:
                ds = json.load(f)
            Ntr = ds['Ntr']
            Ns = ds['Ns']

            seed = ds['seed']
            np.random.seed(seed)

            kernel = ds['kernel']
            polyorder = None
            if kernel in ['poly', 'or']:
                polyorder = ds['polyorder']


            Xtrindex = ds['Xtrindex']
            Xtr = np.repeat(self.X[Xtrindex, :], [Ns] * len(Xtrindex), axis=0)
            Ytrmm = ds['Ytrmm']
            Ytrmm2D = np.array([Ytrmm]).transpose()

            Xteindex = np.in1d(np.arange(self.nens), Xtrindex)
            Xte = self.X[~Xteindex, :]
            ntest = len(Xte)
            MCte = self.MC[~Xteindex]
            DeltaMCte = self.DeltaMC[~Xteindex]
            Mte = np.array([self.approxmeancountval(x) for x in Xte])

            itermodely = np.zeros(len(ds['modely']['savedmodelparams']),
                                  dtype=object)
            itermodelz = np.zeros(len(ds['modely']['savedmodelparams']),
                                  dtype=object)
            for i in range(len(ds['modely']['savedmodelparams'])):
                kernelObjy = self.getKernel(kernel, polyorder)
                kernelObjz = self.getKernel(kernel, polyorder)
                modely = GPy.models.GPHeteroscedasticRegression(Xtr,
                                                                Ytrmm2D,
                                                                kernel=kernelObjy
                                                                )

                modely.update_model(False)
                modely.initialize_parameter()
                modely[:] = ds['modely']['savedmodelparams'][i]
                modely.update_model(True)

                Ztr = ds['modelz']['Ztr'][i]
                Ztr2D = np.array([Ztr]).transpose()

                modelz = GPy.models.GPRegression(
                    self.X[Xtrindex, :],
                    Ztr2D,
                    kernel=kernelObjz,
                    normalizer=True
                )
                modelz.update_model(False)
                modelz.initialize_parameter()
                modelz[:] = ds['modelz']['savedmodelparams'][i]
                modelz.update_model(True)

                itermodely[i] = modely
                itermodelz[i] = modelz

            (minindex,metric) = self.getBestModel(Xte,MCte,Mte,
                modelyarr=itermodely,modelzarr=itermodelz)
            if metric < bestmetricval:
                print("Updating best metric val. from {} to {}".format(bestmetricval, metric))
                bestmetricval = metric
                bestmodely = itermodely[minindex]
                bestmodelz = itermodelz[minindex]
            metricdataForPrint[pfile] = metric
            iterationdataForPrint[pfile] = {'bestiter':minindex+1,
                                            'totaliters':len(ds['modely']['savedmodelparams'])}
            print("Done with file no. {} : {}".format(pno,pfile))
            sys.stdout.flush()

        for k, v in sorted(metricdataForPrint.items(), key=lambda item: item[1]):
            print("%.2E \t %s (%d / %d)" % (v, os.path.basename(k),
                  iterationdataForPrint[k]['bestiter'],
                  iterationdataForPrint[k]['totaliters']))

            # # 0 mean GP to model f
            # model = GPy.core.GP(Xtr,
            #                     Ytrmm2D,
            #                     kernel=kernelObj,
            #                     likelihood=lik,
            #                     )

        return bestmodely,bestmodelz

    def getMetrics(self,Xte,MCte,Mte,modely,modelz):
        Zbar, Zv = modelz.predict(Xte)
        Zmean = np.array([z[0] for z in Zbar])
        Zvar = np.array([z[0] for z in Zv])
        Vmean = np.exp(Zmean + (Zvar / 2))

        ybar, vy = modely._raw_predict(Xte)

        Ymean = np.array([y[0] for y in ybar])
        Ymean += Mte

        Yvar = np.array([y[0] for y in vy])
        Yvar += Vmean
        Ysd = np.sqrt(Yvar)

        ############### METRIC
        chi2metric = np.mean(((Ymean - MCte) / Ysd) ** 2)
        msemetric = np.mean((Ymean-MCte)**2)
        return msemetric,chi2metric

    def getBestModel(self,Xte,MCte,Mte,modelyarr,modelzarr):
        metricarr = np.zeros(len(modelyarr),dtype=np.float)
        for i in range(len(modelyarr)):
            modely = modelyarr[i]
            modelz = modelzarr[i]
            (msemetric,chi2metric) = self.getMetrics(Xte,MCte,Mte,modely,modelz)
            metricarr[i] = chi2metric
        print(metricarr)
        minindex = np.argmin(metricarr)
        # print(minindex)
        return minindex,metricarr[minindex]

class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    parser = argparse.ArgumentParser(description='Baysian Optimal Experimental Design for Model Fitting',
                                     formatter_class=SaneFormatter)
    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument("-a", "--approx", dest="APPROX", type=str, default=None,required=True,
                        help="Polynomial/Rational approximation file over MC mean counts\n"
                             "REQUIRED argument")
    requiredNamed.add_argument("-b", "--buildtype", dest="BUILDTYPE", type=str, default="data", required=True,
                        choices=["data","savedparams"],
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

    parser.add_argument("--usempituning", dest="MPITUNE", default=False, action="store_true",
                        help="Use MPI for Tuning\n"
                             "REQUIRED only build type (\"-b\", \"--buildtype\") is \"data\"")

    parser.add_argument("--nrestart", dest="NRESTART", type=int, default=1,
                        help="Number of optimization restarts (multistart)\n"
                             "REQUIRED only build type (\"-b\", \"--buildtype\") is \"data\"")
    parser.add_argument("-k", "--kernel", dest="KERNEL", type=str, default="sqe",
                               choices=["matern32", "matern52","sqe","ratquad","poly","or"],
                               help="Kernel to use (ARD will be set to True for all (where applicable)\n"
                                    "REQUIRED only build type (\"-b\", \"--buildtype\") is \"data\"")
    parser.add_argument('--stopcond', dest="STOPCOND", default=10 **-4, type=float,
                        help="Stoping condition metric\n"
                                    "REQUIRED only build type (\"-b\", \"--buildtype\") is \"data\"")
    parser.add_argument('-s','--seed', dest="SEED", default=6391273, type=int,
                        help="Seed (Control for n-fold crossvalidation)\n"
                             "REQUIRED only build type (\"-b\", \"--buildtype\") is \"data\"")

    requiredNamed.add_argument("-p", "--paramfile", dest="PARAMFILES", type=str, default=[], nargs='+',
                               help="Parameter and Xinfo JSON file (s).\n"
                                    "REQUIRED only build type (\"-b\", \"--buildtype\") is \"savedparams\"")

    parser.add_argument("-v", "--debug", dest="DEBUG", action="store_true", default=False,
                        help="Turn on some debug messages")



    args = parser.parse_args()
    if rank == 0:
        print(args)
    GP = GaussianProcess(
        DATAFILE=args.DATAFILE,
        BUILDTYPE=args.BUILDTYPE,
        KERNEL=args.KERNEL,
        OBS=args.OBS,
        NRESTART=args.NRESTART,
        KEEPOUT=args.KEEPOUT,
        OUTDIR=args.OUTDIR,
        APPROX=args.APPROX,
        SEED=args.SEED,
        STOPCOND=args.STOPCOND,
        PARAMFILES=args.PARAMFILES,
        MPITUNE = args.MPITUNE,
        DEBUG = args.DEBUG
    )



