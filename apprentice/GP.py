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
# from memory_profiler import profile

class GaussianProcess():
    # @profile
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

        if self.buildtype in ["mlhgp","sk","gp"]:
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
            self.METRIC = kwargs['METRIC']
            self.paramsavefiles = kwargs['PARAMFILES']
            import json
            with open(self.paramsavefiles[0], 'r') as f:
                ds = json.load(f)
            self.obsname = ds['obsname']


        import apprentice
        self.meanappset = apprentice.appset.AppSet(kwargs['APPROX'], binids=[self.obsname])
        if len(self.meanappset._binids)!=1 or \
            self.meanappset._binids[0] != self.obsname:
            print("Something went wrong.\n"
                  "Mean function could not be created.\n"
                  "This code does not support multi output GP")
            exit(1)
        if self.buildtype == "savedparams":
            path, filename = os.path.split(kwargs['APPROX'])
            errfile = "err{}".format(filename)
            errfilepath = os.path.join(path,errfile)
            self.meanerrappset = apprentice.appset.AppSet(errfilepath, binids=[self.obsname])
            if len(self.meanappset._binids) != 1 or \
                    self.meanappset._binids[0] != self.obsname:
                print("Something went wrong.\n"
                      "Error mean function could not be created.\n"
                      "This code does not support multi output GP")
                exit(1)
        if self.buildtype == 'mlhgp':
            self.modely,self.modelz = self.buildMLHGPmodelFromData()
        elif self.buildtype == 'sk':
            self.modely,self.modelz = self.buildSKmodelFromData()
        elif self.buildtype == 'gp':
            self.modely = self.buildGPmodelFromData()
            self.modelz = None
        elif self.buildtype == 'savedparams':
            self.modely,self.modelz,self.bestparamfile = self.buildGPmodelFromSavedParam()

    def errapproxmeancountval(self, x):
        return self.meanerrappset.vals(x)[0]

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

    def buildSKmodelFromData(self):
        import json
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        Ns = 25
        np.random.seed(self.SEED)



        ############################
        # Training Data Prep
        ############################
        Ntr = int((1 - self.keepout) * self.nens)

        Xtrindex = np.random.choice(np.arange(self.nens), Ntr, replace=False)
        Xtr = np.repeat(self.X[Xtrindex, :], [Ns] * len(Xtrindex), axis=0)

        MCtr = np.repeat(self.MC[Xtrindex], Ns)

        DeltaMCtr = np.repeat(self.DeltaMC[Xtrindex], Ns)
        DeltaMCSqByNtr = (DeltaMCtr**2)/Ns
        DeltaMCSqByNtr2D = np.array([DeltaMCSqByNtr]).transpose()

        Ytr = np.random.normal(MCtr, DeltaMCtr)
        Mtr = np.array([self.approxmeancountval(x) for x in Xtr])
        # Y-M (Training labels)
        Ytrmm = Ytr - Mtr
        Ytrmm2D = np.array([Ytrmm]).transpose()

        ############################
        # Testing Data Prep
        ############################
        Xteindex = np.in1d(np.arange(self.nens), Xtrindex)
        Xte = self.X[~Xteindex, :]
        ntest = len(Xte)
        MCte = self.MC[~Xteindex]
        DeltaMCte = self.DeltaMC[~Xteindex]
        Mte = np.array([self.approxmeancountval(x) for x in Xte])


        ############################
        # Miscell Init
        ############################
        start = timer()
        if rank == 0:
            print("##############################")
            print(datetime.datetime.now())
            print("##############################")
            sys.stdout.flush()

        polyorder = None
        if self.kernel in ['poly', 'or']:
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
                'buildtype': self.buildtype,
                "Ytrmm": Ytrmm.tolist(),
                'modely': {},
                'modelz': {},
                'log': {'stopcond': self.STOPCOND}
            }
            if self.kernel in ['poly', 'or']:
                database['polyorder'] = polyorder

        ############################
        # Kernel for y
        ############################
        kernelObjY = self.getKernel(self.kernel, polyorder)

        ############################
        # GP Model y
        ############################
        modelY = GPy.models.GPHeteroscedasticRegression(Xtr,
                                                           Ytrmm2D,
                                                           kernel=kernelObjY
                                                           )
        modelY['.*het_Gauss.variance'] = DeltaMCSqByNtr2D
        modelY.het_Gauss.variance.fix()

        ############################
        # Tune GP Model y
        ############################
        modelY[:] = self.mpitune(modelY, num_restarts=self.nrestart,
                                    useMPI=self.MPITUNE, robust=True)
        modelY.update_model(True)
        if self._debug and rank == 0:
            print(modelY.objective_function())

        ############################
        # Kernel for z
        ############################
        kernelObjZ = self.getKernel(self.kernel, polyorder)

        ############################
        # GP Model z
        ############################
        Vtr = (self.DeltaMC[Xtrindex])**2
        Ztr = [np.log(v) for v in Vtr]
        Ztr2D = np.array([Ztr]).transpose()
        modelZ = GPy.models.GPRegression(
                            self.X[Xtrindex, :],
                            Ztr2D,
                            kernel=kernelObjZ,
                            normalizer=True
                )

        ############################
        # Tune GP Model z
        ############################
        modelZ[:] = self.mpitune(modelZ, num_restarts=self.nrestart,
                                 useMPI=self.MPITUNE, robust=True)
        modelZ.update_model(True)
        if self._debug and rank == 0:
            print(modelZ.objective_function())

        ############################
        # Dump Results and Exit
        ############################
        if rank == 0:
            database['modely']['param_names_flat'] = modelY.parameter_names_flat().tolist()
            database['modelz']['param_names_flat'] = modelZ.parameter_names_flat().tolist()
            database['modely']["param_names"] = modelY.parameter_names()
            database['modelz']["param_names"] = modelZ.parameter_names()
            database['modely']['savedmodelparams'] = []
            database['modelz']['savedmodelparams'] = []
            database['modely']['objective'] = []
            database['modely']['metrics'] = {'meanmsemetric': [], 'sdmsemetric':[],'chi2metric': []}
            database['modelz']['objective'] = []
            database['modelz']['Ztr'] = []
            database['modely']['savedmodelparams'].append(modelY.param_array.tolist())
            database['modelz']['savedmodelparams'].append(modelZ.param_array.tolist())
            database['modely']['objective'].append(modelY.objective_function())
            database['modelz']['objective'].append(modelZ.objective_function())
            database['modelz']['Ztr'].append(Ztr)
            (meanmsemetric, sdmsemetric, chi2metric) = self.getMetrics(Xte, MCte, DeltaMCte,Mte, modelY, modelZ)
            database['modely']['metrics']['meanmsemetric'].append(meanmsemetric)
            database['modely']['metrics']['sdmsemetric'].append(sdmsemetric)
            database['modely']['metrics']['chi2metric'].append(chi2metric)
            database['log']['iterations_done'] = None
            database['log']['timetaken'] = timer() - start
            with open(self.outfile, 'w') as f:
                json.dump(database, f, indent=4)

            print("##############################")
            print(datetime.datetime.now())
            print("##############################")
            sys.stdout.flush()
        return modelY,modelZ

    def buildGPmodelFromData(self):
        import json
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        np.random.seed(self.SEED)

        Ns = 15
        ############################
        # Training Data Prep
        ############################
        Ntr = int((1 - self.keepout) * self.nens)

        Xtrindex = np.random.choice(np.arange(self.nens), Ntr, replace=False)
        ################################################################
        # Comment out for using only MC as Ytr - START
        ################################################################
        Xtr = np.repeat(self.X[Xtrindex, :], [Ns] * len(Xtrindex), axis=0)
        # Xtr = self.X[Xtrindex, :] # to revert back comment this line and uncomment line above

        MCtr = np.repeat(self.MC[Xtrindex], Ns)

        DeltaMCtr = np.repeat(self.DeltaMC[Xtrindex], Ns)
        DeltaMCSqByNtr = (DeltaMCtr**2)/Ns
        DeltaMCSqByNtr2D = np.array([DeltaMCSqByNtr]).transpose()

        Ytr = np.random.normal(MCtr, DeltaMCtr)
        # Ytr = self.MC[Xtrindex] # to revert back comment this line and uncomment line above
        ################################################################
        # Comment out for using only MC as Ytr - END
        ################################################################
        Mtr = np.array([self.approxmeancountval(x) for x in Xtr])

        # Y-M (Training labels)
        Ytrmm = Ytr - Mtr
        Ytrmm2D = np.array([Ytrmm]).transpose()

        ############################
        # Testing Data Prep
        ############################
        Xteindex = np.in1d(np.arange(self.nens), Xtrindex)
        Xte = self.X[~Xteindex, :]
        ntest = len(Xte)
        MCte = self.MC[~Xteindex]
        DeltaMCte = self.DeltaMC[~Xteindex]
        Mte = np.array([self.approxmeancountval(x) for x in Xte])


        ############################
        # Miscell Init
        ############################
        start = timer()
        if rank == 0:
            print("##############################")
            print(datetime.datetime.now())
            print("##############################")
            sys.stdout.flush()

        polyorder = None
        if self.kernel in ['poly', 'or']:
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
                'buildtype':self.buildtype,
                'modely': {},
                'log': {'stopcond': self.STOPCOND}

            }
            if self.kernel in ['poly', 'or']:
                database['polyorder'] = polyorder

        ############################
        # Kernel for y
        ############################
        kernelObjY = self.getKernel(self.kernel, polyorder)

        ############################
        # Likelihood for y
        ############################
        liklihoodY = GPy.likelihoods.Gaussian()

        ############################
        # GP Model y
        ############################
        modelY = GPy.core.GP(Xtr,
                            Ytrmm2D,
                            kernel=kernelObjY,
                            likelihood=liklihoodY
                            )
        ############################
        # Tune GP Model y
        ############################
        modelY[:] = self.mpitune(modelY, num_restarts=self.nrestart,
                                    useMPI=self.MPITUNE, robust=True)
        modelY.update_model(True)
        if self._debug and rank == 0:
            print(modelY.objective_function())

        ############################
        # Dump Results and Exit
        ############################
        if rank == 0:
            database['modely']['param_names_flat'] = modelY.parameter_names_flat().tolist()
            database['modely']["param_names"] = modelY.parameter_names()
            database['modely']['savedmodelparams'] = []
            database['modely']['objective'] = []
            database['modely']['metrics'] = {'meanmsemetric': [], 'sdmsemetric':[],'chi2metric': []}
            database['modely']['savedmodelparams'].append(modelY.param_array.tolist())
            database['modely']['objective'].append(modelY.objective_function())
            (meanmsemetric, sdmsemetric, chi2metric) = self.getMetrics(Xte, MCte, DeltaMCte,Mte, modelY, None)
            database['modely']['metrics']['meanmsemetric'].append(meanmsemetric)
            database['modely']['metrics']['sdmsemetric'].append(sdmsemetric)
            database['modely']['metrics']['chi2metric'].append(chi2metric)
            database['log']['iterations_done'] = None
            database['log']['timetaken'] = timer() - start
            with open(self.outfile, 'w') as f:
                json.dump(database, f, indent=4)

            print("##############################")
            print(datetime.datetime.now())
            print("##############################")
            sys.stdout.flush()
        return modelY

    def buildMLHGPmodelFromData(self):
        import json
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        Ns = 25
        np.random.seed(self.SEED)

        ############################
        # Training Data Prep
        ############################
        Ntr = int((1-self.keepout) *self.nens)

        Xtrindex = np.random.choice(np.arange(self.nens), Ntr, replace=False)
        Xtr = self.X[Xtrindex, :]

        Ytr = self.MC[Xtrindex]

        Mtr = np.array([self.approxmeancountval(x) for x in Xtr])
        # Y-M (Training labels)
        Ytrmm = Ytr - Mtr
        Ytrmm2D = np.array([Ytrmm]).transpose()

        ############################
        # Testing Data Prep
        ############################
        Xteindex = np.in1d(np.arange(self.nens), Xtrindex)
        Xte = self.X[~Xteindex, :]
        ntest = len(Xte)
        MCte = self.MC[~Xteindex]
        DeltaMCte = self.DeltaMC[~Xteindex]
        Mte = np.array([self.approxmeancountval(x) for x in Xte])

        ############################
        # Miscell Init
        ############################
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
                'buildtype': self.buildtype,
                "Ytrmm": Ytrmm.tolist(),
                'modely':{},
                'modelz':{},
                'log':{'stopcond':self.STOPCOND}
            }
            if self.kernel in ['poly', 'or']:
                database['polyorder'] = polyorder

        ############################
        # Kernel for y0
        ############################
        kernelObjY0 = self.getKernel(self.kernel, polyorder)

        ############################
        # Likelihood for y0
        ############################
        liklihoodY0 = GPy.likelihoods.Gaussian()

        ############################
        # GP Model y0
        ############################
        modelY0 = GPy.core.GP(Xtr,
                             Ytrmm2D,
                             kernel=kernelObjY0,
                             likelihood=liklihoodY0
                             )
        ############################
        # Tune GP Model y0
        ############################
        modelY0[:] = self.mpitune(modelY0, num_restarts=self.nrestart,
                                 useMPI=self.MPITUNE, robust=True)
        modelY0.update_model(True)
        if self._debug and rank == 0:
            print(modelY0.objective_function())

        ############################
        # Prepare to iterate
        ############################
        currenthetobjective = modelY0.objective_function()
        oldhetobjective = np.infty
        modelY = modelY0
        iteration = 0
        maxIterations=200
        bestobjectivey = np.infty
        bestmodely = None
        bestmodelz = None
        modelZ = None

        ############################
        # Iterate
        ############################
        while (currenthetobjective-oldhetobjective)**2 > self.STOPCOND:
            ########################################################
            # Estimate emperical variance for all training data
            ########################################################
            if iteration == 0:
                Ymean, Ysd = self.predictStaticHomoscedastic(Xtr,Mtr,modelY)
            else:
                Ymean, Ysd = self.predictStaticHeteroscedastic(Xtr,Mtr,modelY,modelZ)

            Vtr = []
            for no,(mean,sd) in enumerate(zip(Ymean,Ysd)):
                samples = np.random.normal(mean,sd,Ns)
                sqr = [0.5*((Ytr[no]-s)**2) for s in samples]
                val = sum(sqr)/Ns
                Vtr.append(val)

            ########################################################
            # Construct training data for GP Z
            ########################################################
            Ztr = [np.log(v) for v in Vtr]
            Ztr2D = np.array([Ztr]).transpose()

            ############################
            # Kernel for z
            ############################
            kernelObjZ = self.getKernel(self.kernel, polyorder)

            ############################
            # GP Model z
            ############################
            modelZ = GPy.models.GPRegression(
                                self.X[Xtrindex, :],
                                Ztr2D,
                                kernel=kernelObjZ,
                                normalizer = True
                                )
            ############################
            # Tune GP Model z
            ############################
            modelZ[:] = self.mpitune(modelZ, num_restarts=self.nrestart,
                                  useMPI=self.MPITUNE, robust=True)
            modelZ.update_model(True)
            if self._debug and rank == 0:
                 print(modelZ.objective_function())

            ########################################################
            # Construct the heteroscedastic variance for y
            ########################################################
            Zbar,Zv = modelZ.predict(Xtr)
            Zmean = np.array([z[0] for z in Zbar])
            Zvar = np.array([z[0] for z in Zv])
            Vmean = np.exp(Zmean + (Zvar / 2))
            Vmean2D = np.array([Vmean]).transpose()

            ############################
            # Kernel for y
            ############################
            kernelObjY = self.getKernel(self.kernel, polyorder)

            ############################
            # GP Model y
            ############################
            modelY = GPy.models.GPHeteroscedasticRegression(Xtr,
                                                            Ytrmm2D,
                                                            kernel=kernelObjY
                                                            )
            ########################################################
            # Set heteroscedastic variance for y
            ########################################################
            modelY['.*het_Gauss.variance'] = Vmean2D
            modelY.het_Gauss.variance.fix()

            ############################
            # Tune GP Model y
            ############################
            modelY[:] = self.mpitune(modelY, num_restarts=self.nrestart,
                                  useMPI=self.MPITUNE, robust=True)
            modelY.update_model(True)
            if self._debug and rank == 0:
                 print(modelY.objective_function())

            ############################
            # Record progress of iteration
            ############################
            oldhetobjective = currenthetobjective
            currenthetobjective = modelY.objective_function()
            if bestobjectivey > modelY.objective_function():
                bestobjectivey = modelY.objective_function()
                bestmodely = modelY
                bestmodelz = modelZ

            ############################
            # Dump Results
            ############################
            if rank == 0:
                if iteration==0:
                    database['modely']['param_names_flat'] = modelY.parameter_names_flat().tolist()
                    database['modelz']['param_names_flat'] = modelZ.parameter_names_flat().tolist()
                    database['modely']["param_names"] = modelY.parameter_names()
                    database['modelz']["param_names"] = modelZ.parameter_names()
                    database['modely']['savedmodelparams'] = []
                    database['modelz']['savedmodelparams'] = []
                    database['modely']['objective'] = []
                    database['modely']['metrics'] = {'meanmsemetric': [], 'sdmsemetric':[],'chi2metric': []}
                    database['modelz']['objective'] = []
                    database['modelz']['Ztr'] = []
                database['modely']['savedmodelparams'].append(modelY.param_array.tolist())
                database['modelz']['savedmodelparams'].append(modelZ.param_array.tolist())
                database['modely']['objective'].append(modelY.objective_function())
                database['modelz']['objective'].append(modelZ.objective_function())
                database['modelz']['Ztr'].append(Ztr)
                (meanmsemetric, sdmsemetric, chi2metric) = self.getMetrics(Xte, MCte, DeltaMCte, Mte, modelY, modelZ)
                database['modely']['metrics']['meanmsemetric'].append(meanmsemetric)
                database['modely']['metrics']['sdmsemetric'].append(sdmsemetric)
                database['modely']['metrics']['chi2metric'].append(chi2metric)
                database['log']['iterations_done'] = iteration
                database['log']['timetaken'] = timer()-start
                with open(self.outfile, 'w') as f:
                    json.dump(database, f, indent=4)
                if self._debug:
                    print('On iteration {}, OBJDIFF is {}\n'.format(iteration,(currenthetobjective-oldhetobjective)**2))
                sys.stdout.flush()
            iteration += 1

            ############################
            # Check Max Iterations
            ############################
            if iteration == maxIterations:
                break

        if rank == 0:
            print("##############################")
            print(datetime.datetime.now())
            print("##############################")
            sys.stdout.flush()
        return bestmodely,bestmodelz

    # @profile
    def  buildGPmodelFromSavedParam(self):
        import json
        bestmetricval = np.infty
        bestparamfile = None
        metricdataForPrint = {}
        iterationdataForPrint = {}
        metrickey = self.METRIC
        print("Total No. of files = {}".format(len(self.paramsavefiles)))
        for pno, pfile in enumerate(self.paramsavefiles):
            with open(pfile, 'r') as f:
                ds = json.load(f)
            metricarr = np.array(ds['modely']['metrics'][metrickey])
            minindex = np.argmin(metricarr)
            metric = metricarr[minindex]
            if metric < bestmetricval:
                print("Updating best metric val. from {} to {} ({}:{})".format(bestmetricval, metric,
                                                                               os.path.basename(pfile),minindex+1))
                bestmetricval = metric
                bestparamfile = pfile

            metricdataForPrint[pfile] = metric
            iterationdataForPrint[pfile] = {'bestiter': minindex + 1,
                                    'totaliters': len(ds['modely']['savedmodelparams'])}

        for k, v in sorted(metricdataForPrint.items(), key=lambda item: item[1]):
            print("%.2E \t %s (%d / %d)" % (v, os.path.basename(k),
                  iterationdataForPrint[k]['bestiter'],
                  iterationdataForPrint[k]['totaliters']))

        with open(bestparamfile, 'r') as f:
            ds = json.load(f)
        metricarr = np.array(ds['modely']['metrics'][metrickey])
        minindex = np.argmin(metricarr)
        print("METRIC KEY is {}".format(metrickey))
        print("Best Kernel is {}".format(ds['kernel']))
        print("Best parameter file is: {} \nand best iteration no. is {}".format(bestparamfile,minindex+1))
        print("with meanmsemetric %.2E"%(ds['modely']['metrics']['meanmsemetric'][minindex]))
        print("with sdmsemetric %.2E" % (ds['modely']['metrics']['sdmsemetric'][minindex]))
        print("with chi2metric %.2E" % (ds['modely']['metrics']['chi2metric'][minindex]))
        Ns = ds['Ns']

        seed = ds['seed']
        np.random.seed(seed)

        kernel = ds['kernel']
        polyorder = None
        if kernel in ['poly', 'or']:
            polyorder = ds['polyorder']

        Xtrindex = ds['Xtrindex']

        kernelObjy = self.getKernel(kernel, polyorder)

        if 'buildtype' in ds and ds['buildtype'] == "mlhgp":
            ################################################################
            # Comment out for using only MC as Ytr - START
            ################################################################
            # Xtr = np.repeat(self.X[Xtrindex, :], [Ns] * len(Xtrindex), axis=0)
            Xtr = self.X[Xtrindex, :]  # to revert back comment this line and uncomment line above
            ################################################################
            # Comment out for using only MC as Ytr - END
            ################################################################
            Ytrmm = ds['Ytrmm']
            Ytrmm2D = np.array([Ytrmm]).transpose()
            modely = GPy.models.GPHeteroscedasticRegression(Xtr,
                                                            Ytrmm2D,
                                                            kernel=kernelObjy
                                                            )
        elif 'buildtype' in ds and ds['buildtype'] == "gp":
            ################################################################
            # Comment out for using only MC as Ytr - START
            ################################################################
            Xtr = np.repeat(self.X[Xtrindex, :], [Ns] * len(Xtrindex), axis=0)
            # Xtr = self.X[Xtrindex, :]  # to revert back comment this line and uncomment line above
            ################################################################
            # Comment out for using only MC as Ytr - END
            ################################################################
            Ytrmm = ds['Ytrmm']
            Ytrmm2D = np.array([Ytrmm]).transpose()
            liklihoodY = GPy.likelihoods.Gaussian()
            modely = GPy.core.GP(Xtr,
                            Ytrmm2D,
                            kernel=kernelObjy,
                            likelihood=liklihoodY
                            )
        else:
            Xtr = np.repeat(self.X[Xtrindex, :], [Ns] * len(Xtrindex), axis=0)
            Ytrmm = ds['Ytrmm']
            Ytrmm2D = np.array([Ytrmm]).transpose()
            modely = GPy.models.GPHeteroscedasticRegression(Xtr,
                                                            Ytrmm2D,
                                                            kernel=kernelObjy
                                                            )

        modely.update_model(False)
        modely.initialize_parameter()
        modely[:] = ds['modely']['savedmodelparams'][minindex]
        modely.update_model(True)

        modelz = None
        if 'buildtype' not in ds or ds['buildtype'] != "gp":
            kernelObjz = self.getKernel(kernel, polyorder)

            Ztr = ds['modelz']['Ztr'][minindex]
            Ztr2D = np.array([Ztr]).transpose()

            modelz = GPy.models.GPRegression(
                self.X[Xtrindex, :],
                Ztr2D,
                kernel=kernelObjz,
                normalizer=True
            )
            modelz.update_model(False)
            modelz.initialize_parameter()
            modelz[:] = ds['modelz']['savedmodelparams'][minindex]
            modelz.update_model(True)

        return modely,modelz,bestparamfile

    def predictHeteroscedastic(self,Xte):
        Mte = np.array([self.approxmeancountval(x) for x in Xte])
        Zbar, Zv = self.modelz.predict(Xte)
        Zmean = np.array([z[0] for z in Zbar])
        Zvar = np.array([z[0] for z in Zv])
        Vmean = np.exp(Zmean + (Zvar / 2))

        ybar, vy = self.modely._raw_predict(Xte)

        Ymean = np.array([y[0] for y in ybar])
        Ymean += Mte

        Yvar = np.array([y[0] for y in vy])
        Yvar += Vmean
        Ysd = np.sqrt(Yvar)

        return Ymean,Ysd

    def predictHomoscedastic(self,Xte):
        Mte = np.array([self.approxmeancountval(x) for x in Xte])

        ybar, vy = self.modely.predict(Xte)

        Ymean = np.array([y[0] for y in ybar])
        Ymean += Mte

        Yvar = np.array([y[0] for y in vy])
        Ysd = np.sqrt(Yvar)

        return Ymean,Ysd

    def predictStaticHomoscedastic(self, Xte, Mte, modely):
        ybar, vy = modely.predict(Xte)

        Ymean = np.array([y[0] for y in ybar])
        Ymean += Mte

        Yvar = np.array([y[0] for y in vy])
        Ysd = np.sqrt(Yvar)

        return Ymean, Ysd

    def predictStaticHeteroscedastic(self,Xte,Mte,modely,modelz):
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

        return Ymean,Ysd

    def getMetrics(self,Xte,MCte,DeltaMCte,Mte,modely,modelz):
        if modelz is None:
            (Ymean, Ysd) = self.predictStaticHomoscedastic(Xte,Mte,modely)
        else:
            (Ymean, Ysd) = self.predictStaticHeteroscedastic(Xte, Mte, modely, modelz)

        ############### METRIC
        chi2metric = np.mean(((Ymean - MCte) / Ysd) ** 2)
        meanmsemetric = np.mean((Ymean-MCte)**2)
        sdmsemetric = np.mean((Ysd-DeltaMCte)**2)
        return meanmsemetric,sdmsemetric,chi2metric

    def printRAmetrics(self,RAFOLD):

        print("\n\n\n\n")
        bestparamfile = self.bestparamfile
        import json
        with open(bestparamfile, 'r') as f:
            ds = json.load(f)

        Xtrindex = ds['Xtrindex']
        Xteindex = np.in1d(np.arange(self.nens), Xtrindex)
        Xte = self.X[~Xteindex, :]
        MCte = self.MC[~Xteindex]
        DeltaMCte = self.DeltaMC[~Xteindex]

        if RAFOLD:
            import apprentice
            filearr = []
            msemeanarr = []
            sdmeanarr = []
            chi2arr = []
            for pno, pfile in enumerate(self.paramsavefiles):
                OUTDIR = os.path.dirname(pfile)
                with open(pfile, 'r') as f:
                    ds = json.load(f)
                seed = ds['seed']
                Moutfile = os.path.join(OUTDIR, 'RA',"{}_MCRA_S{}.json".format(self.obsname.replace('/', '_'),
                                                                           seed))
                DeltaMoutfile = os.path.join(OUTDIR, 'RA',"{}_DeltaMCRA_S{}.json".format(self.obsname.replace('/', '_'),
                                                                            seed))
                meanappset = apprentice.appset.AppSet(Moutfile, binids=[self.obsname])
                if len(meanappset._binids) != 1 or \
                        meanappset._binids[0] != self.obsname:
                    print("Something went wrong.\n"
                          "RA Fold Mean function could not be created.")
                    exit(1)
                meanerrappset = apprentice.appset.AppSet(DeltaMoutfile, binids=[self.obsname])
                if len(meanerrappset._binids) != 1 or \
                        meanerrappset._binids[0] != self.obsname:
                    print("Something went wrong.\n"
                          "RA Fold Error mean function could not be created.")
                    exit(1)

                Mte = np.array([meanappset.vals(x)[0] for x in Xte])
                DeltaMte = np.array([meanerrappset.vals(x)[0] for x in Xte])

                filearr.append(os.path.basename(Moutfile))
                msemeanarr.append(np.mean((Mte - MCte) ** 2))
                sdmeanarr.append(np.mean((DeltaMte - DeltaMCte) ** 2))
                chi2arr.append(np.mean(((Mte - MCte) / DeltaMte) ** 2))

            if self.METRIC == "meanmsemetric":
                bestindex = np.argmin(msemeanarr)
                print("Best file is %s" % (filearr[bestindex]))
                print("RAMEAN (meanmsemetric_RA) is %.2E" % (msemeanarr[bestindex]))
                print("RAMEAN (sdmsemetric_RA) is %.2E" % (sdmeanarr[bestindex]))
                print("RAMEAN (chi2metric_RA) is %.2E" % (chi2arr[bestindex]))
            elif self.METRIC == "sdmsemetric":
                bestindex = np.argmin(sdmeanarr)
                print("Best file is %s" % (filearr[bestindex]))
                print("RAMEAN (meanmsemetric_RA) is %.2E" % (msemeanarr[bestindex]))
                print("RAMEAN (sdmsemetric_RA) is %.2E" % (sdmeanarr[bestindex]))
                print("RAMEAN (chi2metric_RA) is %.2E" % (chi2arr[bestindex]))
            elif self.METRIC == "chi2metric":
                bestindex = np.argmin(chi2arr)
                print("Best file is %s" % (filearr[bestindex]))
                print("RAMEAN (meanmsemetric_RA) is %.2E" % (msemeanarr[bestindex]))
                print("RAMEAN (sdmsemetric_RA) is %.2E" % (sdmeanarr[bestindex]))
                print("RAMEAN (chi2metric_RA) is %.2E" % (chi2arr[bestindex]))
            return self.paramsavefiles[bestindex]
        else:
            Mte = np.array([self.approxmeancountval(x) for x in Xte])
            DeltaMte = np.array([self.errapproxmeancountval(x) for x in Xte])
            if self.METRIC == "meanmsemetric":
                meanmsemetric_RA = np.mean((Mte - MCte) ** 2)
                print("RAMEAN (meanmsemetric_RA) is %.2E" % meanmsemetric_RA)
            elif self.METRIC == "sdmsemetric":
                sdmsemetric_RA = np.mean((DeltaMte - DeltaMCte) ** 2)
                print("RAMEAN (sdmsemetric_RA) is %.2E" % sdmsemetric_RA)
            elif self.METRIC == "chi2metric":
                chi2metric_RA = np.mean(((Mte - MCte) / DeltaMte) ** 2)
                print("RAMEAN (chi2metric_RA) is %.2E" % chi2metric_RA)
        return None

    def buildRAmodelFromData(self,OUTDIR):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        Ntr = int((1 - self.keepout) * self.nens)
        np.random.seed(self.SEED)
        Xtrindex = np.random.choice(np.arange(self.nens), Ntr, replace=False)
        Xtr = self.X[Xtrindex, :]
        MCtr = self.MC[Xtrindex]
        DeltaMCtr = self.DeltaMC[Xtrindex]

        path = os.path.join(OUTDIR, 'RA')
        os.makedirs(path,exist_ok=True)
        MCoutfile = os.path.join(path,"{}_MCRA_S{}.json".format(self.obsname.replace('/', '_'),
                                                                          self.SEED))
        DeltaMCoutfile = os.path.join(path,"{}_DeltaMCRA_S{}.json".format(self.obsname.replace('/', '_'),
                                                                   self.SEED))

        m = 3
        n = 1
        import apprentice
        MCRA = apprentice.RationalApproximationSIP(Xtr, MCtr,
                                            m=m,
                                            n=n,
                                            trainingscale="Cp",
                                            roboptstrategy = 'ms',
                                            localoptsolver = 'scipy',
                                            fitstrategy = 'filter',
                                            strategy=0,
                                            pnames=None,
                                            debug = self._debug
                                            )

        DeltaMCRA = apprentice.RationalApproximationSIP(Xtr, DeltaMCtr,
                                                   m=m,
                                                   n=n,
                                                   trainingscale="Cp",
                                                   roboptstrategy='ms',
                                                   localoptsolver='scipy',
                                                   fitstrategy='filter',
                                                   strategy=0,
                                                   pnames=None,
                                                   debug=self._debug
                                                   )
        if rank==0:
            import json
            d = {self.obsname:MCRA.asDict}
            with open(MCoutfile, "w") as f:
                json.dump(d, f, indent=4, sort_keys=True)

            d = {self.obsname: DeltaMCRA.asDict}
            with open(DeltaMCoutfile, "w") as f:
                json.dump(d, f, indent=4, sort_keys=True)



class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ttttt = "[\"mlhgp\", \"sk\", \"gp\"]"
    parser = argparse.ArgumentParser(description='Baysian Optimal Experimental Design for Model Fitting',
                                     formatter_class=SaneFormatter)
    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument("-a", "--approx", dest="APPROX", type=str, default=None,required=True,
                        help="Polynomial/Rational approximation file over MC mean counts\n"
                             "REQUIRED argument")
    requiredNamed.add_argument("-b", "--buildtype", dest="BUILDTYPE", type=str, default="gp", required=True,
                        choices=["mlhgp","savedparams","sk","gp"],
                        help="Build type\n"
                             "REQUIRED argument")
    requiredNamed.add_argument("-d", "--datafile", dest="DATAFILE", type=str, default=None, required=True,
                        help="MC Data file where the first n-2 columns are X, (n-1)st "
                             "column is MC and nth column is DeltaMC\n"
                             "REQUIRED argument")

    parser.add_argument("--obsname", dest="OBS", type=str, default=None,
                        help="Observable Name\n"
                        "REQUIRED only build type (\"-b\", \"--buildtype\") is in "+ttttt)
    parser.add_argument("--keepout", dest="KEEPOUT", type=float, default=20.,
                        help="Percentage in \[0,100\] of the data to be left out as test data. \n"
                             "Train on the (100-keepout) percent data and then test on the rest. \n"
                             "REQUIRED only build type (\"-b\", \"--buildtype\") is in "+ttttt)
    parser.add_argument("-o", "--outdir", dest="OUTDIR", type=str, default=None,
                        help="Output Directory \n"
                             "REQUIRED only build type (\"-b\", \"--buildtype\") is in "+ttttt)

    parser.add_argument("--usempituning", dest="MPITUNE", default=False, action="store_true",
                        help="Use MPI for Tuning\n"
                             "REQUIRED only build type (\"-b\", \"--buildtype\") is in "+ttttt)

    parser.add_argument("--nrestart", dest="NRESTART", type=int, default=1,
                        help="Number of optimization restarts (multistart)\n"
                             "REQUIRED only build type (\"-b\", \"--buildtype\") is in "+ttttt)
    parser.add_argument("-k", "--kernel", dest="KERNEL", type=str, default="sqe",
                               choices=["matern32", "matern52","sqe","ratquad","poly","or"],
                               help="Kernel to use (ARD will be set to True for all (where applicable)\n"
                                    "REQUIRED only build type (\"-b\", \"--buildtype\") is in "+ttttt)
    parser.add_argument('--stopcond', dest="STOPCOND", default=10 **-4, type=float,
                        help="Stoping condition metric\n"
                                    "REQUIRED only build type (\"-b\", \"--buildtype\") is in "+ttttt)
    parser.add_argument('-s','--seed', dest="SEED", default=6391273, type=int,
                        help="Seed (Control for n-fold crossvalidation)\n"
                             "REQUIRED only build type (\"-b\", \"--buildtype\") is in "+ttttt)

    parser.add_argument("-p", "--paramfile", dest="PARAMFILES", type=str, default=[], nargs='+',
                               help="Parameter and Xinfo JSON file (s).\n"
                                    "REQUIRED only build type (\"-b\", \"--buildtype\") is \"savedparams\"")
    parser.add_argument("-m", "--metric", dest="METRIC", type=str, default="meanmsemetric",
                        choices=["meanmsemetric", "sdmsemetric", "chi2metric"],
                        help="Metric based on which to select the best GP parameters on.\n"
                             "REQUIRED only build type (\"-b\", \"--buildtype\") is \"savedparams\"")

    parser.add_argument("--dorafold", dest="DORAFOLD", default=False, action="store_true",
                        help="When build type (\"-b\", \"--buildtype\") is in "+ttttt+",\n"
                            "rational approx with order (3,1) will be calculated for the current data fold.\n"
                            "When build type (\"-b\", \"--buildtype\") is is \"savedparams\",\n"
                            "results of the rational approx from the best fold will be printed"
                        )

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
        METRIC = args.METRIC,
        DEBUG = args.DEBUG
    )

    if args.DORAFOLD and args.BUILDTYPE in ["mlhgp","sk","gp"] :
        GP.buildRAmodelFromData(OUTDIR=args.OUTDIR)
    if args.BUILDTYPE == "savedparams":
        GP.printRAmetrics(
            RAFOLD = args.DORAFOLD
        )




