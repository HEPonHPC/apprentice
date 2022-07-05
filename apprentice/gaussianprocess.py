import pprint

import numpy as np
import datetime
from timeit import default_timer as timer
import os, sys
import apprentice
import GPy
from apprentice.surrogatemodel import SurrogateModel
from apprentice.mpi4py_ import MPI_
from apprentice.space import Space

class GaussianProcess(SurrogateModel):
    __allowed = (
                 "debug_","debug",
                 "training_size_","training_size",
                 "strategy_","strategy",
                 'seed_',"seed",
                 "use_mpi_tune_",'use_mpi_tune',
                 "stopping_bound_","stopping_bound",
                 "kernel_","kernel",
                 "max_restarts_","max_restarts",
                 "keepout_percentage_","keepout_percentage",
                 "mean_surrogate_model_","mean_surrogate_model",
                "error_surrogate_model_","error_surrogate_model",
                "sample_size_","sample_size",
                "polynomial_order_", "polynomial_order",
                 'Ntr','Xtr','Ytrmm'
                 )

    def __init__(self, dim, fnspace=None, **kwargs: dict):
        super().__init__(dim, fnspace)
        for k, v in kwargs.items():
            if k in ['debug',"training_size",'strategy','seed','use_mpi_tune',
                     "stopping_bound","kernel","max_restarts","keepout_percentage",
                     "mean_surrogate_model","error_surrogate_model", "sample_size","polynomial_order"]:
                k+="_"
            elif k in ['pnames','pnames_','scale_min','scale_max','scale_min_', 'scale_max_']: continue
            assert (k in self.__class__.__allowed)
            setattr(self,k, v)

    @property
    def use_mpi_tune(self):
        if hasattr(self, 'use_mpi_tune_'):
            return self.use_mpi_tune_
        else:return False

    @property
    def stopping_bound(self):
        if hasattr(self, 'stopping_bound_'):
            return self.stopping_bound_
        else:return 10 **-4

    @property
    def polynomial_order(self):
        if hasattr(self, 'polynomial_order_'):
            return self.polynomial_order_
        else:return 3

    @property
    def sample_size(self):
        if hasattr(self, 'sample_size_'):
            return self.sample_size_
        else:return 25

    @property
    def fit_strategy(self, print_descr = False):
        strat_num_to_descr = {
            "1": "Most Likely Heteroscedastic Gaussian Process (HeGP-ML)",
            "2": "Heteroscedastic Gaussian Process using Stochastic Kriging (HeGP-SK)",
            "3": "Homoscedastic Gaussian Process (HoGP)"
        }
        if print_descr:
            pprint.pprint(strat_num_to_descr)

        strat_num_to_accr = {
            "1": "HeGP-ML",
            "2": "HeGP-SK",
            "3": "HoGP"
        }
        allowed_nums = [k for k in strat_num_to_accr.keys()]
        allowed_accr = [v for v in strat_num_to_accr.values()]
        if hasattr(self, 'strategy_'):
            if self.strategy_ in allowed_nums:
                return self.strategy_
            elif self.strategy_ in allowed_accr:
                return allowed_nums[allowed_accr.index(self.strategy_)]
            else:
                raise Exception("Strategy not allowed: {}".format(self.strategy_))
        return "3" # default is Homoscedastic Gaussian Process

    @property
    def debug(self):
        if hasattr(self, 'debug_'):
            return self.debug_
        else:return False

    @property
    def max_restarts(self):
        if hasattr(self, 'max_restarts_'):
            return self.max_restarts_
        else: return 100

    @property
    def mean_surrogate_model(self):
        if hasattr(self, 'mean_surrogate_model_'):
            return self.mean_surrogate_model_
        raise Exception("Mean surrogate model required but not provided")

    @property
    def error_surrogate_model(self):
        if hasattr(self, 'error_surrogate_model_'):
            return self.error_surrogate_model_
        raise Exception("Error surrogate model required but not provided")
    @property
    def dim(self):
        return self.fnspace.dim

    @property
    def pnames(self):
        return self.fnspace.pnames

    @property
    def kernel(self):
        if hasattr(self, 'kernel_'):
            return self.kernel_
        else:return "sqe"

    @property
    def seed(self):
        if hasattr(self, 'seed_'):
            return self.seed_
        else:return 97834667

    @property
    def keepout_percentage(self):
        if hasattr(self, 'keepout_percentage_'):
            return self.keepout_percentage_
        else:
            return 20

    def fit(self,X,Y):
        """
        Do everything.
        """
        strategy = self.fit_strategy
        if strategy == '1':
            (self.modely,self.modelz,self.data) = self.build_MLHGP_model_from_data(X,Y)
        elif strategy == '2':
            (self.modely,self.modelz,self.data) = self.build_SK_model_from_data(X,Y)
        elif strategy == '3':
            (self.modely,self.data) = self.build_GP_model_from_data(X,Y)
            self.modelz = None
        else: raise Exception("fit() strategy %i not implemented"%strategy)

    def save(self, fname):
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            import json
            with open(fname, "w") as f:
                json.dump(self.as_dict, f,indent=4)

    @property
    def as_dict(self):
        self.data["fnspace"] = self.fnspace.as_dict
        return self.data

    def __repr__(self):
        """
        Print-friendly representation.
        """
        return "<GaussianProcess dim:{} kernel:{} strategy:{}>".format(self.dim, self.kernel,self.fit_strategy)

    @staticmethod
    def get_kernel(kernel_str, dim, poly_order):
        availablekernels = ["sqe","ratquad","matern32","matern52","poly"]
        if kernel_str == "sqe":
            kernel_obj = GPy.kern.RBF(input_dim=dim, ARD=True)
        elif kernel_str == "ratquad":
            kernel_obj = GPy.kern.RatQuad(input_dim=dim, ARD=True)
        elif kernel_str == "matern32":
            kernel_obj = GPy.kern.Matern32(input_dim=dim, ARD=True)
        elif kernel_str == "matern52":
            kernel_obj = GPy.kern.Matern52(input_dim=dim, ARD=True)
        elif kernel_str == "poly":
            kernel_obj = GPy.kern.Poly(input_dim=dim, order=poly_order)
        elif kernel_str == "or":
            kernel_obj = GaussianProcess.get_kernel(availablekernels[0],dim,poly_order)
            for i in range(1,len(availablekernels)):
                kernel_obj += GaussianProcess.get_kernel(availablekernels[i],dim,poly_order)
        else:
            raise Exception("Kernel {} unknown. Quitting now!".format(kernel_str))
        return kernel_obj

    @staticmethod
    def mpi_tune(model,num_restarts,use_mpi=False,robust=True,debug=False):
        comm = MPI_.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        if not use_mpi:
            model.optimize_restarts(num_restarts=num_restarts,
                                    robust=robust,verbose=debug)
            return model.param_array
        else:
            allWork = MPI_.chunk_it([i for i in range(num_restarts)])
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
                    model.optimize_restarts(num_restarts=1,robust=robust,verbose=debug)
                except Exception as e:
                    if robust:
                        print(("Warning - optimization restart on rank {} failed".format(ii)))
                    else:
                        raise e
                _paramarray[ii] = model.param_array
                _F[ii] = model.objective_function()

                if rank == 0 and debug:
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
                allWork = MPI_.chunk_it([i for i in range(num_restarts)])
                for r in range(size): _paramarray[allWork[r]] = a[r]
                for r in range(size): _F[allWork[r]] = b[r]
                myreturnvalue = _paramarray[np.argmin(_F)]
                if debug:
                    print("Objective values from all parallel runs:")
                    print(_F)
                    # print(_paramarray)
                    sys.stdout.flush()
            myreturnvalue = comm.bcast(myreturnvalue, root=0)
            return myreturnvalue

    def build_MLHGP_model_from_data(self,X,Y):
        np.random.seed(self.seed)
        X = np.array(X)
        Y = np.array(Y)
        # nens,nparam = X.shape
        # nens = self.training_size
        # nparam = self.dim
        ############################
        # Training Data Prep
        ############################
        Ntr = int((1-(self.keepout_percentage/100)) *self.training_size)

        Xtrindex = np.random.choice(np.arange(self.training_size), Ntr, replace=False)
        Xtr = X[Xtrindex, :]

        V = Y[:,0]
        DV = Y[:,1]

        Ytr = V[Xtrindex]

        Mtr = np.array([self.mean_surrogate_model(x) for x in Xtr])
        # Y-M (Training labels)
        Ytrmm = Ytr - Mtr
        Ytrmm2D = np.array([Ytrmm]).transpose()

        ############################
        # Testing Data Prep
        ############################
        Xteindex = np.in1d(np.arange(self.training_size), Xtrindex)
        Xte = X[~Xteindex, :]
        ntest = len(Xte)
        MCte = V[~Xteindex]
        DeltaMCte = DV[~Xteindex]
        Mte = np.array([self.mean_surrogate_model(x) for x in Xte])

        ############################
        # Miscell Init
        ############################
        start = timer()
        if self.debug:
            print("##############################")
            print(datetime.datetime.now())
            print("##############################")
            sys.stdout.flush()

        database = {
            "training_size":self.training_size,
            "use_mpi_tune":self.use_mpi_tune,
            "Ntr": Ntr,
            # "Mtr":Mtr.tolist(),
            "sample_size": self.sample_size,
            'seed': self.seed,
            'kernel': self.kernel,
            "keepout_percentage": self.keepout_percentage,
            "X":X[Xtrindex, :].tolist(),
            # "Xtrindex": Xtrindex.tolist(),
            'strategy': self.fit_strategy,
            "Ytrmm": Ytrmm.tolist(),
            'modely':{},
            'modelz':{},
            'stopping_bound':self.stopping_bound
        }
        if self.kernel in ['poly', 'or']:
            database['polynomial_order'] = self.polynomial_order


        ############################
        # Kernel for y0
        ############################
        kernelObjY0 = GaussianProcess.get_kernel(self.kernel,self.dim,self.polynomial_order)

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
        modelY0[:] = GaussianProcess.mpi_tune(modelY0, num_restarts=self.max_restarts,
                                  use_mpi=self.use_mpi_tune, robust=True,debug=self.debug)
        modelY0.update_model(True)
        if self.debug:
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
        while (currenthetobjective-oldhetobjective)**2 > self.stopping_bound:
            ########################################################
            # Estimate emperical variance for all training data
            ########################################################
            if iteration == 0:
                Ymean, Ysd = GaussianProcess.predict_static_homoscedastic(Xtr,Mtr,modelY)
            else:
                Ymean, Ysd = GaussianProcess.predict_static_heteroscedastic(Xtr,Mtr,modelY,modelZ)

            Vtr = []
            for no,(mean,sd) in enumerate(zip(Ymean,Ysd)):
                samples = np.random.normal(mean,sd,self.sample_size)
                sqr = [0.5*((Ytr[no]-s)**2) for s in samples]
                val = sum(sqr)/self.sample_size
                Vtr.append(val)

            ########################################################
            # Construct training data for GP Z
            ########################################################
            Ztr = [np.log(v) for v in Vtr]
            Ztr2D = np.array([Ztr]).transpose()

            ############################
            # Kernel for z
            ############################
            kernelObjZ = GaussianProcess.get_kernel(self.kernel,self.dim,self.polynomial_order)

            ############################
            # GP Model z
            ############################
            modelZ = GPy.models.GPRegression(
                                    X[Xtrindex, :],
                                    Ztr2D,
                                    kernel=kernelObjZ,
                                    normalizer = True
            )
            ############################
            # Tune GP Model z
            ############################
            modelZ[:] = self.mpi_tune(modelZ, num_restarts=self.max_restarts,
                                     use_mpi=self.use_mpi_tune, robust=True,debug=self.debug)
            modelZ.update_model(True)
            if self.debug:
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
            kernelObjY = GaussianProcess.get_kernel(self.kernel,self.dim,self.polynomial_order)

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

            modelY[:] = GaussianProcess.mpi_tune(modelY, num_restarts=self.max_restarts,
                                     use_mpi=self.use_mpi_tune, robust=True,debug=self.debug)
            modelY.update_model(True)
            if self.debug:
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
            (meanmsemetric, sdmsemetric, chi2metric) = GaussianProcess.get_metrics(Xte, MCte, DeltaMCte, Mte, modelY, modelZ)
            database['modely']['metrics']['meanmsemetric'].append(meanmsemetric)
            database['modely']['metrics']['sdmsemetric'].append(sdmsemetric)
            database['modely']['metrics']['chi2metric'].append(chi2metric)
            database['iterations_done'] = iteration
            database['timetaken'] = timer()-start
            if self.debug:
                print('On iteration {}, OBJDIFF is {}\n'.format(iteration,(currenthetobjective-oldhetobjective)**2))
            sys.stdout.flush()
            iteration += 1
            ############################
            # Check Max Iterations
            ############################
            if iteration == maxIterations:
                break
        if self.debug:
            print("##############################")
            print(datetime.datetime.now())
            print("##############################")
            sys.stdout.flush()

        return bestmodely,bestmodelz,database

    def build_SK_model_from_data(self,X,Y):
        np.random.seed(self.seed)
        X = np.array(X)
        Y = np.array(Y)
        ############################
        # Training Data Prep
        ############################
        Ntr = int((1 - self.keepout_percentage/100) * self.training_size)
        Xtrindex = np.random.choice(np.arange(self.training_size), Ntr, replace=False)
        Xtr = np.repeat(X[Xtrindex, :], [self.sample_size] * len(Xtrindex), axis=0)

        V = Y[:,0]
        DV = Y[:,1]

        Ytr = np.repeat(V[Xtrindex], self.sample_size)

        DYtr = np.repeat(DV[Xtrindex], self.sample_size)
        DYSqByNtr = (DYtr**2)/self.sample_size
        DYSqByNtr2D = np.array([DYSqByNtr]).transpose()

        Ytr = np.random.normal(Ytr, DYtr)
        Mtr = np.array([self.mean_surrogate_model(x) for x in Xtr])
        # Y-M (Training labels)
        Ytrmm = Ytr - Mtr
        Ytrmm2D = np.array([Ytrmm]).transpose()

        ############################
        # Testing Data Prep
        ############################
        Xteindex = np.in1d(np.arange(self.training_size), Xtrindex)
        Xte = X[~Xteindex, :]
        ntest = len(Xte)
        MCte = V[~Xteindex]
        DeltaMCte = DV[~Xteindex]
        Mte = np.array([self.mean_surrogate_model(x) for x in Xte])

        ############################
        # Miscell Init
        ############################
        start = timer()
        if self.debug:
            print("##############################")
            print(datetime.datetime.now())
            print("##############################")
            sys.stdout.flush()

        database = {
            # "savedmodelparams": model.param_array.tolist(),
            # "param_names_flat": model.parameter_names_flat().tolist(),
            # "param_names": model.parameter_names(),
            "training_size":self.training_size,
            "use_mpi_tune":self.use_mpi_tune,
            "Ntr": Ntr,
            # "Mtr":Mtr.tolist(),
            "sample_size": self.sample_size,
            'seed': self.seed,
            'kernel': self.kernel,
            "keepout_percentage": self.keepout_percentage,
            "X":X[Xtrindex, :].tolist(),
            # "Xtrindex": Xtrindex.tolist(),
            'strategy': self.fit_strategy,
            "Ytrmm": Ytrmm.tolist(),
            'modely': {},
            'modelz': {},
            'stopping_bound':self.stopping_bound
        }
        if self.kernel in ['poly', 'or']:
            database['polynomial_order'] = self.polynomial_order

        ############################
        # Kernel for y
        ############################
        kernelObjY = GaussianProcess.get_kernel(self.kernel,self.dim,self.polynomial_order)

        ############################
        # GP Model y
        ############################
        modelY = GPy.models.GPHeteroscedasticRegression(Xtr,
                                                        Ytrmm2D,
                                                        kernel=kernelObjY
                                                        )
        modelY['.*het_Gauss.variance'] = DYSqByNtr2D
        modelY.het_Gauss.variance.fix()

        ############################
        # Tune GP Model y
        ############################
        modelY[:] = GaussianProcess.mpi_tune(modelY, num_restarts=self.max_restarts,
                                             use_mpi=self.use_mpi_tune, robust=True,debug=self.debug)
        modelY.update_model(True)
        if self.debug:
            print(modelY.objective_function())

        ############################
        # Kernel for z
        ############################
        kernelObjZ = GaussianProcess.get_kernel(self.kernel,self.dim,self.polynomial_order)

        ############################
        # GP Model z
        ############################
        Vtr = (DV[Xtrindex])**2
        Ztr = [np.log(v) for v in Vtr]
        Ztr2D = np.array([Ztr]).transpose()
        modelZ = GPy.models.GPRegression(
            X[Xtrindex, :],
            Ztr2D,
            kernel=kernelObjZ,
            normalizer=True
        )

        ############################
        # Tune GP Model z
        ############################
        modelZ[:] = GaussianProcess.mpi_tune(modelZ, num_restarts=self.max_restarts,
                                 use_mpi=self.use_mpi_tune, robust=True,debug=self.debug)
        modelZ.update_model(True)
        if self.debug:
            print(modelZ.objective_function())

        ############################
        # Dump Results and Exit
        ############################
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
        (meanmsemetric, sdmsemetric, chi2metric) = GaussianProcess.get_metrics(Xte, MCte, DeltaMCte, Mte, modelY, modelZ)
        database['modely']['metrics']['meanmsemetric'].append(meanmsemetric)
        database['modely']['metrics']['sdmsemetric'].append(sdmsemetric)
        database['modely']['metrics']['chi2metric'].append(chi2metric)
        database['iterations_done'] = None
        database['timetaken'] = timer() - start

        if self.debug:
            print("##############################")
            print(datetime.datetime.now())
            print("##############################")
            sys.stdout.flush()

        return modelY,modelZ,database

    def build_GP_model_from_data(self,X, Y):
        import json
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        np.random.seed(self.seed)

        ############################
        # Training Data Prep
        ############################
        Ntr = int((1 - (self.keepout_percentage/100)) * self.training_size)

        Xtrindex = np.random.choice(np.arange(self.training_size), Ntr, replace=False)
        ################################################################
        # Comment out for using only MC as Ytr - START
        ################################################################
        Xtr = np.repeat(X[Xtrindex, :], [self.sample_size] * len(Xtrindex), axis=0)
        # Xtr = self.X[Xtrindex, :] # to revert back comment this line and uncomment line above

        V = Y[:,0]
        DV = Y[:,1]

        Vtr = np.repeat(V[Xtrindex], self.sample_size)

        DVtr = np.repeat(DV[Xtrindex], self.sample_size)
        DVSqByNtr = (DVtr**2)/self.sample_size
        DVSqByNtr2D = np.array([DVSqByNtr]).transpose()

        Ytr = np.random.normal(Vtr, DVtr)
        # Ytr = self.MC[Xtrindex] # to revert back comment this line and uncomment line above
        ################################################################
        # Comment out for using only MC as Ytr - END
        ################################################################
        Mtr = np.array([self.mean_surrogate_model(x) for x in Xtr])

        # Y-M (Training labels)
        Ytrmm = Ytr - Mtr
        Ytrmm2D = np.array([Ytrmm]).transpose()

        ############################
        # Testing Data Prep
        ############################
        Xteindex = np.in1d(np.arange(self.training_size), Xtrindex)
        Xte = X[~Xteindex, :]
        ntest = len(Xte)
        Dte = V[~Xteindex]
        DVte = DV[~Xteindex]
        Mte = np.array([self.mean_surrogate_model(x) for x in Xte])


        ############################
        # Miscell Init
        ############################
        start = timer()
        if self.debug:
            print("##############################")
            print(datetime.datetime.now())
            print("##############################")
            sys.stdout.flush()


        database = {
            "training_size":self.training_size,
            "use_mpi_tune":self.use_mpi_tune,
            "Ntr": Ntr,
            # "Mtr":Mtr.tolist(),
            "sample_size": self.sample_size,
            'seed': self.seed,
            'kernel': self.kernel,
            "keepout_percentage": self.keepout_percentage,
            "X":X[Xtrindex, :].tolist(),
            # "Xtrindex": Xtrindex.tolist(),
            'strategy': self.fit_strategy,
            "Ytrmm": Ytrmm.tolist(),
            'modely':{},
            'modelz':{},
            'stopping_bound':self.stopping_bound
        }
        if self.kernel in ['poly', 'or']:
            database['polynomial_order'] = self.polynomial_order

        ############################
        # Kernel for y
        ############################
        kernelObjY = GaussianProcess.get_kernel(self.kernel,self.dim,self.polynomial_order)

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
        modelY[:] = GaussianProcess.mpi_tune(modelY, num_restarts=self.max_restarts,
                                              use_mpi=self.use_mpi_tune, robust=True,debug=self.debug)
        modelY.update_model(True)
        if self.debug:
            print(modelY.objective_function())

        ############################
        # Dump Results and Exit
        ############################
        database['modely']['param_names_flat'] = modelY.parameter_names_flat().tolist()
        database['modely']["param_names"] = modelY.parameter_names()
        database['modely']['savedmodelparams'] = []
        database['modely']['objective'] = []
        database['modely']['metrics'] = {'meanmsemetric': [], 'sdmsemetric':[],'chi2metric': []}
        database['modely']['savedmodelparams'].append(modelY.param_array.tolist())
        database['modely']['objective'].append(modelY.objective_function())
        (meanmsemetric, sdmsemetric, chi2metric) = GaussianProcess.get_metrics(Xte, Dte, DVte, Mte, modelY, None)
        database['modely']['metrics']['meanmsemetric'].append(meanmsemetric)
        database['modely']['metrics']['sdmsemetric'].append(sdmsemetric)
        database['modely']['metrics']['chi2metric'].append(chi2metric)
        database['iterations_done'] = None
        database['timetaken'] = timer() - start
        if self.debug:
            print("##############################")
            print(datetime.datetime.now())
            print("##############################")
            sys.stdout.flush()

        return modelY,database

    @classmethod
    def from_file(cls, filepath,**kwargs):
        comm = MPI_.COMM_WORLD
        rank = comm.Get_rank()
        d = None
        if rank == 0:
            import json
            with open(filepath,'r') as f:
                d = json.load(f)
        d = comm.bcast(d,root=0)
        return cls.from_data_structure(d,**kwargs)

    @classmethod
    def from_data_structure(cls,data_structure,**kwargs):
        if not isinstance(data_structure, dict):
            raise Exception("data_structure has to be a dictionary")

        dim = data_structure['fnspace']['dim_']
        a = data_structure['fnspace']['a_']
        b = data_structure['fnspace']['b_']
        sa = data_structure['fnspace']['sa_']
        sb = data_structure['fnspace']['sb_']
        pnames = data_structure['fnspace']['pnames_']
        data_structure.pop('fnspace')

        fnspace = Space(dim,
                        a=a,
                        b=b,
                        sa=sa,
                        sb=sb,
                        pnames=pnames)

        # TODO Default right now is chi2metric. Ignoring other metrics and user input of metric to use for now
        metricarr = np.array(data_structure['modely']['metrics']['chi2metric'])
        minindex = np.argmin(metricarr)

        np.random.seed(data_structure['seed'])

        X = np.array(data_structure['X'])
        Ytrmm = data_structure['Ytrmm']
        Ytrmm2D = np.array([Ytrmm]).transpose()
        po = 3
        if data_structure['kernel'] in ['poly', 'or']:
            po = data_structure['polynomial_order']

        kernelObjy = GaussianProcess.get_kernel(data_structure['kernel'],fnspace.dim,po)

        if data_structure['strategy']  == "1":
            ################################################################
            # Comment out for using only MC as Ytr - START
            ################################################################
            # Xtr = np.repeat(self.X[Xtrindex, :], [Ns] * len(Xtrindex), axis=0)
            Xtr = X  # to revert back comment this line and uncomment line above
            ################################################################
            # Comment out for using only MC as Ytr - END
            ################################################################

            modely = GPy.models.GPHeteroscedasticRegression(Xtr,
                                                            Ytrmm2D,
                                                            kernel=kernelObjy
                                                            )
        elif data_structure['strategy'] == "3":
            ################################################################
            # Comment out for using only MC as Ytr - START
            ################################################################
            # TODO CHECK
            Xtr = np.repeat(X, [data_structure['sample_size']] * len(X), axis=0)
            # Xtr = self.X[Xtrindex, :]  # to revert back comment this line and uncomment line above
            ################################################################
            # Comment out for using only MC as Ytr - END
            ################################################################
            liklihoodY = GPy.likelihoods.Gaussian()
            modely = GPy.core.GP(Xtr,
                                 Ytrmm2D,
                                 kernel=kernelObjy,
                                 likelihood=liklihoodY
                                 )
        elif data_structure['strategy'] == "2":
            # TODO CHECK
            Xtr = np.repeat(X, [data_structure['sample_size']] * len(X), axis=0)
            modely = GPy.models.GPHeteroscedasticRegression(Xtr,
                                                            Ytrmm2D,
                                                            kernel=kernelObjy
                                                            )
        else: raise Exception("Strategy is not implemented: {}".format(data_structure['strategy']))

        modely.update_model(False)
        modely.initialize_parameter()
        modely[:] = data_structure['modely']['savedmodelparams'][minindex]
        modely.update_model(True)

        modelz = None
        if data_structure['strategy'] == '1' or data_structure['strategy'] == '2':
            kernelObjz = GaussianProcess.get_kernel(data_structure['kernel'],fnspace.dim,po)

            Ztr = data_structure['modelz']['Ztr'][minindex]
            Ztr2D = np.array([Ztr]).transpose()
            modelz = GPy.models.GPRegression(
                X,
                Ztr2D,
                kernel=kernelObjz,
                normalizer=True
            )
            modelz.update_model(False)
            modelz.initialize_parameter()
            modelz[:] = data_structure['modelz']['savedmodelparams'][minindex]
            modelz.update_model(True)
        data_structure.pop('X')
        data_structure.pop('Ytrmm')
        data_structure.pop('modely')
        if 'modelz' in data_structure: data_structure.pop('modelz')
        if 'iterations_done' in data_structure: data_structure.pop('iterations_done')
        data_structure.pop('timetaken')
        GP = cls(dim,fnspace,**data_structure)
        GP.mean_surrogate_model_ = kwargs['mean_surrogate_model']
        if 'error_surrogate_model' in kwargs: GP.error_surrogate_model_ = kwargs['error_surrogate_model']
        GP.modely,GP.modelz,GP.data = (modely,modelz,data_structure)
        return GP

    @staticmethod
    def predict_static_homoscedastic(Xte, Mte, modely):
        ybar, vy = modely.predict(Xte)

        Ymean = np.array([y[0] for y in ybar])
        Ymean += Mte

        Yvar = np.array([y[0] for y in vy])
        Ysd = np.sqrt(Yvar)

        return Ymean, Ysd

    @staticmethod
    def predict_static_heteroscedastic(Xte,Mte,modely,modelz):
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

    @staticmethod
    def get_metrics(Xte,MCte,DeltaMCte,Mte,modely,modelz):
        if modelz is None:
            (Ymean, Ysd) = GaussianProcess.predict_static_homoscedastic(Xte,Mte,modely)
        else:
            (Ymean, Ysd) = GaussianProcess.predict_static_heteroscedastic(Xte,Mte,modely,modelz)

        ############### METRIC
        chi2metric = np.mean(((Ymean - MCte) / Ysd) ** 2)
        meanmsemetric = np.mean((Ymean-MCte)**2)
        sdmsemetric = np.mean((Ysd-DeltaMCte)**2)
        return meanmsemetric,sdmsemetric,chi2metric

    def predict_heteroscedastic(self,X):
        Mte = np.array([self.mean_surrogate_model(x) for x in X])
        Zbar, Zv = self.modelz.predict(np.array(X))
        Zmean = np.array([z[0] for z in Zbar])
        Zvar = np.array([z[0] for z in Zv])
        Vmean = np.exp(Zmean + (Zvar / 2))

        ybar, vy = self.modely._raw_predict(np.array(X))

        Ymean = np.array([y[0] for y in ybar])
        Ymean += Mte

        Yvar = np.array([y[0] for y in vy])
        Yvar += Vmean
        Ysd = np.sqrt(Yvar)

        return Ymean,Ysd

    def predict_homoscedastic(self,X):
        Mte = np.array([self.mean_surrogate_model(x) for x in X])

        ybar, vy = self.modely.predict(np.array(X))

        Ymean = np.array([y[0] for y in ybar])
        Ymean += Mte

        Yvar = np.array([y[0] for y in vy])
        Ysd = np.sqrt(Yvar)

        return Ymean,Ysd

    def f_x(self,x):
        if hasattr(self, 'modelz') and self.modelz is not None:
            (Ymean, Ysd) = self.predict_heteroscedastic([x])
        else:
            (Ymean, Ysd) = self.predict_homoscedastic([x])
        return Ymean[0], Ysd[0]

    def f_X(self,X):
        if hasattr(self, 'modelz') and self.modelz is not None:
            return self.predict_heteroscedastic(X)
        else:
            return self.predict_homoscedastic(X)
