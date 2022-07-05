import pprint

import numpy as np

import apprentice
from apprentice.surrogatemodel import SurrogateModel
from apprentice.space import Space
from scipy.optimize import minimize
from timeit import default_timer as timer

#https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
def timeit(method):
    import time
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed

class RationalApproximation(SurrogateModel):
    __allowed = ("m_", "m", "n_", "n", "pcoeff","qcoeff",
                 "pcoeff_","qcoeff_",'training_size',
                 "training_size_", "pnames","pnames_", "strategy_", "strategy",
                 'scale_min_', 'scale_max_','scale_min', 'scale_max'

                 'ftol_','iterations_','use_abstract_model_','tmpdir_','fit_solver_','debug_',
                 'ftol','iterations','use_abstract_model','tmpdir','fit_solver','debug',

                 'local_solver_','max_restarts_', 'threshold_',
                 'local_solver','max_restarts', 'threshold',)

    def __init__(self,dim, fnspace=None, **kwargs: dict):
        super().__init__(dim, fnspace)
        for k, v in kwargs.items():
            if k in ['m','n','training_size',"pcoeff","qcoeff",'strategy','ftol','iterations','use_abstract_model',
                     'tmpdir','fit_solver','debug',
                     'local_solver','max_restarts', 'threshold']:
                k+="_"
            elif k in ['pnames','pnames_','scale_min','scale_max','scale_min_', 'scale_max_']: continue
            assert (k in self.__class__.__allowed)
            setattr(self,k, v)

        if self.dim==1: self.recurrence=apprentice.monomial.recurrence1D
        else           : self.recurrence=apprentice.monomial.recurrence
        self.set_structures()

    # def __init__(self, X=None, Y=None, order=(2,1), fname=None, initDict=None, strategy=1, scale_min=-1, scale_max=1, pnames=None, set_structures=True):
    #     """
    #     Multivariate rational approximation f(x)_mn =  g(x)_m/h(x)_n
    #
    #     kwargs:
    #         fname --- to read in previously calculated Pade approximation
    #å
    #         X     --- anchor points
    #         Y     --- function values
    #         order --- tuple (m,n) m being the order of the numerator polynomial --- if omitted: auto
    #     """
    #     self._vmin=None
    #     self._vmax=None
    #     self._xmin = None
    #     self._xmax = None
    #     if initDict is not None:
    #         self.mkFromDict(initDict, set_structures=set_structures)
    #     elif fname is not None:
    #         self.mkFromJSON(fname, set_structures=set_structures)
    #     elif X is not None and Y is not None:
    #         self._m=order[0]
    #         self._n=order[1]
    #         self._scaler = apprentice.Scaler(np.atleast_2d(np.array(X, dtype=np.float64)), a=scale_min, b=scale_max, pnames=pnames)
    #         self._X   = self._scaler.scaledPoints
    #         self._dim = self._X[0].shape[0]
    #         self._Y   = np.array(Y, dtype=np.float64)
    #         self._trainingsize=len(X)
    #         if self._dim==1: self.recurrence=apprentice.monomial.recurrence1D
    #         else           : self.recurrence=apprentice.monomial.recurrence
    #         self.fit(strategy=strategy)
    #     else:
    #         raise Exception("Constructor not called correctly, use either fname, initDict or X and Y")

    @property
    def max_restarts(self):
        if hasattr(self, 'max_restarts_'):
            return self.max_restarts_
        else: return 100

    @property
    def threshold(self):
        if hasattr(self, 'threshold_'):
            return self.threshold_
        else: return 0.02

    @property
    def ftol(self):
        if hasattr(self, 'ftol_'):
            return self.ftol_
        else: return 1e-6

    @property
    def iterations(self):
        if hasattr(self, 'iterations_'):
            return self.iterations_
        else: return 200

    @property
    def tmpdir(self):
        if hasattr(self, 'tmpdir_'):
            return self.tmpdir_
        else: return "/tmp"

    @property
    def debug(self):
        if hasattr(self, 'debug_'):
            return self.debug_
        else: return False

    @property
    def use_abstract_model(self):
        if hasattr(self, 'use_abstract_model_'):
            return self.use_abstract_model_
        return True

    @property
    def dim(self):
        return self.fnspace.dim

    @property
    def pnames(self):
        return self.fnspace.pnames

    @property
    def order_numerator(self):
        if hasattr(self, 'm_'):
            return self.m_
        return 1

    @property
    def order_denominator(self):
        if hasattr(self, 'n_'):
            return self.n_
        return 1

    @property
    def fit_solver(self):
        if hasattr(self, 'fit_solver_'):
            return self.fit_solver_
        return "scipy"

    @property
    def local_solver(self):
        if hasattr(self, 'local_solver_'):
            return self.local_solver_
        return "scipy"

    @property
    def fit_strategy(self):
        strat_num_to_str = {
            "1.1": "m/1",
            "2.1": "m/n pole s1",
            "2.2": "m/n pole s2",
            "2.3": "m/n pole s3",
            "3.1": "m/n polefree sip"
        }
        allowed_nums = [k for k in strat_num_to_str.keys()]
        allowed_str = [v for v in strat_num_to_str.values()]
        if hasattr(self, 'strategy_'):
            if self.strategy_ in allowed_nums:
                return self.strategy_
            elif self.strategy_ in allowed_str:
                return allowed_nums[allowed_str.index(self.strategy_)]
            else:
                raise Exception("Strategy not allowed: {}".format(self.strategy_))
        return "3.1" # default is SIP

    @property
    def coeff_numerator(self):
        if hasattr(self, 'pcoeff_'):
            return np.array(self.pcoeff_)
        raise Exception("Numerator coeffecients cannot be found. Perform a fit first")

    @property
    def coeff_denominator(self):
        if hasattr(self, 'qcoeff_'):
            return np.array(self.qcoeff_)
        raise Exception("Denominator coeffecients cannot be found. Perform a fit first")

    def set_structures(self):
        m=self.order_numerator
        n=self.order_denominator
        self.struct_p_ = apprentice.monomialStructure(self.dim, m)
        self.struct_q_ = apprentice.monomialStructure(self.dim, n)
        from apprentice import tools
        self.M_        = tools.numCoeffsPoly(self.dim, m)
        self.N_        = tools.numCoeffsPoly(self.dim, n)
        self.K_ = 1 + m + n

    @property
    def M(self): return self.M_
    @property
    def N(self): return self.N_


    # @timeit
    def coeff_solve(self, VM, VN, Y):
        """
        This does the solving for the numerator and denominator coefficients
        following Anthony's recipe.
        """
        Fmatrix=np.diag(Y)
        # rcond changes from 1.13 to 1.14
        rcond = -1 if np.version.version < "1.15" else None
        # Solve VM x = diag(Y)
        MM, res, rank, s  = np.linalg.lstsq(VM, Fmatrix, rcond=rcond)

        Zmatrix = MM.dot(VN)
        # Solve (VM Z - F VN)x = 0
        U, S, Vh = np.linalg.svd(VM.dot(Zmatrix) - Fmatrix.dot(VN))
        self.qcoeff_ = Vh[-1] # The column of (i.e. row of Vh) corresponding to the smallest singular value is the least squares solution
        self.pcoeff_ = Zmatrix.dot(self.coeff_denominator)

    # @timeit
    def coeff_solve2(self, VM, VN, Y):
        """
        This does the solving for the numerator and denominator coefficients.
        F = p/q is reformulated as 0 = p - qF using the VanderMonde matrices.
        That defines the problem Ax = b and we solve for x in an SVD manner,
        exploiting A = U x S x V.T
        There is an additional manipulation exploiting on setting the constant
        coefficient in q to 1.
        """
        FQ = - (VN.T * Y).T # This is something like -F*q
        A = np.hstack([VM, FQ[:,1:]]) # Note that we leave the b0 terms out when defining A
        U, S, Vh = np.linalg.svd(A)
        # Given A = U Sigma VT, for A x = b, it follows, that: x = V Sigma^-1 UT b
        # b really is b0 * F but we explicitly choose b0 to be 1
        # The solution formula is taken from numerical recipes
        UTb = np.dot(U.T, Y.T)[:S.size]
        x = np.dot(Vh.T, 1./S * UTb)
        self.pcoeff_ = x[:self.M]
        self.qcoeff_ = np.concatenate([[1],x[self.M:]]) # b0 is set to 1 !!!

    # @timeit
    def coeff_solve3(self, VM, VN, Y):
        """
        This does the solving for the numerator and denominator coefficients.
        F = p/q is reformulated as 0 = p - qF using the VanderMonde matrices.
        That defines the problem Ax = 0 and we solve for x in an SVD manner,
        exploiting A = U x S x V.T
        We get the solution as the last column in V (corresponds to the smallest singular value)
        """
        FQ = - (VN.T * Y).T
        A = np.hstack([VM, FQ])
        U, S, Vh = np.linalg.svd(A)
        self.pcoeff_ = Vh[-1][:self.M]
        self.qcoeff_ = Vh[-1][self.M:]

    def fit_order_denominator_one(self,X,Y):
        def fit_order_one():
            def scipyfit(coeffs0, cons, iprint=3):
                def fast_leastSqObj(coeff, ipop, ipoq, M, N, Y):
                    return np.sum(np.square(Y * np.sum(coeff[M:M+N] * ipoq, axis=1) - np.sum(coeff[:M] * ipop, axis=1)))
                def fast_jac(coeff, ipop, ipoq, M, N, Y):
                    """
                    Exact analytic gradient of fast_leastSqObj
                    """
                    core = Y * np.sum(coeff[M:M+N] * ipoq, axis=1) - np.sum(coeff[:M] * ipop, axis=1)
                    grads = np.empty_like(coeff)

                    pgrad = -2*np.sum(core[:,np.newaxis]                    *ipop, axis=0)
                    qgrad =  2*np.sum(core[:,np.newaxis] * Y[:,np.newaxis] * ipoq, axis=0)

                    grads[:M]    = pgrad
                    grads[M:M+N] = qgrad

                    return grads


                self.ipo_           = np.empty((self.training_size,2), "object")
                for i in range(self.training_size):
                    self.ipo_[i][0] = self.recurrence(X[i,:],self.struct_p_)
                    self.ipo_[i][1] = self.recurrence(X[i,:],self.struct_q_)
                start = timer()
                ipop = np.array([self.ipo_[i][0] for i in range(self.training_size)])
                ipoq = np.array([self.ipo_[i][1] for i in range(self.training_size)])

                ret = minimize(fast_leastSqObj, coeffs0 , args=(ipop, ipoq, self.M, self.N, Y),
                               jac=fast_jac, method = 'SLSQP', constraints=cons,
                               options={'maxiter': self.iterations, 'ftol': self.ftol, 'disp': self.debug, 'iprint': iprint})
                end = timer()
                optstatus = {'message':ret.get('message'),'status':ret.get('status'),'noOfIterations':ret.get('nit'),'time':end-start}
                return ret.get('x'), ret.get('fun'), optstatus

            def constraintOrder1V(coeff, M, N, L, U):
                """ Inequality constraints for the order 1 denominator case """
                b = coeff[M]
                v = coeff[M+N:M+N+N-1]
                w = coeff[M+N+N-1:]

                c = np.zeros(2*N-1)
                c[0] = b + np.dot(v, L) - np.dot(w,U) - 1e-6
                c[1:] = coeff[M+N:]
                return c

            def constraintAVW(coeff, M, N):
                """ Equality constraints for the order 1 denominator case """
                a = coeff[M+1:M+N]
                v = coeff[M+N:M+N+N-1]
                w = coeff[M+N+N-1:]
                return a - v + w

            """ The dual problem for order 1 denominator polynomials """
            coeffs0 = np.random.random(self.M+3*self.N-2)

            cons = np.empty(0, "object")
            cons = np.append(cons, {'type': 'ineq', 'fun':constraintOrder1V, 'args':(self.M, self.N, self.function_space.sa_, self.function_space.sb_)})
            # cons = np.append(cons, {'type': 'ineq', 'fun':constraintOrder1V, 'jac':localGG, 'args':(self.M, self.N, self._scaler._a, self._scaler._b)})
            # cons = np.append(cons, {'type': 'ineq', 'fun':constraintOrder1V, 'jac':constraintOrder1G, 'args':(self.M, self.N, self._scaler._a, self._scaler._b)})
            cons = np.append(cons, {'type': 'eq', 'fun':constraintAVW, 'args':(self.M, self.N)})

            coeffs, leastSq, optstatus = scipyfit(coeffs0, cons)

            self.pcoeff_ = coeffs[:self.M]
            self.qcoeff_ = coeffs[self.M:self.M+self.N]

        def fit_order_one_ampl_abstract(model,fit_solver):
            from pyomo import environ

            trainingset = [i for i in range(self.training_size)]
            Ydict = dict(zip(trainingset, Y.T))
            # print(trset,Ydict)
            input_data = {
                None: {
                    "trainingsizeset": {None: trainingset},
                    'Y': Ydict
                }
            }
            instance = model.create_instance(input_data)
            opt = environ.SolverFactory(fit_solver)
            plevel = 5
            if not self.debug:
                plevel = 1

            import os
            from pyomo.common.tempfiles import TempfileManager
            TempfileManager.tempdir = self.tmpdir
            if not os.path.exists(self.tmpdir):
                os.makedirs(self.tmpdir)
            # self.logfp = "/tmp/log.log"
            if self.debug:
                instance.pprint()

            isDone=False
            while not isDone:
                try:
                    ret = opt.solve(instance,
                                    tee=False,
                                    # tee=True,
                                    # logfile=self.logfp,
                                    keepfiles=False,
                                    # options={'file_print_level': 5, 'print_level': plevel}
                                    )
                    isDone=True
                except Exception as e:
                    print("{} --- retrying ...".format(e))
                    pass

            if self.debug:
                ret.write()

            if self.debug:
                print(np.array([instance.pcoeff[i].value for i in instance.prange]))
                print(np.array([instance.qcoeff[i].value for i in instance.qrange]))
            self.pcoeff_ = np.array([instance.pcoeff[i].value for i in instance.prange])
            self.qcoeff_ = np.array([instance.qcoeff[i].value for i in instance.qrange])

        def fit_order_one_ampl(model,fit_solver):
            from pyomo import environ
            isDone = False
            ntries = 0
            plevel = 5
            if not self.debug:
                plevel = 1

            import os
            if not os.path.exists(self.tmpdir):
                os.makedirs(self.tmpdir)
            from pyomo.common.tempfiles import TempfileManager
            TempfileManager.tempdir = self.tmpdir
            # self.logfp = "/tmp/log.log"
            if self.debug:
                model.pprint()
            while not isDone:
                opt = environ.SolverFactory(fit_solver)

                try:
                    ret = opt.solve(model,
                                    tee=False,
                                    # tee=True,
                                    # logfile=self.logfp,
                                    keepfiles=False,
                                    # options={'file_print_level': 5, 'print_level': plevel}
                                    )
                    isDone=True
                except Exception as e:
                    ntries+=1
                    if ntries > 5:
                        fit_solver = 'ipopt'
                    print("{} --- retrying ...".format(e))
                    pass

            if self.debug:
                ret.write()

            if self.debug:
                print(np.array([model.pcoeff[i].value for i in model.prange]))
                print(np.array([model.qcoeff[i].value for i in model.qrange]))
            self.pcoeff_ = np.array([model.pcoeff[i].value for i in model.prange])
            self.qcoeff_ = np.array([model.qcoeff[i].value for i in model.qrange])

        def create_order_one_model(abstract=False):
            from pyomo import environ

            def lsqObj(model):
                thesum = 0
                for index in range(model.trainingsize):
                    p_ipo = model.pipo[index]
                    q_ipo = model.qipo[index]
                    P = sum([model.pcoeff[i] * p_ipo[i] for i in model.prange])
                    Q = sum([model.qcoeff[i] * q_ipo[i] for i in model.qrange])

                    thesum += (model.Y[index] * Q - P) ** 2
                return thesum

            def constraintOrder1V(model):
                b = model.qcoeff[0]
                v = model.vcoeff[:]
                w = model.wcoeff[:]

                vL = sum([x * y for x, y in zip(v, model.L)])
                wU = sum([x * y for x, y in zip(w, model.U)])

                return b + vL - wU >= 1e-6

            def constraintAVW(model, index):
                a = model.qcoeff[1 + index]
                v = model.vcoeff[index]
                w = model.wcoeff[index]

                return a == v - w

            if abstract:
                model = environ.AbstractModel()
            else:
                model = environ.ConcreteModel()
            model.dimrange = range(self.dim)

            model.prange = range(self.M)
            model.qrange = range(self.N)
            model.vrange = range(self.N - 1)
            model.wrange = range(self.N - 1)

            model.M = self.M
            model.N = self.N
            model.trainingsize = self.training_size

            model.pipo = self.ipo_[:, 0]
            model.qipo = self.ipo_[:, 1]

            if abstract:
                model.trainingsizeset = environ.Set()
                model.Y = environ.Param(model.trainingsizeset)
            else:
                model.Y = Y

            model.L = self.function_space.sa_
            model.U = self.function_space.sb_

            model.pcoeff = environ.Var(model.prange, initialize=1.)
            model.qcoeff = environ.Var(model.qrange, initialize=1.)
            model.vcoeff = environ.Var(model.vrange, bounds=(0, None), initialize=0.)
            model.wcoeff = environ.Var(model.wrange, bounds=(0, None), initialize=0.)

            model.obj = environ.Objective(rule=lsqObj, sense=1)
            model.constraintOrder1V = environ.Constraint(rule=constraintOrder1V)
            model.constraintAVW = environ.Constraint(model.vrange, rule=constraintAVW)

            return model

        if self.order_denominator != 1:
            raise Exception("Incorrect fitiing strategy: {} used for "
                            "denominator of order "
                            "{}".format(self.fit_strategy,self.coeff_denominator))
        self.ipo_           = np.empty((self.training_size,2), "object")
        for i in range(self.training_size):
            self.ipo_[i][0] = self.recurrence(X[i,:],self.struct_p_)
            self.ipo_[i][1] = self.recurrence(X[i,:],self.struct_q_)
        if self.fit_solver!= "scipy":
            if self.use_abstract_model is not False:
                if not hasattr(self, 'abstract_model_'):
                    self.abstract_model_ = create_order_one_model(abstract=True)
                fit_order_one_ampl_abstract(model=self.abstract_model_,fit_solver=self.fit_solver)
            else:
                self.abstract_model_ = None
                model = create_order_one_model(abstract=False)
                fit_order_one_ampl(model=model,fit_solver=self.fit_solver)
        else: fit_order_one()

    def fit_sip(self, X,Y):
        def fast_robustSampleV(coeff, q_ipo, M, N):
            return np.sum(coeff[M:M+N] * q_ipo, axis=1) - 1.0

        def fast_robustSample(coeff, q_ipo, M, N):
            return np.sum(coeff[M:M+N] * q_ipo) - 1.0

        def fast_leastSqObj(coeff, ipop, ipoq, M, N, Y):
            return np.sum(np.square(Y * np.sum(coeff[M:M+N] * ipoq, axis=1) - np.sum(coeff[:M] * ipop, axis=1)))

        def fast_jac(coeff, ipop, ipoq, M, N, Y):
            """
            Exact analytic gradient of fast_leastSqObj
            """
            core = Y * np.sum(coeff[M:M+N] * ipoq, axis=1) - np.sum(coeff[:M] * ipop, axis=1)
            grads = np.empty_like(coeff)

            pgrad = -2*np.sum(core[:,np.newaxis]                    *ipop, axis=0)
            qgrad =  2*np.sum(core[:,np.newaxis] * Y[:,np.newaxis] * ipoq, axis=0)

            grads[:M]    = pgrad
            grads[M:M+N] = qgrad

            return grads
        def scipyfit(coeffs0, cons, maxiter=1001, ftol=1e-9, iprint=2):
            start = timer()
            ipop = np.array([self.ipo_[i][0] for i in range(self.training_size)])
            ipoq = np.array([self.ipo_[i][1] for i in range(self.training_size)])
            ret = minimize(fast_leastSqObj, coeffs0 , args=(ipop, ipoq, self.M, self.N, Y),
                           jac=fast_jac, method = 'SLSQP', constraints=cons,
                           options={'maxiter': maxiter, 'ftol': ftol, 'disp': self.debug, 'iprint': iprint})
            end = timer()
            optstatus = {'message':ret.get('message'),'status':ret.get('status'),'noOfIterations':ret.get('nit'),'time':end-start}
            return ret.get('x'), ret.get('fun'), optstatus

        def pyomofit(iterationNo, solver='ipopt'):
            from pyomo import environ

            def lsqObjPyomo(model):
                sum = 0
                # sigma = 0 #only strategy 0 of SIP is implemented in this class
                for index in range(model.trainingsize):
                    p_ipo = model.pipo[index]
                    q_ipo = model.qipo[index]

                    P=0
                    coeffsump=0
                    for i in model.prange:
                        P += model.coeff[i]*p_ipo[i]
                        coeffsump+=model.coeff[i]**2

                    Q=0
                    coeffsumq=0
                    for i in model.qrange:
                        Q += model.coeff[i]*q_ipo[i-model.M]
                        coeffsumq+=model.coeff[i]**2

                    sum += (model.Y[index] * Q - P)**2 # sigma = 0
                    # sum += (model.Y[index] * Q - P)**2 + sigma*(coeffsump + coeffsumq)
                return sum

            def robustConstrPyomo(model,index):
                q_ipo = model.qipo[index]

                Q=0
                for i in model.qrange:
                    Q += model.coeff[i]*q_ipo[i-model.M]
                return Q >= 0.2

            # concrete model
            model = environ.ConcreteModel()
            model.dimrange = range(self.dim)
            model.prange = range(self.M)
            model.qrange = range(self.M,self.M+self.N)
            model.coeffrange = range(self.M+self.N)
            model.M = self.M
            model.N = self.N
            model.trainingsize = self.training_size
            model.trainingsizerangeforconstr = range(self.training_size+iterationNo)


            model.pipo = self.ipo_[:,0].tolist()
            model.qipo = self.ipo_[:,1].tolist()

            model.Y = Y.tolist()

            model.coeff = environ.Var(model.coeffrange)
            for d in range(self.M + self.N): model.coeff[d].value = 1
            model.coeff[self.M].value = 2

            model.lsqfit = environ.Objective(rule=lsqObjPyomo, sense=1)

            model.robustConstr = environ.Constraint(model.trainingsizerangeforconstr, rule=robustConstrPyomo)

            opt = environ.SolverFactory(solver)
            # opt = environ.SolverFactory('ipopt')
            # opt.options['eps'] = 1e-10
            # opt.options['iprint'] = 1

            # pyomodebug = self._fitpyomodebug
            # if(pyomodebug == 0):
            # ret = opt.solve(model)
            from pyomo.common.tempfiles import TempfileManager
            TempfileManager.tempdir = self.tmpdir
            ret = opt.solve(model,
                            tee=False,
                            # tee=True,
                            # logfile=self.logfp,
                            keepfiles=False,
                            # options={'file_print_level': 5, 'print_level': plevel}
                            )

            optstatus = {'message':str(ret.solver.termination_condition),'status':str(ret.solver.status),'time':ret.solver.time,'error_rc':ret.solver.error_rc}

            coeffs = np.array([model.coeff[i].value for i in range(self.M + self.N)])
            leastSq = environ.value(model.lsqfit)

            return coeffs,leastSq,optstatus

        def multiple_restart_for_iter_rob_o(coeffs):
            def rob_o_pyomo(coeffs, solver='ipopt',r=0):
                from pyomo import environ

                def robObjPyomo(model):
                    res = 0
                    for l in range(len(model.struct_q)):
                        mon = model.struct_q[l]
                        term = 1
                        for d in model.dimrange:
                            try:
                                exp = mon[d]
                            except:
                                exp = mon
                            term *= model.x[d] ** exp
                        res += model.coeffs[l+model.M] * term
                    return res

                def variableBound(model, i):
                    b = (model.box[i][0], model.box[i][1])
                    return b

                model = environ.ConcreteModel()
                model.struct_q = self.struct_q_.tolist()
                model.coeffs = coeffs.tolist()
                model.dimrange = range(self.dim)
                model.box = self.function_space.box.tolist()
                model.M = self.M
                model.x = environ.Var(model.dimrange, bounds=variableBound)
                if(r == 0):
                    for d in range(self.dim):
                        model.x[d].value = (self.function_space.box[d][0]+self.function_space.box[d][1])/2
                else:
                    for d in range(self.dim):
                        model.x[d].value = np.random.rand()*(self.function_space.box[d][1]-self.function_space.box[d][0])+self.function_space.box[d][0]
                model.robO = environ.Objective(rule=robObjPyomo, sense=1)
                opt = environ.SolverFactory(solver)

                """
                Control where the log file is written by passing logfile=<name>
                to the solve method.
        
                If you want to print solver log to console, add tee=True to solve method
        
                If you want the solution and problem files to be logged,
                you can set keepfiles=True for that file to not be deleted.
        
                Also, if you set keepfiles to True, you can find the location of Solver log file,
                Solver problem files, and Solver solution file printed on console (usually
                located in /var/folders/)
                """
                from pyomo.common.tempfiles import TempfileManager
                TempfileManager.tempdir = self.tmpdir
                ret = opt.solve(model,
                                tee=False,
                                # tee=True,
                                # logfile=self.logfp,
                                keepfiles=False,
                                # options={'file_print_level': 5, 'print_level': plevel}
                                )
                optstatus = {'message':str(ret.solver.termination_condition),'status':str(ret.solver.status),'time':ret.solver.time,'error_rc':ret.solver.error_rc}
                x = np.array([model.x[i].value for i in range(self.dim)])
                robO = environ.value(model.robO)
                return x, robO, optstatus
            def restart_rob_o_scipy(coeffs, solver, r):
                def robust_objective(x, coeff):
                    q_ipo = self.recurrence(x, self.struct_q_)
                    return np.dot(coeff[self.M:], q_ipo)
                x0 = []
                if(r == 0):
                    x0 = np.array([(self.function_space.box[i][0]+self.function_space.box[i][1])/2 for i in range(self.dim)], dtype=np.float64)
                else:
                    x0 = np.zeros(self.dim, dtype=np.float64)
                    for d in range(self.dim):
                        x0[d] = np.random.rand()*(self.function_space.box[d][1]-self.function_space.box[d][0])+self.function_space.box[d][0]
                start = timer()
                ret = minimize(robust_objective, x0, bounds=self.function_space.box, args = (coeffs,),method = solver, options={'maxiter': 1000,'ftol': 1e-4, 'disp': False})
                end = timer()
                optstatus = {'message':ret.get('message'),'status':ret.get('status'),'noOfIterations':ret.get('nit'),'time':end-start}

                x = ret.get('x')
                robO = ret.get('fun')
                return x, robO, optstatus
            minx = []
            restartInfo = []
            minrobO = np.inf
            totaltime = 0
            norestarts = 0
            localoptfunc = ""
            if(self.local_solver == 'scipy'):
                solver = 'L-BFGS-B'
                localoptfunc = restart_rob_o_scipy
            elif(self.local_solver == 'filter' or self.local_solver == 'ipopt'):
                solver = self.local_solver
                localoptfunc = rob_o_pyomo
            else:raise Exception("localoptsolver, %s unknown"%self.local_solver)

            for r in range(self.max_restarts):
                x, robO, optstatus = localoptfunc(coeffs,solver,r)
                totaltime += optstatus['time']

                if(minrobO > robO):
                    minrobO = robO
                    minx = x
                rinfo = {'robustArg':x.tolist(),'robustObj':robO, 'log':optstatus}
                restartInfo.append(rinfo)
                norestarts += 1
                if(robO < self.threshold):
                    break
            restartInfo.append({'log':{'time':totaltime, 'noRestarts':norestarts}})
            return minx, minrobO, restartInfo
        # Only strategy 0 implemented
        # Fitsolver comes through self.fit_solver
        # localsolver is an additional argument self.lcoal_solver
        # roboptstrategy is always MS (Multistart)
        self.ipo_            = np.empty((self.training_size,2), "object")
        for i in range(self.training_size):
            self.ipo_[i][0] = self.recurrence(X[i,:],self.struct_p_)
            self.ipo_[i][1] = self.recurrence(X[i,:],self.struct_q_)
        self.iterationinfo_ = []
        ipoq = np.array([self.ipo_[i][1] for i in range(self.training_size)])
        cons = np.empty(0, "object")
        cons = np.append(cons, {'type': 'ineq', 'fun':fast_robustSampleV, 'args':(ipoq, self.M, self.N)})
        for iter in range(1, self.iterations+1):
            data = {}
            data['iterationNo'] = iter
            if self.debug:
                print("Starting lsq for iter %d"%(iter))
            if(self.fit_solver == 'scipy'):
                coeffs0 = np.ones((self.M+self.N))
                coeffs, leastSq, optstatus = scipyfit(coeffs0, cons)
                if optstatus['status']!=0:
                    fixme=True
                    while fixme:
                        coeffs0 = np.random.random(coeffs0.shape)
                        coeffs, leastSq, optstatus = scipyfit(coeffs0, cons)
                        if optstatus['status']==0:
                            fixme=False
            elif(self.fit_solver == 'filter' or self.fit_solver == 'ipopt'):
                coeffs, leastSq, optstatus = pyomofit(iter-1,self.fit_solver)
            else:raise Exception("Fit Solver %s not implemented"%self.fit_solver)
            data['log'] = optstatus
            data['leastSqObj'] = leastSq
            data['pcoeff'] = coeffs[0:self.M].tolist()
            data['qcoeff'] = coeffs[self.M:self.M+self.N].tolist()

            if self.debug:
                print('starting Multistart local optimization for iteration {}'.format(iter))
            x, robO, restartInfo = multiple_restart_for_iter_rob_o(coeffs)
            data['robOptInfo'] = {'robustArg':x.tolist(),'robustObj':robO,'info':restartInfo}

            self.iterationinfo_.append(data)

            if(robO >= self.threshold):
                break

            q_ipo_new = self.recurrence(x,self.struct_q_)
            if(self.fit_solver == 'scipy'):
                cons = np.append(cons, {'type': 'ineq', 'fun':fast_robustSample, 'args':(q_ipo_new, self.M, self.N)})
            elif(self.fit_solver == 'filter' or self.fit_solver == 'ipopt'):
                self.ipo_ = np.vstack([self.ipo_, np.empty((1,2),"object")])
                self.ipo_[self.training_size+iter-1][1] = q_ipo_new

        if(len(self.iterationinfo_) == self.iterations and self.iterationinfo_[self.iterations-1]['robOptInfo']["robustObj"]<self.threshold):
            import json
            j = json.dumps(self.iterationinfo_,indent=4, sort_keys=True)
            raise Exception(j+"\nCould not find a robust objective")
        self.pcoeff_ = np.array(self.iterationinfo_[len(self.iterationinfo_)-1]["pcoeff"])
        self.qcoeff_ = np.array(self.iterationinfo_[len(self.iterationinfo_)-1]["qcoeff"])
        # End of fit_sip

    def fit(self, X, Y):
        """
        Do everything.
        """
        X = self.fnspace.scale(X)
        # Set M, N, K, polynomial structures
        m = self.order_numerator
        n = self.order_denominator
        from apprentice import tools
        n_required = tools.numCoeffsRapp(self.dim, (m, n))
        if n_required > Y.shape[0]:
            raise Exception("Not enough inputs: got %i but require %i to do m=%i n=%i"%(Y.shape[0], n_required, m,n))

        self.set_structures()

        strategy = self.fit_strategy
        if strategy=="1.1": self.fit_order_denominator_one(X,Y)
        elif "2." in strategy:
            from apprentice import monomial
            VM = monomial.vandermonde(X, m)
            VN = monomial.vandermonde(X, n)
            if   strategy=="2.1": self.coeff_solve( VM, VN, Y)
            elif strategy=="2.2": self.coeff_solve2(VM, VN, Y)
            elif strategy=="2.3": self.coeff_solve3(VM, VN, Y)
        elif strategy=="3.1": self.fit_sip(X,Y)
        # NOTE, strat 1 is faster for smaller problems (Npoints < 250)
        else: raise Exception("fit() strategy %i not implemented"%strategy)

    def Q(self, X):
        """
        Evaluation of the denom poly at X.
        """
        rec_q = np.array(self.recurrence(X, self.struct_q_))
        q = self.coeff_denominator.dot(rec_q)
        return q

    def denom(self, X):
        """
        Alias for Q, for compatibility
        """
        return self.Q(X)

    def P(self, X):
        """
        Evaluation of the numer poly at X.
        """
        rec_p = np.array(self.recurrence(X, self.struct_p_))
        p = self.coeff_numerator.dot(rec_p)
        return p

    def predict(self, X):
        """
        Return the prediction of the RationalApproximation at X.
        """
        X=self.function_space.scale(np.array(X))
        return self.P(X)/self.Q(X)


    def gradient(self, X):
        import numpy as np
        struct_p = np.array(self.struct_p_, dtype=float)
        struct_q = np.array(self.struct_q_, dtype=float)
        X = self.function_space.scale(np.array(X))
        p = self.P(X)
        q = self.Q(X)

        if self.dim==1:
            # p'
            struct_p[1:]=self.function_space.jacfac[0]*struct_p[1:]*np.power(X, struct_p[1:]-1)
            pprime = np.dot(np.atleast_2d(struct_p),self.coeff_numerator)
            # q'
            struct_q[1:]=self.function_space.jacfac[0]*struct_q[1:]*np.power(X, struct_q[1:]-1)
            qprime = np.dot(np.atleast_2d(struct_q),self.coeff_denominator)
        else:
            from apprentice.tools import gradientRecursion
            GRECP = gradientRecursion(X, struct_p, self.function_space.jacfac)
            pprime = np.sum(GRECP * self.coeff_numerator, axis=1)
            GRECQ = gradientRecursion(X, struct_q, self.function_space.jacfac)
            qprime = np.sum(GRECQ * self.coeff_denominator, axis=1)


        return pprime/q - p/q/q*qprime

    def f_x(self, x):
        """
        Operator version of predict.
        """
        return self.predict(x)

    def f_X(self,X):
        return [self.predict(x) for x in X]

    def __repr__(self):
        """
        Print-friendly representation.
        """
        return "<RationalApproximation dim:{} m:{} n:{}>".format(self.dim, self.order_numerator, self.order_denominator)

    @property
    def as_dict(self):
        """
        Store all info in dict as basic python objects suitable for JSON
        """
        d={}
        d["m"] = self.order_numerator
        d["n"] = self.order_denominator
        d["training_size"] = self.training_size
        d["strategy"] = self.fit_strategy
        d["pcoeff"] = list(self.coeff_numerator)
        d["qcoeff"] = list(self.coeff_denominator)
        d["fnspace"] = self.fnspace.as_dict
        return d

    def save(self, fname):
        import json
        with open(fname, "w") as f:
            json.dump(self.as_dict, f,indent=4)

    @classmethod
    def from_data_structure(cls,data_structure):
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

        return cls(dim,fnspace,**data_structure)

    # def mkFromDict(self, pdict, set_structures=True):
    #     self._pcoeff = np.array(pdict["pcoeff"])
    #     self._qcoeff = np.array(pdict["qcoeff"])
    #     self._m      = int(pdict["m"])
    #     self._n      = int(pdict["n"])
    #     self._dim    = int(pdict["dim"])
    #     self._scaler = apprentice.Scaler(pdict["scaler"])
    #     if "vmin" in pdict: self._vmin = pdict["vmin"]
    #     if "vmax" in pdict: self._vmax = pdict["vmax"]
    #     if self._dim==1: self.recurrence=apprentice.monomial.recurrence1D
    #     else           : self.recurrence=apprentice.monomial.recurrence
    #     try:
    #         self._trainingsize = int(pdict["trainingsize"])
    #     except:
    #         pass
    #     if set_structures: self.setStructures()

    @classmethod
    def from_file(cls, filepath):
        import json
        with open(filepath,'r') as f:
            d = json.load(f)
        return cls.from_data_structure(d)

    # def mkFromJSON(self, fname, set_structures=True):
    #     import json
    #     d = json.load(open(fname))
    #     self.mkFromDict(d, set_structures=set_structures)

    # def fmin(self, nsamples=1, nrestart=1, use_grad=False):
    #     return apprentice.tools.extreme(self, nsamples, nrestart, use_grad, mode="min")
    #
    # def fmax(self, nsamples=1, nrestart=1, use_grad=False):
    #     return apprentice.tools.extreme(self, nsamples, nrestart, use_grad, mode="max")

    @property
    def coeff_norm(self):
        nrm = 0
        for p in self.coeff_numerator:
            nrm+= abs(p)
        for q in self.coeff_denominator:
            nrm+= abs(q)
        return nrm

    @property
    def coeff2_norm(self):
        nrm = 0
        for p in self.coeff_numerator:
            nrm+= p*p
        for q in self.coeff_denominator:
            nrm+= q*q
        return np.sqrt(nrm)

    # def wraps(self, v):
    #     dec=True
    #     if self.vmin is not None and self.vmax is not None:
    #         if self.vmin > v or self.vmax < v:dec=False
    #     return dec

# if __name__=="__main__":
#
#     import sys
#
#     def mkTestData(NX, dim=1):
#         def anthonyFunc(x):
#             return (10*x)/(x**3 - 4* x + 5)
#         NR = 1
#         np.random.seed(555)
#         X = sorted(np.random.rand(NX, dim))
#         Y = np.array([anthonyFunc(*x) for x in X])
#         return X, Y
#
#     X, Y = mkTestData(500)
#
#     import pylab
#     pylab.plot(X, Y, marker="*", linestyle="none", label="Data")
#     pp=RationalApproximation(X,Y, order=(1,3))
#     myg = [pp.gradient(x) for x in X]
#
#     # import time
#     # t0=time.time()
#     # for _ in range(10000): pp.gradient(5)
#     # t1=time.time()
#     # print(t1-t0)
#
#     try:
#         import autograd.numpy as np
#         from autograd import hessian, grad
#         g = grad(pp)
#         # t2=time.time()
#         # for _ in range(10000): g(5.)
#         # t3=time.time()
#         # print("{} vs. {}, ratio {}".format(t1-t0, t3-t2, (t3-t2)/(t1-t0)))
#         G = [g(x) for x in X]
#         pylab.plot(X, G, label="auto gradient")
#     except:
#         pass
#     myg = [pp.gradient(x) for x in X]
#
#     pylab.plot(X, [pp(x) for x in X], label="Rational approx")
#     # pylab.plot(X, FP, marker="s", linestyle="none", label="analytic gradient")
#     pylab.plot(X, myg, label="manual gradient", linestyle="--")
#     # pylab.plot(X, myg, label="manual gradient")
#     pylab.legend()
#
#     pylab.show()
#     exit(0)
#
#
#     TX = sorted(X)
#     for s in range(1,4):
#         r=RationalApproximation(X,Y, order=(5,5), strategy=s)
#
#         YW = [r(p) for p in TX]
#
#         pylab.plot(TX, YW, label="Rational approx m={} n={} strategy {}".format(2,2,s))
#
#
#
#
#     # Store the last and restore immediately, plot to see if all is good
#     r.save("rapptest.json")
#     r=RationalApproximation(fname="rapptest.json")
#     YW = [r(p) for p in TX]
#     pylab.plot(TX, YW, "m--", label="Restored m={} n={} strategy {}".format(2,2,s))
#     pylab.legend()
#     pylab.xlabel("x")
#     pylab.ylabel("f(x)")
#     pylab.savefig("demo.pdf")
#
#     sys.exit(0)
