import numpy as np
from apprentice import tools
from scipy.optimize import minimize
from timeit import default_timer as timer
import apprentice

# from numba import jit, njit

"""
Print scipy model
"""
def printscipymodel(trainingsize,ipop, ipoq, M, N, Y):
    s = "minimize lsq: \n"
    for index in range(trainingsize):
        p_ipo = ipop[index]
        q_ipo = ipoq[index]
        s += "(%f * ("%(Y[index])
        for i in range(M, M+N):
            if(i!=M): s+=" + "
            s += "coeff[%d]*%f"%(i+1,q_ipo[i-M])
        s+= ") - ("
        for i in range(M):
            if(i!=0): s+=" + "
            s += "coeff[%d]*%f"%(i+1,p_ipo[i])
        s+="))**2"
        if(index != trainingsize-1 and Y[index+1]>=0):
            s+="+"
    s+=";"
    s+="\nsubject to \n"
    for index in range(trainingsize):
        q_ipo = ipoq[index]

        s+="%c: "%(chr(65+index))
        for i in range(M, M+N):
            if(i!=M): s+=" + "
            s += "coeff[%d]*%f"%(i+1,q_ipo[i-M])

        s+=">=1;\n"
    print(s)

# @njit(fastmath=True, parallel=True)
def fast_robustSample_for_fmin_slsqp(coeff, trainingsize, ipop, ipoq, M, N, Y):
    # c=""
    ret = np.zeros(trainingsize,dtype=np.float64)
    for ts in range(trainingsize):
        mysum = 0
        q_ipo = ipoq[ts]
        for i in range(M, M+N):
            mysum += coeff[i]*q_ipo[i-M]
            # c+= " {}*coeff[{}]".format(q_ipo[i-M], i)
        # c+=" -1"
        # print(c)
        ret[ts] = mysum - 1.0
    return ret

def fast_robustSampleV(coeff, q_ipo, M, N):
    return np.sum(coeff[M:M+N] * q_ipo, axis=1) - 1.0

def fast_robustSample(coeff, q_ipo, M, N):
    return np.sum(coeff[M:M+N] * q_ipo) - 1.0

def fast_leastSqObj(coeff, trainingsize, ipop, ipoq, M, N, Y):
    return np.sum(np.square(Y * np.sum(coeff[M:M+N] * ipoq, axis=1) - np.sum(coeff[:M] * ipop, axis=1)))

def fast_jac(coeff, _, ipop, ipoq, M, N, Y):
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

def fast_jac2(coeff, trainingsize, ipop, ipoq, M, N, Y):
    """
    Forward difference gradient of fast_leastSqObj
    """
    f_0 = fast_leastSqObj(coeff, trainingsize, ipop, ipoq, M, N, Y)
    h = 1.5e-8
    jac = np.zeros_like(coeff)

    for num in range(len(coeff)):
        temp=np.zeros_like(coeff)
        temp[num]+=h
        jac[num]=(fast_leastSqObj(coeff+temp, trainingsize, ipop, ipoq, M, N, Y) -f_0)/h
    return jac

def calculateNonLin(dim, n):
    if(n==0): return 0
    from apprentice import tools
    N = tools.numCoeffsPoly(dim, n)
    return N - (dim + 1)

# from sklearn.base import BaseEstimator, RegressorMixin
# class RationalApproximationSIP(BaseEstimator, RegressorMixin):
class RationalApproximationSIP():
    def __init__(self, *args, **kwargs):
        """
        Multivariate rational approximation p(x)_m/q(x)_n

        args:
            fname   --- to read in previously calculated Pade approximation stored as the JSON file

            dict    --- to read in previously calculated Pade approximation stored as the dictionary object obtained after parsing the JSON file

            X       --- anchor points
            Y       --- function values

        kwargs:
            m               --- order of the numerator polynomial --- if omitted: auto 1 used
            n               --- order of the denominator polynomial --- if omitted: auto 1 used
            trainingscale   --- size of training data to use --- if omitted: auto 1x used
                                .5x is the half the numbner of coeffs in numerator and denominator,
                                1x is the number of coeffs in numerator and denominator,
                                2x is twice the number of coeffecients,
                                Cp is 100% of the data
            box             --- box (2D array of dim X [min,max]) within which to perform the approximation --- if omitted: auto dim X [-1, 1] used
            pnames          --- list of parameter names to pass to the scaler to scale
            scalemin        --- scalar or list of shape = dimension of minimum scale value for X --- if omitted: auto -1 used on all dimensions
            scalemax        --- scalar or list of shape = dimension of maximum scale value for X --- if omitted: auto 1 used on all dimensions
            strategy        --- strategy to use --- if omitted: auto 0 used
                                0: min ||f*q(x)_n - p(x)_m||^2_2 sub. to q(x)_n >=1
                                1: min ||f*q(x)_n - p(x)_m||^2_2 sub. to q(x)_n >=1 and some p and/or q coeffecients set to 0
                                2: min ||f*q(x)_n - p(x)_m||^2_2 + lambda*||c_pq||_1 sub. to q(x)_n >=1
            fitstrategy     --- strategy to perform the fitting (least squares and sparse) --- if omitted: auto 'scipy' used
                                scipy: SLSQP optimization solver in scipy (scipy.SLSQP)
                                filter: filterSQP solver through pyomo (REQUIRED: pyomo and filter executable in PATH)
                                ipopt:  IPOPT solver through pyomo (REQUIRED: pyomo and ipopt executable in PATH)
            roboptstrategy  --- strategy to optimize robust objective --- if omitted: auto 'ms' used
                                ss: single start algorithm using scipy.L-BFGS-B local optimizer
                                ms: multistart algorithm (with 10 restarts at random points from the box) using scipy.L-BFGS-B local optimizer
                                msbarontime: multistart algorithm using scipy.L-BFGS-B local optimizer that restarts for the amount of time baron would run for the no. of nonlinearities
                                baron: baron through pyomo (REQUIRED: Pyomo and baron executable in PATH)
                                solve: solve q(x) at random points in the box of X
                                ss_ms_so_ba: runs single start, multistart, baron and solve, and logs the different objective function values obtained
                                mlsl: multi-level single-linkage multistart algorithm from nlopt using nlopt.LD_LBFGS local optimizer
            localoptsolver  --- strategy to perform local optimization in robust optimization with single start and multistart approaches --- if omitted: auto 'scipy' used
                                scipy: L-BFGS-B optimization solver in scipy (scipy.L-BFGS-B)
                                filter: filterSQP solver through pyomo (REQUIRED: pyomo and filter executable in PATH)
            penaltyparam    --- lambda to use for strategy 2 --- if omitted: auto 0.1 used
            penaltybin      --- penalty binary array for numberator and denomintor of the bits to keep active in strategy 1 and put in penalty term for activity 2
                                represented in a 2D array of shape(2,(m/n)+1) where for each numberator and denominator, the bits represent penalized coeffecient degrees and constant (1: not peanlized, 0 penalized)
                                required for strategy 1 and 2

        """
        self._debug = kwargs["debug"] if kwargs.get("debug") is not None else  False
        import os
        if len(args) == 0:
            pass
        else:
            if type(args[0])==dict:
                self.mkFromDict(args[0])
            elif type(args[0]) == str:
                self.mkFromJSON(args[0])
            else:
                # Scaler related kwargs
                _pnames      = kwargs["pnames"]   if kwargs.get("pnames")   is not None else None
                _scale_min   = kwargs["scalemin"] if kwargs.get("scalemin") is not None else -1
                _scale_max   = kwargs["scalemax"] if kwargs.get("scalemax") is not None else  1
                self._scaler = apprentice.Scaler(np.array(args[0], dtype=np.float64), a=_scale_min, b=_scale_max, pnames=_pnames)
                self._X      = self._scaler.scaledPoints
                self._dim = self._X[0].shape[0]
                self._Y      = np.array(args[1], dtype=np.float64)
                if self._dim==1: self.recurrence=apprentice.monomial.recurrence1D
                else           : self.recurrence=apprentice.monomial.recurrence
                self.mkFromData(kwargs=kwargs)

    @property
    def dim(self): return self._dim
    @property
    def M(self): return self._M
    @property
    def N(self): return self._N
    @property
    def m(self): return self._m
    @property
    def n(self): return self._n
    @property
    def trainingscale(self): return self._trainingscale
    @property
    def trainingsize(self): return self._trainingsize
    @property
    def box(self): return self._scaler.box_scaled
    @property
    def strategy(self): return self._strategy
    @property
    def roboptstrategy(self): return self._roboptstrategy
    @property
    def localoptsolver(self): return self._localoptsolver
    @property
    def fitstrategy(self): return self._fitstrategy
    @property
    def penaltyparam(self): return self._penaltyparam
    @property
    def ppenaltybin(self): return self._ppenaltybin
    @property
    def qpenaltybin(self): return self._qpenaltybin
    @property
    def pcoeff(self): return self._pcoeff
    @property
    def qcoeff(self): return self._qcoeff
    @property
    def iterationinfo(self): return self._iterationinfo
    @property
    def fittime(self): return self._fittime

    def mkFromJSON(self, fname):
        import json
        d = json.load(open(fname))
        self.mkFromDict(d)

    def mkFromDict(self, pdict):
        self._scaler        = apprentice.Scaler(pdict["scaler"])
        self._pcoeff        = np.array(pdict["pcoeff"])
        self._qcoeff        = np.array(pdict["qcoeff"])
        self._iterationinfo = pdict["iterationinfo"]
        self._dim           = pdict["dim"]
        if self._dim==1: self.recurrence=apprentice.monomial.recurrence1D
        else           : self.recurrence=apprentice.monomial.recurrence
        self._m             = pdict["m"]
        self._n             = pdict["n"]
        self._M             = pdict["M"]
        self._N             = pdict["N"]
        self._fittime       = pdict["log"]["fittime"]
        self._strategy      = pdict["strategy"]
        self._roboptstrategy= pdict["roboptstrategy"]
        self._localoptsolver= pdict["localoptsolver"]
        self._fitstrategy   = pdict["fitstrategy"]
        self._trainingscale = pdict["trainingscale"]
        self._trainingsize  = pdict["trainingsize"]
        self._penaltyparam  = 0.0

        if(self.strategy ==1 or self.strategy==2):
            self._ppenaltybin = pdict['chosenppenalty']
            self._qpenaltybin = pdict['chosenqpenalty']

        if(self.strategy == 2):
            self._penaltyparam = pdict['lambda']

        self._struct_p      = apprentice.monomialStructure(self.dim, self.m)
        self._struct_q      = apprentice.monomialStructure(self.dim, self.n)

    def mkFromData(self, kwargs):
        """
        Calculate the Pade approximation
        """

        self._dim               = self._X[0].shape[0]

        self._m                 = int(kwargs["m"]) if kwargs.get("m") is not None else 1
        self._n                 = int(kwargs["n"]) if kwargs.get("n") is not None else 1
        self._M                 = tools.numCoeffsPoly(self.dim, self.m)
        self._N                 = tools.numCoeffsPoly(self.dim, self.n)
        self._strategy          = int(kwargs["strategy"]) if kwargs.get("strategy") is not None else 0
        self._dumpaftereachiter = int(kwargs["dumpaftereachiter"]) if kwargs.get("dumpaftereachiter") is not None else  0
        import uuid
        self._uniqueid          = str(uuid.uuid4().fields[-1])

        self._roboptstrategy    = kwargs["roboptstrategy"] if kwargs.get("roboptstrategy") is not None else "ms"
        self._localoptsolver    = kwargs["localoptsolver"] if kwargs.get("localoptsolver") is not None else "scipy"
        self._fitstrategy       = kwargs["fitstrategy"] if kwargs.get("fitstrategy") is not None else "scipy"

        self._fitpyomodebug = kwargs["fitpyomodebug"] if kwargs.get("fitpyomodebug") is not None else 0
        if self._fitpyomodebug == 2:
            self._debugfolder      = kwargs["debugfolder"]
            self._fnname           = kwargs["fnname"]

        self._trainingscale = kwargs["trainingscale"] if kwargs.get("trainingscale") is not None else "1x"
        if(self.trainingscale == ".5x" or self.trainingscale == "0.5x"):
            self._trainingscale = ".5x"
            self._trainingsize = int(0.5*(self.M+self.N))
        elif(self.trainingscale == "1x"):
            self._trainingsize = self.M+self.N
        elif(self.trainingscale == "2x"):
            self._trainingsize = 2*(self.M+self.N)
        elif(self.trainingscale == "Cp"):
            self._trainingsize = len(self._X)

        self._penaltyparam  = kwargs["penaltyparam"] if kwargs.get("penaltyparam") is not None else 0.0

        if(kwargs.get("ppenaltybin") is not None):
            self._ppenaltybin = kwargs["ppenaltybin"]
        elif(self.strategy ==1 or self.strategy==2):
            raise Exception("Binary Penalty for numerator required for strategy 1 and 2")

        if(kwargs.get("qpenaltybin") is not None):
            self._qpenaltybin = kwargs["qpenaltybin"]
        elif(self.strategy ==1 or self.strategy==2):
            raise Exception("Binary Penalty for denomintor equired for strategy 1 and 2")

        self._struct_p      = apprentice.monomialStructure(self.dim, self.m)
        self._struct_q      = apprentice.monomialStructure(self.dim, self.n)

        self._ipo            = np.empty((self.trainingsize,2), "object")
        for i in range(self.trainingsize):
            self._ipo[i][0] = self.recurrence(self._X[i,:],self._struct_p)
            self._ipo[i][1] = self.recurrence(self._X[i,:],self._struct_q)
        start = timer()
        self.fit()
        end = timer()
        self._fittime = end-start

    def scipyfit2(self,coeffs0):
        """
        Test function that uses fmin_slsqp for fitting. The function is slow  but
        is good for debugging - Do not remove
        """
        from scipy.optimize import fmin_slsqp
        start = timer()
        ipop =[self._ipo[i][0] for i in range(self.trainingsize)]
        ipoq =[self._ipo[i][1] for i in range(self.trainingsize)]
        ret = fmin_slsqp
        out,fx,its,imode,smode = fmin_slsqp(fast_leastSqObj, coeffs0 ,
                    args=(self.trainingsize, ipop, ipoq, self.M, self.N, self._Y),
                    f_ieqcons=fast_robustSample_for_fmin_slsqp,fprime=fast_jac,
                    iter=100000,iprint=3,full_output=True, disp=3)
        print(out)
        print(fx)
        print(its,imode,smode)
        exit(1)

    def callbackF(self, coeff):
        """
        Callback function to print obj value per iteration - Do not remove
        """
        ipop =[self._ipo[i][0] for i in range(self.trainingsize)]
        ipoq =[self._ipo[i][1] for i in range(self.trainingsize)]
        #print '{0:4d}   {1: .6E}'.format(self.Nfeval, fast_leastSqObj(coeff, self.trainingsize, ipop, ipoq, self.M, self.N, self._Y))
        self.Nfeval += 1

    def scipyfit(self, coeffs0, cons, maxiter=1001, ftol=1e-9, iprint=2):
        start = timer()
        ipop = np.array([self._ipo[i][0] for i in range(self.trainingsize)])
        ipoq = np.array([self._ipo[i][1] for i in range(self.trainingsize)])
        ret = minimize(fast_leastSqObj, coeffs0 , args=(self.trainingsize, ipop, ipoq, self.M, self.N, self._Y),
                jac=fast_jac, method = 'SLSQP', constraints=cons,
                options={'maxiter': maxiter, 'ftol': ftol, 'disp': self._debug, 'iprint': iprint})
        end = timer()
        optstatus = {'message':ret.get('message'),'status':ret.get('status'),'noOfIterations':ret.get('nit'),'time':end-start}
        return ret.get('x'), ret.get('fun'), optstatus

    # Does 1 fitting for now
    def pyomofit(self, iterationNo, solver='filter'):
        from pyomo import environ

        def lsqObjPyomo(model):
            sum = 0
            sigma = self._penaltyparam
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

                # sum += (model.Y[index] * Q - P)**2 # sigma = 0
                sum += (model.Y[index] * Q - P)**2 + sigma*(coeffsump + coeffsumq)
            return sum

        def robustConstrPyomo(model,index):
            q_ipo = model.qipo[index]

            Q=0
            for i in model.qrange:
                Q += model.coeff[i]*q_ipo[i-model.M]
            return Q >= 0.2

        if(self._strategy != 0):
            raise Exception("strategy %d for fitstrategy, %s not yet implemented"%(self.strategy, self.fitstrategy))

        # concrete model
        model = environ.ConcreteModel()
        model.dimrange = range(self._dim)
        model.prange = range(self.M)
        model.qrange = range(self.M,self.M+self.N)
        model.coeffrange = range(self.M+self.N)
        model.M = self._M
        model.N = self._N
        model.trainingsize = self.trainingsize
        model.trainingsizerangeforconstr = range(self._trainingsize+iterationNo)


        model.pipo = self._ipo[:,0].tolist()
        model.qipo = self._ipo[:,1].tolist()

        model.Y = self._Y.tolist()

        model.coeff = environ.Var(model.coeffrange)
        for d in range(self.M + self.N): model.coeff[d].value = 1
        model.coeff[self.M].value = 2

        model.lsqfit = environ.Objective(rule=lsqObjPyomo, sense=1)

        model.robustConstr = environ.Constraint(model.trainingsizerangeforconstr, rule=robustConstrPyomo)

        opt = environ.SolverFactory(solver)
        # opt = environ.SolverFactory('ipopt')
        # opt.options['eps'] = 1e-10
        # opt.options['iprint'] = 1

        pyomodebug = self._fitpyomodebug
        if(pyomodebug == 0):
            ret = opt.solve(model)
        elif(pyomodebug == 1):
            import uuid
            uniquefn = str(uuid.uuid4())
            logfn = "/tmp/%s.log"%(uniquefn)
            print("Log file name: %s"%(logfn))
            ret = opt.solve(model,tee=True,logfile=logfn)
            model.pprint()
            ret.write()
        elif(pyomodebug==2):
            if(solver == 'filter'):
                opt.options['iprint'] = 1
                opt.options['rho'] = 10**10
                # opt.options['maxiter'] = 1
                # opt.options['z_print'] = -1
            logfn = "%s/%s_p%d_q%d_ts%s_i%d.log"%(self._debugfolder,self._fnname,self.m,self.n,self.trainingscale,iterationNo)
            self.printDebug("Starting %s"%(solver))
            ret = opt.solve(model,logfile=logfn)
            #ret = opt.solve(model, logfile=logfn, keepfiles=True)

        optstatus = {'message':str(ret.solver.termination_condition),'status':str(ret.solver.status),'time':ret.solver.time,'error_rc':ret.solver.error_rc}

        coeffs = np.array([model.coeff[i].value for i in range(self._M + self._N)])
        leastSq = model.lsqfit()

        return coeffs,leastSq,optstatus

    def fit(self, maxIterations=1000, maxRestarts=100, threshold=0.02):
        # Strategies:
        # 0: LSQ with SIP and without penalty
        # 1: LSQ with SIP and some coeffs set to 0 (using constraints)
        # 2: LSQ with SIP, penaltyParam > 0 and all or some coeffs in L1 term

        p_penaltyIndex, q_penaltyIndex = [], []
        if(self.strategy ==1 or self.strategy == 2):
            p_penaltyIndex, q_penaltyIndex = self.createPenaltyIndexArr()

        cons = np.empty(0, "object")
        coeffs0 = []

        if(self._fitstrategy == 'scipy'):
            ipoq = np.array([self._ipo[i][1] for i in range(self.trainingsize)])
            cons = np.append(cons, {'type': 'ineq', 'fun':fast_robustSampleV, 'args':(ipoq, self.M, self.N)})

            if(self.strategy == 0):
                # TODO need to check if this is a feasible point!
                coeffs0 = np.ones((self.M+self.N))
                coeffs0[self.M] = 2
            elif(self.strategy == 1):
                coeffs0 = np.zeros((self.M+self.N))
                for index in p_penaltyIndex:
                    cons = np.append(cons,{'type': 'eq', 'fun':self.coeffSetTo0, 'args':(index, "p")})
                for index in q_penaltyIndex:
                    cons = np.append(cons,{'type': 'eq', 'fun':self.coeffSetTo0, 'args':(index, "q")})
            elif(self.strategy == 2):
                coeffs0 = np.zeros(2*(self.M+self.N))
                for index in p_penaltyIndex:
                    cons = np.append(cons,{'type': 'ineq', 'fun':self.abs1, 'args':(index, "p")})
                    cons = np.append(cons,{'type': 'ineq', 'fun':self.abs2, 'args':(index, "p")})
                for index in q_penaltyIndex:
                    cons = np.append(cons,{'type': 'ineq', 'fun':self.abs1, 'args':(index, "q")})
                    cons = np.append(cons,{'type': 'ineq', 'fun':self.abs2, 'args':(index, "q")})
            else:
                raise Exception("strategy %i not implemented"%self.strategy)

        self._iterationinfo = []
        for iter in range(1, maxIterations+1):
            data = {}
            data['iterationNo'] = iter
            self.printDebug("Starting lsq for iter %d"%(iter))

            if(self._fitstrategy == 'scipy'):
                coeffs, leastSq, optstatus = self.scipyfit(coeffs0, cons)
                # This is a bit brutal trial and error,
                # if the starting point was not good, we just try again with a random
                # vector, otherwise coeffs is always the same and this loop does nothing
                # but waste time
                if optstatus['status']!=0:
                    fixme=True
                    while fixme:
                        coeffs0 = np.random.random(coeffs0.shape)
                        coeffs, leastSq, optstatus = self.scipyfit(coeffs0, cons)
                        if optstatus['status']!=0:
                            fixme=False
            elif(self._fitstrategy == 'filter' or self._fitstrategy == 'ipopt'):
                coeffs, leastSq, optstatus = self.pyomofit(iter-1,self._fitstrategy)
            else:raise Exception("fitstrategy %s not implemented"%self.fitstrategy)

            data['log'] = optstatus
            data['leastSqObj'] = leastSq
            data['pcoeff'] = coeffs[0:self.M].tolist()
            data['qcoeff'] = coeffs[self.M:self.M+self.N].tolist()

            if(self.strategy == 2 or  self.strategy == 0):
                lsqsplit = {}
                l1term = 0
                if(self.strategy == 2):
                    l1term = self.computel1Term(coeffs,p_penaltyIndex,q_penaltyIndex)
                elif(self.strategy == 0):
                    l1term = self.computel1Term(coeffs)
                if self.penaltyparam != 0:
                    lsqsplit['l1term'] = l1term
                else:
                    lsqsplit['l1term'] = 0
                lsqsplit['l2term'] = leastSq - self.penaltyparam * l1term
                data['leastSqSplit'] = lsqsplit

            # data['restartInfo'] = []
            robO = 0
            x = []
            if(self._roboptstrategy == 'ss'):
                self.printDebug("Starting ss")
                x, robO, restartInfo = self.multipleRestartForIterRobO(coeffs, 1, threshold)
                data['robOptInfo'] = {'robustArg':x.tolist(),'robustObj':robO,'info':restartInfo}
            elif(self._roboptstrategy == 'ms'):
                self.printDebug("Starting ms")
                x, robO, restartInfo = self.multipleRestartForIterRobO(coeffs,maxRestarts,threshold)
                data['robOptInfo'] = {'robustArg':x.tolist(),'robustObj':robO,'info':restartInfo}
            elif(self._roboptstrategy == 'mlsl'):
                x, robO, restartInfo = self.mlslRobO(coeffs,threshold)
                data['robOptInfo'] = {'robustArg':x.tolist(),'robustObj':robO,'info':restartInfo}
            elif(self._roboptstrategy == 'baron'):
                self.printDebug("Starting ba")
                x, robO, optInfo = self.pyomoRobO(coeffs,threshold,'baron')
                restartInfo = [{'robustArg':x.tolist(),'robustObj':robO,'log':optInfo}]
                data['robOptInfo'] = {'robustArg':x.tolist(),'robustObj':robO,'info':restartInfo}
            elif(self._roboptstrategy == 'solve'):
                self.printDebug("Starting so")
                x, robO, info = self.solveForEvalsRobO(coeff=coeffs,threshold=threshold)
                data['robOptInfo'] = {'robustArg':x.tolist(),'robustObj':robO,'info':info}
            elif(self._roboptstrategy == 'msbarontime'):
                self.printDebug("Starting msbarontime")
                a = 9.12758564
                b = 0.03065447
                nnl = calculateNonLin(self.dim,self.n)
                time = a*np.exp(b * nnl)
                x, robO, info = self.multipleRestartForTimeRobO(coeffs,time,threshold)
                d = info[len(info)-1]
                d['robustArg'] = x.tolist()
                d['robustObj'] = robO
                info = [d]
                data['robOptInfo'] = {'robustArg':x.tolist(),'robustObj':robO,'info':info}
            elif(self._roboptstrategy == 'ss_ms_so_ba'):
                # ss
                self.printDebug("Starting ss")
                ssx, ssrobO, ssrestartInfo = self.multipleRestartForIterRobO(coeffs, 1, threshold)
                sstime = ssrestartInfo[0]['log']['time'] #in sec

                # ba
                self.printDebug("Starting ba")
                bax, barobO, baOptInfo = self.pyomoRobO(coeffs,threshold,'baron')
                barestartInfo = [{'robustArg':bax.tolist(),'robustObj':barobO,'log':baOptInfo}]
                batime = baOptInfo['time']

                # ms
                self.printDebug("Starting ms")
                msx, msrobO, msrestartInfo = self.multipleRestartForTimeRobO(coeffs,batime,threshold)
                #overriding msrestartInfo to contain the size of the output JSON file
                d = msrestartInfo[len(msrestartInfo)-1]
                d['robustArg'] = msx.tolist()
                d['robustObj'] = msrobO
                msrestartInfo = [d]

                # so
                self.printDebug("Starting so")
                sox1, sorobO1, soinfo1 = self.solveForTimeRobO(coeff=coeffs,maxTime=batime,threshold=threshold)
                # sox2, sorobO2, soinfo2 = self.solveForTimeRobO(coeff=coeffs,maxTime=2*batime,threshold=threshold)
                # sox3, sorobO3, soinfo3 = self.solveForTimeRobO(coeff=coeffs,maxTime=3*batime,threshold=threshold)
                # sox4, sorobO4, soinfo4 = self.solveForTimeRobO(coeff=coeffs,maxTime=4*batime,threshold=threshold)

                soinfo1.append({'robustArg':sox1.tolist(),'robustObj':sorobO1})
                # soinfo2.append({'robustArg':sox2.tolist(),'robustObj':sorobO2})
                # soinfo3.append({'robustArg':sox3.tolist(),'robustObj':sorobO3})
                # soinfo4.append({'robustArg':sox4.tolist(),'robustObj':sorobO4})
                #
                robOarr = np.array([
                    ssrobO,
                    msrobO,
                    barobO,
                    sorobO1,
                    # sorobO2,
                    # sorobO3,
                    # sorobO4
                ])
                xdict = {
                    0:ssx,
                    1:msx,
                    2:bax,
                    3:sox1,
                    # 4:sox2,
                    # 5:sox3,
                    # 6:sox4
                }
                robO = np.min(robOarr)
                x = xdict[np.argmin(robOarr)]

                diffd = {}
                diffd['ss'] = ssrobO
                diffd["ms"] = msrobO
                diffd['ba'] = barobO
                diffd['so1x'] = sorobO1
                # diffd['so2x'] = sorobO2
                # diffd['so3x'] = sorobO3
                # diffd['so4x'] = sorobO4
                restartInfo = {
                    'ssInfo':ssrestartInfo,
                    'msInfo':msrestartInfo,
                    'baInfo':barestartInfo,
                    'so1xInfo':soinfo1,
                    # 'so2xInfo':soinfo2,
                    # 'so3xInfo':soinfo3,
                    # 'so4xInfo':soinfo4
                }
                data['robOptInfo'] = {'robustArg':x.tolist(),'robustObj':robO,'info':restartInfo,'diff':diffd}
            else: raise Exception("rob opt strategy unknown")

            self._iterationinfo.append(data)
            if self._dumpaftereachiter == 1:
                self._pcoeff = np.array(["see pcoeff from last iter"])
                self._qcoeff = np.array(["see qcoeff from last iter"])
                self._fittime = 0
                self.save("/tmp/"+self._uniqueid+".json")
                self.printDebug("Output written to: /tmp/"+self._uniqueid+".json")

            if(robO >= threshold):
                break

            q_ipo_new = self.recurrence(x,self._struct_q)
            if(self._fitstrategy == 'scipy'):
                cons = np.append(cons,{'type': 'ineq', 'fun':self.robustSample, 'args':(q_ipo_new,)})
            elif(self._fitstrategy == 'filter' or self._fitstrategy == 'ipopt'):
                self._ipo = np.vstack([self._ipo, np.empty((1,2),"object")])
                self._ipo[self.trainingsize+iter-1][1] = q_ipo_new

        if(len(self._iterationinfo) == maxIterations and self._iterationinfo[maxIterations-1]['robOptInfo']["robustObj"]<threshold):
            import json
            j = json.dumps(self._iterationinfo,indent=4, sort_keys=True)
            raise Exception(j+"\nCould not find a robust objective")
        self._pcoeff = np.array(self._iterationinfo[len(self._iterationinfo)-1]["pcoeff"])
        self._qcoeff = np.array(self._iterationinfo[len(self._iterationinfo)-1]["qcoeff"])

    def solveForTimeRobO(self, coeff, maxTime=5, threshold=0.2):
        info = []
        minx = []
        minq = np.inf
        totaltime = 0
        actualEvals = 0
        while(totaltime < maxTime):
            x, q, time = self.solveRobO(coeff,actualEvals)
            if(minq > q):
                minq = q
                minx = x
            if(q < 3*threshold):
                rinfo = {'robustArg':x.tolist(),'robustObj':q}
                info.append(rinfo)
            totaltime += time
            actualEvals += 1
            if(q < threshold):
                break
        info.append({'log':{'maxEvals':actualEvals,'actualEvals':actualEvals,'time':totaltime}})
        return minx, minq, info

    def solveForEvalsRobO(self, coeff, maxEvals=50000,threshold=0.2):
        info = []
        minx = []
        minq = np.inf
        totaltime = 0
        actualEvals = maxEvals
        for r in range(maxEvals):
            x, q, time = self.solveRobO(coeff,r)
            if(minq > q):
                minq = q
                minx = x
            if(q < 3*threshold):
                rinfo = {'robustArg':x.tolist(),'robustObj':q}
                info.append(rinfo)
            totaltime += time
            if(q < threshold):
                actualEvals = r+1
                break
        info.append({'log':{'maxEvals':maxEvals,'actualEvals':actualEvals,'time':totaltime}})
        return minx, minq, info

    def solveRobO(self, coeff,r):
        x=[]
        if(r == 0):
            x = np.array([(self.box[i][0]+self.box[i][1])/2 for i in range(self.dim)], dtype=np.float64)
        else:
            x = np.zeros(self.dim, dtype=np.float64)
            for d in range(self.dim):
                x[d] = np.random.rand()*(self.box[d][1]-self.box[d][0])+self.box[d][0]
        start = timer()
        q_ipo = self.recurrence(x,self._struct_q)
        q = np.sum([coeff[i]*q_ipo[i-self.M] for i in range(self.M,self.M+self.N)])
        end = timer()
        return x, q, end-start

    def pyomoRobO(self, coeffs, threshold=0.2,solver='filter',r=0):
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
        model.struct_q = self._struct_q.tolist()
        model.coeffs = coeffs.tolist()
        model.dimrange = range(self._dim)
        model.box = self.box.tolist()
        model.M = self._M
        model.x = environ.Var(model.dimrange, bounds=variableBound)
        if(r == 0):
            for d in range(self.dim):
                model.x[d].value = (self.box[d][0]+self.box[d][1])/2
        else:
            for d in range(self.dim):
                model.x[d].value = np.random.rand()*(self.box[d][1]-self.box[d][0])+self.box[d][0]
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
        pyomodebug = 0
        if(pyomodebug == 0):
            ret = opt.solve(model)
        elif(pyomodebug == 1):
            import uuid
            uniquefn = str(uuid.uuid4())
            logfn = "/tmp/%s.log"%(uniquefn)
            print("Log file name: %s"%(logfn))
            ret = opt.solve(model,tee=True,logfile=logfn)
            model.pprint()
            ret.write()

        optstatus = {'message':str(ret.solver.termination_condition),'status':str(ret.solver.status),'time':ret.solver.time,'error_rc':ret.solver.error_rc}

        robO = model.robO()
        x = np.array([model.x[i].value for i in range(self._dim)])
        # info = [{'robustArg':x.tolist(),'robustObj':robO,'log':optstatus}]

        return x, robO, optstatus


    """
    MLSL with LBFGS does not converge. Untested. Not fixed. DO NOT USE!!!
    """
    def mlslRobO(self,coeffs, threshold=0.2):
        import nlopt
        localopt = nlopt.opt(nlopt.LD_LBFGS, self._dim)
        localopt.set_lower_bounds(self.box[:,0])
        localopt.set_upper_bounds(self.box[:,1])
        localopt.set_min_objective(lambda x,grad: self.robustObjWithGrad(x,grad,coeffs))
        localopt.set_xtol_rel(1e-4)

        mlslopt = nlopt.opt(nlopt.G_MLSL_LDS, self._dim)
        mlslopt.set_lower_bounds(self.box[:,0])
        mlslopt.set_upper_bounds(self.box[:,1])
        mlslopt.set_min_objective(lambda x,grad: self.robustObjWithGrad(x,grad,coeffs))
        mlslopt.set_local_optimizer(localopt)
        mlslopt.set_stopval(1e-20)
        mlslopt.set_maxtime(500.0)

        x0 = np.array([(self.box[i][0]+self.box[i][1])/2 for i in range(self.dim)], dtype=np.float64)
        x = mlslopt.optimize(x0)
        robO = mlslopt.last_optimum_value()
        info = [{'robustArg':x.tolist(),'robustObj':robO}]

        return x, robO, info

    def multipleRestartForTimeRobO(self, coeffs, maxTime = 5,threshold=0.2):
        minx = []
        restartInfo = []
        minrobO = np.inf
        totaltime = 0
        r = 0
        localoptfunc = ""
        if(self._localoptsolver == 'scipy'):
            solver = 'L-BFGS-B'
            localoptfunc = self.restartRobO
        elif(self._localoptsolver == 'filter'):
            solver = 'filter'
            localoptfunc = self.pyomoRobO
        else:raise Exception("localoptsolver, %s unknown"%self._localoptsolver)

        while(totaltime < maxTime):
            x, robO, optstatus = localoptfunc(coeffs,threshold,solver,r)
            totaltime += optstatus['time']

            if(minrobO > robO):
                minrobO = robO
                minx = x
            rinfo = {'robustArg':x.tolist(),'robustObj':robO, 'log':optstatus}
            restartInfo.append(rinfo)
            r += 1
            if(robO < threshold):
                break
        restartInfo.append({'log':{'time':totaltime, 'noRestarts':r}})
        return minx, minrobO, restartInfo

    def multipleRestartForIterRobO(self, coeffs, maxRestarts=10, threshold=0.2):
        minx = []
        restartInfo = []
        minrobO = np.inf
        totaltime = 0
        norestarts = 0
        localoptfunc = ""
        if(self._localoptsolver == 'scipy'):
            solver = 'L-BFGS-B'
            localoptfunc = self.restartRobO
        elif(self._localoptsolver == 'filter'):
            solver = 'filter'
            localoptfunc = self.pyomoRobO
        else:raise Exception("localoptsolver, %s unknown"%self._localoptsolver)

        for r in range(maxRestarts):
            x, robO, optstatus = localoptfunc(coeffs,threshold,solver,r)
            totaltime += optstatus['time']

            if(minrobO > robO):
                minrobO = robO
                minx = x
            rinfo = {'robustArg':x.tolist(),'robustObj':robO, 'log':optstatus}
            restartInfo.append(rinfo)
            norestarts += 1
            if(robO < threshold):
                break
        restartInfo.append({'log':{'time':totaltime, 'noRestarts':norestarts}})
        return minx, minrobO, restartInfo

    def restartRobO(self, coeffs, threshold, solver, r):
        x0 = []
        if(r == 0):
            x0 = np.array([(self.box[i][0]+self.box[i][1])/2 for i in range(self.dim)], dtype=np.float64)
        else:
            x0 = np.zeros(self.dim, dtype=np.float64)
            for d in range(self.dim):
                x0[d] = np.random.rand()*(self.box[d][1]-self.box[d][0])+self.box[d][0]
        start = timer()
        ret = minimize(self.robustObj, x0, bounds=self.box, args = (coeffs,),method = solver, options={'maxiter': 1000,'ftol': 1e-4, 'disp': False})
        end = timer()
        optstatus = {'message':ret.get('message').decode(),'status':ret.get('status'),'noOfIterations':ret.get('nit'),'time':end-start}

        x = ret.get('x')
        robO = ret.get('fun')
        return x, robO, optstatus

    def leastSqObj(self,coeff):
        sum = 0
        for index in range(self.trainingsize):
            p_ipo = self._ipo[index][0]
            q_ipo = self._ipo[index][1]

            P = np.sum([coeff[i]*p_ipo[i] for i in range(self.M)])
            Q = np.sum([coeff[i]*q_ipo[i-self.M] for i in range(self.M,self.M+self.N)])

            sum += (self._Y[index] * Q - P)**2
        return sum

    def computel1Term(self,coeff,p_penaltyIndexs=np.array([]), q_penaltyIndexs=np.array([])):
        l1Term = 0
        for index in p_penaltyIndexs:
            term = coeff[self.M+self.N+index]
            if(abs(term) > 10**-5):
                l1Term += term
        for index in q_penaltyIndexs:
            term = coeff[self.M+self.N+self.M+index]
            if(abs(term) > 10**-5):
                l1Term += term
        return l1Term

    def computel1Term(self,coeff):
        coeffsump = 0
        for i in range(self.M):
            coeffsump += coeff[i]**2
        coeffsumq = 0
        for i in range(self.M,self.M+self.N):
            coeffsumq += coeff[i]**2
        return (coeffsump + coeffsumq)

    def leastSqObjWithPenalty(self,coeff,p_penaltyIndexs=np.array([]), q_penaltyIndexs=np.array([])):
        sum = self.leastSqObj(coeff)
        l1Term = self.penaltyparam * self.computel1Term(coeff, p_penaltyIndexs, q_penaltyIndexs)
        return sum+l1Term

    def abs1(self,coeff, index, pOrq="q"):
        ret = -1
        if(pOrq == "p"):
            ret = coeff[self.M+self.N+index] - coeff[index]
        elif(pOrq == "q"):
            ret = coeff[self.M+self.N+self.M+index] - coeff[self.M+index]
        return ret

    def abs2(self,coeff, index, pOrq="q"):
        ret = -1
        if(pOrq == "p"):
            ret = coeff[self.M+self.N+index] + coeff[index]
        elif(pOrq == "q"):
            ret = coeff[self.M+self.N+self.M+index] + coeff[self.M+index]
        return ret

    def coeffSetTo0(self, coeff, index, pOrq="q"):
        ret = -1
        if(pOrq == "p"):
            ret = coeff[index]
        elif(pOrq == "q"):
            ret = coeff[self.M+index]
        return ret

    def robustSample(self, coeff, q_ipo):
        return np.sum([coeff[i]*q_ipo[i-self.M] for i in range(self.M,self.M+self.N)]) - 1

    def robustObjWithGrad(self, x, grad, coeff):
        if grad.size > 0:
            g = tools.getPolyGradient(coeff=coeff[self.M:self.M+self.N],X=x, dim=self._dim,n=self._n)
            for i in range(grad.size): grad[i] = g[i]

        q_ipo = self.recurrence(x,self._struct_q)

        res = np.sum([coeff[i]*q_ipo[i-self.M] for i in range(self.M,self.M+self.N)])
        return res

    def robustObj(self, x, coeff):
        q_ipo = self.recurrence(x, self._struct_q)
        return np.dot(coeff[self.M:], q_ipo)

    def createPenaltyIndexArr(self):
        p_penaltyBinArr = self.ppenaltybin
        q_penaltyBinArr = self.qpenaltybin

        p_penaltyIndex = np.array([], dtype=np.int64)
        for index in range(self.m+1):
            if(p_penaltyBinArr[index] == 0):
                if(index == 0):
                    p_penaltyIndex = np.append(p_penaltyIndex, 0)
                else:
                    A = tools.numCoeffsPoly(self.dim, index-1)
                    B = tools.numCoeffsPoly(self.dim, index)
                    for i in range(A, B):
                        p_penaltyIndex = np.append(p_penaltyIndex, i)

        q_penaltyIndex = np.array([],dtype=np.int64)
        for index in range(self.n+1):
            if(q_penaltyBinArr[index] == 0):
                if(q_penaltyBinArr[index] == 0):
                    if(index == 0):
                        q_penaltyIndex = np.append(q_penaltyIndex, 0)
                    else:
                        A = tools.numCoeffsPoly(self.dim, index-1)
                        B = tools.numCoeffsPoly(self.dim, index)
                        for i in range(A, B):
                            q_penaltyIndex = np.append(q_penaltyIndex, i)

        return p_penaltyIndex, q_penaltyIndex

    def numer(self, X):
        """
        Evaluation of the denom poly at X.
        """
        rec_p = np.array(self.recurrence(X, self._struct_p))
        p = self._pcoeff.dot(rec_p)
        return p

    def denom(self, X):
        """
        Evaluation of the numer poly at X.
        """
        rec_q = np.array(self.recurrence(X, self._struct_q))
        q = self._qcoeff.dot(rec_q)
        return q

    def predict(self, X):
        """
        Return the prediction of the RationalApproximation at X.
        """
        X=self._scaler.scale(np.array(X))
        return self.numer(X)/self.denom(X)

    def predictOverArray(self, Xarr):
        """
        Return array of Rational Aptoximation predictions over an array X.
        """
        return [self.predict(X) for X in Xarr]

    def __call__(self, X):
        """
        Operator version of predict.
        """
        return self.predict(X)

    @property
    def asDict(self):
        """
        Store all info in dict as basic python objects suitable for JSON
        """
        d={}
        d['pcoeff']                 = self._pcoeff.tolist()
        d['qcoeff']                 = self._qcoeff.tolist()
        d['dim']              = self._dim
        d['m'] = self._m
        d['n'] = self._n
        d["scaler"] = self._scaler.asDict
        if self._debug:
            d['M'] = self._M
            d['N'] = self._N
            d['iterationinfo']    = self._iterationinfo
            d["log"] = {"fittime":self._fittime}
            d['strategy'] = self._strategy
            d['uniqueid'] = self._uniqueid
            d['roboptstrategy'] = self._roboptstrategy
            if self._roboptstrategy in ['ss','ms','msbarontime','ss_ms_so_ba']:
                d['localoptsolver'] = self._localoptsolver
            else: d['localoptsolver'] = "N/A"
            d['fitstrategy'] = self._fitstrategy
            d['trainingscale'] = self._trainingscale
            d['trainingsize'] = self._trainingsize

            if(self.strategy ==1 or self.strategy==2):
                d['chosenppenalty'] = self._ppenaltybin
                d['chosenqpenalty'] = self._qpenaltybin


            if(self.strategy==2 or self.strategy==0):
                d['lambda'] = self._penaltyparam
        return d

    @property
    def asJSON(self):
        """
        Store all info in dict as basic python objects suitable for JSON
        """
        d = self.asDict
        import json
        return json.dumps(d,indent=4, sort_keys=True)

    def save(self, fname, indent=4, sort_keys=True):
        import json
        with open(fname, "w") as f:
            json.dump(self.asDict, f,indent=indent, sort_keys=sort_keys)

    def printDebug(self, msg):
        import datetime
        if self._debug: print("[d%d p%d q%d ts%s] [[%s]] %s"%(self._dim,self._m,self._n,self._trainingscale, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), msg))

if __name__=="__main__":
    import sys
    infilePath11 = sys.argv[1]#"benchmarkdata/f16.txt"
    # infilePath1 = "../benchmarkdata/f1_noise_0.1.txt"
    X, Y = tools.readData(infilePath11)
    print(X.shape)
    for i in range(int(sys.argv[2])):
        r = RationalApproximationSIP(X, Y, m=2, n=3, trainingscale="Cp", roboptstrategy='ss', localoptsolver='scipy',
                                     fitstrategy='scipy', strategy=0)
    # print(r.asJSON)
    exit(0)
    r = RationalApproximationSIP(X,Y,
                                m=2,
                                n=3,
                                trainingscale="1x",
                                roboptstrategy = 'ms',
                                localoptsolver = 'scipy',
                                fitstrategy = 'scipy',
                                strategy=0,
                                penaltyparam=10**-1,
                                ppenaltybin=[1,0,0],
                                qpenaltybin=[1,0,0,0]
    )
    rf = RationalApproximationSIP(X,Y,
                                m=2,
                                n=3,
                                trainingscale="1x",
                                roboptstrategy = 'ms',
                                localoptsolver = 'scipy',
                                fitstrategy = 'filter',
                                strategy=0,
                                penaltyparam=10**-1,
                                ppenaltybin=[1,0,0],
                                qpenaltybin=[1,0,0,0]
    )
    # print(r.predictOverArray(X))
    # r.save("/Users/mkrishnamoorthy/Desktop/pythonRASIP.json")

    # r2 = RationalApproximationSIP(r.asDict)
    # print(r.asJSON)
    # print(r2.pcoeff, r2.qcoeff,r2.box,r2.ppenaltybin,r2.qpenaltybin, r2.dim)
    # print(r2(X[0:4,:])) #calls predict

    # r1 = RationalApproximationSIP("/Users/mkrishnamoorthy/Desktop/pythonRASIP.json")
    print(r.asJSON)
    #
    # import pylab
    # pylab.plot(X, Y, marker="*", linestyle="none", label="Data")
    # TX = sorted(X)
    # YW = [r(p) for p in TX]
    # YF = [rf(p) for p in TX]

    # pylab.plot(TX, YW, label="Rational approximation")
    # pylab.plot(TX, YF, label="Rational approximation filter")

    # # Store the last and restore immediately, plot to see if all is good
    # r.save("siptest.json")
    # r=RationalApproximationSIP("siptest.json")
    # YW = [r(p) for p in TX]
    # pylab.plot(TX, YW, "m--", label="Restored approx.")
    # pylab.legend()
    # pylab.xlabel("x")
    # pylab.ylabel("f(x)")
    # pylab.savefig("demoSIP.pdf")

    # sys.exit(0)








# END
