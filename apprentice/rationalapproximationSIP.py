import numpy as np
from apprentice import monomial
from apprentice import tools
from scipy.optimize import minimize


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
            strategy        --- strategy to use --- if omitted: auto 0 used
                                0: min ||f*p(x)_m/q(x)_n||^2_2 sub. to q(x)_n >=1
                                1: min ||f*p(x)_m/q(x)_n||^2_2 sub. to q(x)_n >=1 and some p and/or q coeffecients set to 0
                                2: min ||f*p(x)_m/q(x)_n||^2_2 + lambda*||c_pq|| sub. to q(x)_n >=1
            roboptstrategy  --- strategy to optimize robust objective --- if omitted: auto 'ms' used
                                ms: multistart algorithm (with 10 restarts at random points from the box) using scipy.L-BFGS-B local optimizer
                                mlsl: multi-level single-linkage multistart algorithm from nlopt using nlopt.LD_LBFGS local optimizer
                                baron: pyomo with baron
            penaltyparam    --- lambda to use for strategy 2 --- if omitted: auto 0.1 used
            penaltybin      --- penalty binary array for numberator and denomintor of the bits to keep active in strategy 1 and put in penalty term for activity 2
                                represented in a 2D array of shape(2,(m/n)+1) where for each numberator and denominator, the bits represent penalized coeffecient degrees and constant (1: not peanlized, 0 penalized)
                                required for strategy 1 and 2

        """
        import os
        if len(args) == 0:
            pass
        else:
            if type(args[0])==dict:
                self.mkFromDict(args[0])
            elif type(args[0]) == str:
                self.mkFromJSON(args[0])
            else:
                self._X   = np.array(args[0], dtype=np.float64)
                self._Y   = np.array(args[1], dtype=np.float64)
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
    def box(self): return self._box
    @property
    def strategy(self): return self._strategy
    @property
    def roboptstrategy(self): return self._roboptstrategy
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

    def mkFromJSON(self, fname):
        import json
        d = json.load(open(fname))
        self.mkFromDict(d)

    def mkFromDict(self, pdict):
        self._pcoeff        = np.array(pdict["pcoeff"]).tolist()
        self._qcoeff        = np.array(pdict["qcoeff"]).tolist()
        self._iterationinfo = pdict["iterationinfo"]
        self._dim           = pdict["dim"]
        self._m             = pdict["m"]
        self._n             = pdict["n"]
        self._M             = pdict["M"]
        self._N             = pdict["N"]
        self._strategy      = pdict["strategy"]
        self._roboptstrategy= pdict["roboptstrategy"]
        self._box           = np.array(pdict["box"],dtype=np.float64)
        self._trainingscale = pdict["trainingscale"]
        self._trainingsize  = pdict["trainingsize"]
        self._penaltyparam  = 0.0

        if(self.strategy ==1 or self.strategy==2):
            self._ppenaltybin = pdict['chosenppenalty']
            self._qpenaltybin = pdict['chosenqpenalty']

        if(self.strategy == 2):
            self._penaltyparam = pdict['lambda']

        self._struct_p      = monomial.monomialStructure(self.dim, self.m)
        self._struct_q      = monomial.monomialStructure(self.dim, self.n)
        # self.setStructures(pdict["m"], pdict["n"])

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
        self._roboptstrategy    = kwargs["roboptstrategy"] if kwargs.get("roboptstrategy") is not None else "ms"
        self._box               = np.empty(shape=(0,2))

        if(kwargs.get("box") is not None):
            for arr in kwargs.get("box"):
                newArr =np.array([[arr[0],arr[1]]],dtype=np.float64)
                self._box = np.concatenate((self._box,newArr),axis=0)
        else:
            for i in range(self.dim):
                newArr = np.array([[-1,1]],dtype=np.float64)
                self._box = np.concatenate((self._box,newArr),axis=0)

        self._trainingscale = kwargs["trainingscale"] if kwargs.get("trainingscale") is not None else "1x"
        if(self.trainingscale == ".5x"):
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


        self._struct_p      = monomial.monomialStructure(self.dim, self.m)
        self._struct_q      = monomial.monomialStructure(self.dim, self.n)

        self._ipo            = np.empty((self.trainingsize,2),"object")
        for i in range(self.trainingsize):
            self._ipo[i][0] = monomial.recurrence(self._X[i,:],self._struct_p)
            self._ipo[i][1]= monomial.recurrence(self._X[i,:],self._struct_q)
        self.fit()

    def fit(self):
        # Strategies:
        # 0: LSQ with SIP and without penalty
        # 1: LSQ with SIP and some coeffs set to 0 (using constraints)
        # 2: LSQ with SIP, penaltyParam > 0 and all or some coeffs in L1 term


        cons = np.empty(self.trainingsize, "object")
        for trainingIndex in range(self.trainingsize):
            q_ipo = self._ipo[trainingIndex][1]
            cons[trainingIndex] = {'type': 'ineq', 'fun':self.robustSample, 'args':(q_ipo,)}

        p_penaltyIndex = []
        q_penaltyIndex = []
        if(self.strategy ==1 or self.strategy == 2):
            p_penaltyIndex, q_penaltyIndex = self.createPenaltyIndexArr()
        coeff0 = []
        if(self.strategy == 0):
            coeffs0 = np.zeros((self.M+self.N))
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
            raise Exception("fit() strategy %i not implemented"%self.strategy)

        maxIterations = 100 # hardcode for now. Param later?
        maxRestarts = 10    # hardcode for now. Param later?
        threshold = 0.02
        self._iterationinfo = []
        for iter in range(1,maxIterations+1):
            data = {}
            data['iterationNo'] = iter
            ret = {}
            if(self.strategy == 2):
                ret = minimize(self.leastSqObjWithPenalty, coeffs0, args = (p_penaltyIndex,q_penaltyIndex),method = 'SLSQP', constraints=cons, options={'iprint': 0,'ftol': 1e-6, 'disp': False})
            else:
                ret = minimize(self.leastSqObj, coeffs0 ,method = 'SLSQP', constraints=cons, options={'iprint': 0,'ftol': 1e-6, 'disp': False})
            coeffs = ret.get('x')
            # print(ret)
            # print(np.c_[coeffs[self.M+self.N:self.M+self.N+self.M],coeffs[0:self.M], coeffs[self.M+self.N:self.M+self.N+self.M]-coeffs[0:self.M] ])
            # print(np.c_[coeffs[self.M+self.N+self.M:self.M+self.N+self.M+self.N],coeffs[self.M:self.M+self.N]])
            leastSq = ret.get('fun')
            data['leastSqObj'] = leastSq
            data['pcoeff'] = coeffs[0:self.M].tolist()
            data['qcoeff'] = coeffs[self.M:self.M+self.N].tolist()

            if(self.strategy == 2):
                lsqsplit = {}
                l1term = self.computel1Term(coeffs,p_penaltyIndex,q_penaltyIndex)
                lsqsplit['l1term'] = l1term
                lsqsplit['l2term'] = leastSq - self.penaltyparam * l1term
                data['leastSqSplit'] = lsqsplit

            # data['restartInfo'] = []
            robO = 0
            x = []
            if(self._roboptstrategy == 'ms'):
                maxRestarts = 10
                x, robO, restartInfo = self.multipleRestartRobO(coeffs,maxRestarts,threshold)
                data['robOptInfo'] = restartInfo
            elif(self._roboptstrategy == 'mlsl'):
                x, robO, restartInfo = self.mlslRobO(coeffs,threshold)
                data['robOptInfo'] = restartInfo
            elif(self._roboptstrategy == 'baron'):
                x, robO, restartInfo = self.baronPyomoRobO(coeffs,threshold)
                data['robOptInfo'] = restartInfo

            self._iterationinfo.append(data)
            if(robO >= threshold):
                break
            q_ipo_new = monomial.recurrence(x,self._struct_q)
            cons = np.append(cons,{'type': 'ineq', 'fun':self.robustSample, 'args':(q_ipo_new,)})

        if(len(self._iterationinfo) == maxIterations and self._iterationinfo[maxIterations-1]["robustObj"]<threshold):
            raise Exception("Could not find a robust objective")
        self._pcoeff = self._iterationinfo[len(self._iterationinfo)-1]["pcoeff"]
        self._qcoeff = self._iterationinfo[len(self._iterationinfo)-1]["qcoeff"]

    def variableBound(self, model, i):
        b = (self._box[i][0], self._box[i][1])
        return b

    def baronPyomoRobO(self, coeffs, threshold=0.2):
        from pyomo import environ
        info = np.zeros(shape=(len(self._struct_q),self._dim+1),dtype=np.float64)
        for l in range(len(self._struct_q)):
            for d in range(self._dim):
                info[l][d] = self._struct_q[l][d]
            info[l][self._dim] = coeffs[l+self._M]
        model = environ.ConcreteModel()
        model.dimrange = range(self._dim)
        model.coeffinfo = info
        model.x = environ.Var(model.dimrange, bounds=self.variableBound)
        model.robO = environ.Objective(rule=self.robObjPyomo, sense=1)
        opt = environ.SolverFactory('baron')
        ret = opt.solve(model)
        robO = model.robO()
        x = np.array([model.x[i].value for i in range(self._dim)])
        info = [{'robustArg':x.tolist(),'robustObj':robO}]

        return x, robO, info


    def mlslRobO(self,coeffs, threshold=0.2):
        import nlopt
        localopt = nlopt.opt(nlopt.LD_LBFGS, self._dim)
        localopt.set_lower_bounds(self._box[:,0])
        localopt.set_upper_bounds(self._box[:,1])
        localopt.set_min_objective(lambda x,grad: self.robustObjWithGrad(x,grad,coeffs))
        localopt.set_xtol_rel(1e-4)

        mlslopt = nlopt.opt(nlopt.G_MLSL_LDS, self._dim)
        mlslopt.set_lower_bounds(self._box[:,0])
        mlslopt.set_upper_bounds(self._box[:,1])
        mlslopt.set_min_objective(lambda x,grad: self.robustObjWithGrad(x,grad,coeffs))
        mlslopt.set_local_optimizer(localopt)
        mlslopt.set_stopval(1e-20)
        mlslopt.set_maxtime(500.0)

        x0 = np.array([(self.box[i][0]+self.box[i][1])/2 for i in range(self.dim)], dtype=np.float64)
        x = mlslopt.optimize(x0)
        robO = mlslopt.last_optimum_value()
        info = [{'robustArg':x.tolist(),'robustObj':robO}]

        # print(info)
        # exit(1)

        return x, robO, info

    def multipleRestartRobO(self, coeffs, maxRestarts = 10, threshold=0.2):
        robO = 0
        x = []
        restartInfo = []
        for r in range(maxRestarts):
            x0 = []
            if(r == 0):
                x0 = np.array([(self.box[i][0]+self.box[i][1])/2 for i in range(self.dim)], dtype=np.float64)
            else:
                x0 = np.zeros(self.dim, dtype=np.float64)
                for d in range(self.dim):
                    x0[d] = np.random.rand()*(self.box[d][1]-self.box[d][0])+self.box[d][0]
            ret = minimize(self.robustObj, x0, bounds=self.box, args = (coeffs,),method = 'L-BFGS-B')
            x = ret.get('x')
            robO = ret.get('fun')
            rinfo = {'robustArg':x.tolist(),'robustObj':robO}
            restartInfo.append(rinfo)
            if(robO < threshold):
                break
        return x, robO, restartInfo

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

    def robustSample(self,coeff, q_ipo):
        return np.sum([coeff[i]*q_ipo[i-self.M] for i in range(self.M,self.M+self.N)])-1

    def robObjPyomo(self, model):
        dim = len(model.dimrange)
        res = 0
        for mon in model.coeffinfo:
            term = 1
            for d in model.dimrange:
                term *= model.x[d] ** mon[d]
            res += mon[dim] * term
        return res

    def robustObjWithGrad(self, x, grad, coeff):
        if grad.size > 0:
            g = tools.getPolyGradient(coeff=coeff[self.M:self.M+self.N],X=x, dim=self._dim,n=self._n)
            for i in range(grad.size): grad[i] = g[i]

        q_ipo = monomial.recurrence(x,self._struct_q)

        res = np.sum([coeff[i]*q_ipo[i-self.M] for i in range(self.M,self.M+self.N)])
        return res

    def robustObj(self,x,coeff):
        q_ipo = monomial.recurrence(x,self._struct_q)
        return np.sum([coeff[i]*q_ipo[i-self.M] for i in range(self.M,self.M+self.N)])

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
        ipo = np.empty(len(X),"object")
        for i in range(len(X)):
            ipo[i] = monomial.recurrence(X[i,0:self.dim],self._struct_p)
            ipo[i] = ipo[i].dot(self._pcoeff)
        return ipo

    def denom(self, X):
        """
        Evaluation of the numer poly at X.
        """
        ipo = np.empty(len(X),"object")
        for i in range(len(X)):
            ipo[i] = monomial.recurrence(X[i,0:self.dim],self._struct_q)
            ipo[i] = ipo[i].dot(self._qcoeff)
        return ipo

    def predict(self, X):
        """
        Return the prediction of the RationalApproximation at X.
        """
        return self.numer(X)/self.denom(X)

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
        d['pcoeff']                 = self._pcoeff
        d['qcoeff']                 = self._qcoeff
        d['iterationinfo']    = self._iterationinfo
        d['dim']              = self._dim
        d['m'] = self._m
        d['n'] = self._n
        d['M'] = self._M
        d['N'] = self._N
        d['strategy'] = self._strategy
        d['roboptstrategy'] = self._roboptstrategy
        d['box'] = self._box.tolist()
        d['trainingscale'] = self._trainingscale
        d['trainingsize'] = self._trainingsize

        if(self.strategy ==1 or self.strategy==2):
            d['chosenppenalty'] = self._ppenaltybin
            d['chosenqpenalty'] = self._qpenaltybin


        if(self.strategy==2):
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

if __name__=="__main__":
    import sys
    infilePath11 = "../benchmarkdata/f11_noise_0.1.txt"
    infilePath1 = "../benchmarkdata/f1_noise_0.1.txt"
    X, Y = tools.readData(infilePath11)
    r = RationalApproximationSIP(X,Y,
                                m=2,
                                n=3,
                                trainingscale="1x",
                                box=np.array([[-1,1]]),
                                # box=np.array([[-1,1],[-1,1]]),
                                strategy=2,
                                penaltyparam=10**-1,
                                ppenaltybin=[1,0,0],
                                qpenaltybin=[1,0,0,0]
    )
    # r.save("/Users/mkrishnamoorthy/Desktop/pythonRASIP.json")

    r2 = RationalApproximationSIP(r.asDict)
    print(r2.asJSON)
    print(r2.pcoeff, r2.qcoeff,r2.box,r2.ppenaltybin,r2.qpenaltybin, r2.dim)
    print(r2(X[0:4,:])) #calls predict

    # r1 = RationalApproximationSIP("/Users/mkrishnamoorthy/Desktop/pythonRASIP.json")
    # print(r1.asJSON)








# END
