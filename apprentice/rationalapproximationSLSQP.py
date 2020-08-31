# import numpy as np
import autograd.numpy as np
from apprentice import tools
from scipy.optimize import minimize
from timeit import default_timer as timer
import apprentice

# TODO code and use gradients of  constraintOrder1V and constraintAVW

def constraintOrder1V(coeff, M, N, L, U):
    """ Inequality constraints for the order 1 denominator case """
    b = coeff[M]
    v = coeff[M+N:M+N+N-1]
    w = coeff[M+N+N-1:]

    c = np.zeros(2*N-1)
    c[0] = b + np.dot(v, L) - np.dot(w,U) - 1e-6
    c[1:] = coeff[M+N:]
    return c

def constraintOrder1G(coeff, M, N, L, U):
    """ Gradient of inequality constraints for the order 1 denominator case """
    G = np.zeros((2*N-1,M+3*N-2))
    G[0][M] = 1
    for i in range(N-1):
        G[0][M+N+i] = L[i]
        G[0][M+N+N-1+i] = -U[i]
        G[1+i][M+N+i] = -L[i]
        G[N+i][M+N+N-1+i] = U[i]
    return G

def constraintAVW(coeff, M, N):
    """ Equality constraints for the order 1 denominator case """
    a = coeff[M+1:M+N]
    v = coeff[M+N:M+N+N-1]
    w = coeff[M+N+N-1:]
    return a - v + w

def fast_robustSampleV(coeff, q_ipo, M, N):
    return np.sum(coeff[M:M+N] * q_ipo, axis=1) - 1e-6

def fast_robustSampleG(coeff, q_ipo, M, N):
    G=np.zeros((q_ipo.shape[0], M+N))
    G[:, M:M+N] = q_ipo
    return G

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

class RationalApproximationSLSQP(apprentice.RationalApproximation):
    def __init__(self, *args, **kwargs):
        """
        Multivariate rational approximation p(x)_m/q(x)_n

        args:
            X       --- anchor points
            Y       --- function values

        kwargs:
            order (tuple m,n)
                            --- order of the numerator polynomial --- if omitted: auto 1 used
                            --- order of the denominator polynomial --- if omitted: auto 1 used
            pnames          --- list of parameter names to pass to the scaler to scale
        """
        self._vmin=None
        self._vmax=None
        self._debug = kwargs["debug"] if kwargs.get("debug") is not None else  False
        self._ftol      = float(kwargs["ftol"])  if kwargs.get("ftol")    is not None else 1e-6
        self._slsqp_iter= int(kwargs["itslsqp"]) if kwargs.get("itslsqp") is not None else 200

        self._m=kwargs["order"][0]
        self._n=kwargs["order"][1]

        import os
        if len(args) == 0:
            pass
        else:
            # Scaler related kwargs
            _pnames      = kwargs["pnames"]   if kwargs.get("pnames")   is not None else None
            _scale_min   = kwargs["scalemin"] if kwargs.get("scalemin") is not None else -1
            _scale_max   = kwargs["scalemax"] if kwargs.get("scalemax") is not None else  1
            self._scaler = apprentice.Scaler(np.array(args[0], dtype=np.float64), a=_scale_min, b=_scale_max, pnames=_pnames)
            self._X      = self._scaler.scaledPoints
            self._trainingsize = len(self._X)
            self._dim = self._X[0].shape[0]
            self._Y      = np.array(args[1], dtype=np.float64)
            if self._dim==1: self.recurrence=apprentice.monomial.recurrence1D
            else           : self.recurrence=apprentice.monomial.recurrence
            self.setStructures()
            self.setIPO()

            # TODO for Holger: Add the following two as args to this class
            useampl = bool(kwargs["useampl"])
            solver = kwargs["amplsolver"]
            if self._n == 1:
                if useampl:
                    self.fitOrder1AMPL(solver)
                else:
                    self.fitOrder1()
            else:
                self.fit()

    @property
    def trainingsize(self): return self._trainingsize

    @property
    def box(self): return self._scaler.box_scaled

    def setIPO(self):
        """
        Calculate the Pade approximation
        """
        self._ipo            = np.empty((self.trainingsize,2), "object")
        for i in range(self.trainingsize):
            self._ipo[i][0] = self.recurrence(self._X[i,:],self._struct_p)
            self._ipo[i][1] = self.recurrence(self._X[i,:],self._struct_q)

    def scipyfit(self, coeffs0, cons, iprint=3):
        start = timer()
        ipop = np.array([self._ipo[i][0] for i in range(self.trainingsize)])
        ipoq = np.array([self._ipo[i][1] for i in range(self.trainingsize)])

        import autograd
        from IPython import embed
        embed()
        import sys
        sys.exit(1)


        ret = minimize(fast_leastSqObj, coeffs0 , args=(self.trainingsize, ipop, ipoq, self.M, self.N, self._Y),
                jac=fast_jac, method = 'SLSQP', constraints=cons,
                options={'maxiter': self._slsqp_iter, 'ftol': self._ftol, 'disp': self._debug, 'iprint': iprint})
        end = timer()
        optstatus = {'message':ret.get('message'),'status':ret.get('status'),'noOfIterations':ret.get('nit'),'time':end-start}
        return ret.get('x'), ret.get('fun'), optstatus


    def fitOrder1AMPL(self,solver):
        import os
        # print("REACHED HERE")

        def lsqObj(model):
            sum = 0
            for index in range(model.trainingsize):
                p_ipo = model.pipo[index]
                q_ipo = model.qipo[index]
                P = 0
                # coeffsump = 0
                for i in model.prange:
                    P += model.pcoeff[i] * p_ipo[i]
                    # coeffsump += model.pcoeff[i] ** 2

                Q = 0
                # coeffsumq = 0
                for i in model.qrange:
                    Q += model.qcoeff[i] * q_ipo[i]
                    # coeffsumq += model.qcoeff[i] ** 2

                # sum += (model.Y[index] * Q - P)**2 # sigma = 0
                sum += (model.Y[index] * Q - P) ** 2
            return sum

        def constraintOrder1V(model):
            b = model.qcoeff[0]
            v = model.vcoeff[:]
            w = model.wcoeff[:]

            ret = b
            vL = 0.
            for x, y in zip(v, model.L):
                vL += x*y
            wU = 0.
            for x, y in zip(w, model.U):
                wU += x*y

            ret += vL - wU
            return ret >= 1e-6

        def constraintAVW(model,index):
            a = model.qcoeff[1 + index]
            v = model.vcoeff[index]
            w = model.wcoeff[index]

            return a == v-w


        from pyomo import environ
        model = environ.ConcreteModel()
        model.dimrange = range(self._dim)

        model.prange = range(self.M)
        model.qrange = range(self.N)
        model.vrange = range(self.N-1)
        model.wrange = range(self.N-1)

        model.M = self._M
        model.N = self._N
        model.trainingsize = self.trainingsize

        model.pipo = self._ipo[:, 0].tolist()
        model.qipo = self._ipo[:, 1].tolist()

        model.Y = self._Y.tolist()

        model.L = self._scaler._a
        model.U = self._scaler._b

        model.pcoeff = environ.Var(model.prange,initialize=1.)
        model.qcoeff = environ.Var(model.qrange,initialize=1.)
        model.vcoeff = environ.Var(model.vrange,bounds = (0,None),initialize=0.)
        model.wcoeff = environ.Var(model.wrange,bounds = (0,None),initialize=0.)

        model.obj = environ.Objective(rule=lsqObj, sense=1)
        model.constraintOrder1V = environ.Constraint(rule=constraintOrder1V)
        model.constraintAVW = environ.Constraint(model.vrange, rule=constraintAVW)

        opt = environ.SolverFactory(solver)
        plevel = 5
        if not self._debug:
            plevel = 1

        from pyutilib.services import TempfileManager
        # os.makedirs('../../log/tmp', exist_ok=True)
        # TempfileManager.tempdir = '../../log/tmp'
        # self.logfp = "/tmp/log.log"
        if self._debug:
            model.pprint()
        ret = opt.solve(model,
                        tee=False,
                        # tee=True,
                        # logfile=self.logfp,
                        keepfiles=False,
                        # keepfiles=True,
                        # options={'file_print_level': 5, 'print_level': plevel}
                        )

        if self._debug:
            ret.write()

        if self._debug:
            print(np.array([model.pcoeff[i].value for i in model.prange]))
            print(np.array([model.qcoeff[i].value for i in model.qrange]))
        self._pcoeff = np.array([model.pcoeff[i].value for i in model.prange])
        self._qcoeff = np.array([model.qcoeff[i].value for i in model.qrange])



    def fitOrder1(self):
        """ The dual problem for order 1 denominator polynomials """
        coeffs0 = np.random.random(self.M+3*self.N-2)

        GG = constraintOrder1G(coeffs0, self.M, self.N, self._scaler._a, self._scaler._b)

        # def localGG(*args):
            # return GG

        cons = np.empty(0, "object")
        cons = np.append(cons, {'type': 'ineq', 'fun':constraintOrder1V, 'args':(self.M, self.N, self._scaler._a, self._scaler._b)})
        # cons = np.append(cons, {'type': 'ineq', 'fun':constraintOrder1V, 'jac':localGG, 'args':(self.M, self.N, self._scaler._a, self._scaler._b)})
        # cons = np.append(cons, {'type': 'ineq', 'fun':constraintOrder1V, 'jac':constraintOrder1G, 'args':(self.M, self.N, self._scaler._a, self._scaler._b)})
        cons = np.append(cons, {'type': 'eq', 'fun':constraintAVW, 'args':(self.M, self.N)})

        coeffs, leastSq, optstatus = self.scipyfit(coeffs0, cons)

        self._pcoeff = coeffs[:self.M]
        self._qcoeff = coeffs[self.M:self.M+self.N]

    def fit(self, maxIterations=1000, maxRestarts=100, threshold=0.2):


        ipoq = np.array([self._ipo[i][1] for i in range(self.trainingsize)])
        cons = np.empty(0, "object")
        cons = np.append(cons, {'type': 'ineq', 'fun':fast_robustSampleV, 'jac':fast_robustSampleG,  'args':(ipoq, self.M, self.N)})


        # TODO need to check if this is a feasible point!
        coeffs0 = np.ones((self.M+self.N))
        coeffs0[self.M] = 2

        self._iterationinfo = []
        for iter in range(1, maxIterations+1):
            data = {}
            coeffs, leastSq, optstatus = self.scipyfit(coeffs0, cons)
            # This is a bit brutal trial and error,
            # if the starting point was not good, we just try again with a random
            # vector, otherwise coeffs is always the same and this loop does nothing
            # but waste time
            if optstatus['status'] not in [0,9]:
                fixme=True
                while fixme:
                    coeffs0 = np.random.random(coeffs0.shape)
                    coeffs, leastSq, optstatus = self.scipyfit(coeffs0, cons)
                    if optstatus['status']!=0:
                        fixme=False

            data['pcoeff'] = coeffs[0:self.M].tolist()
            data['qcoeff'] = coeffs[self.M:self.M+self.N].tolist()

            robO = 0
            x, robO, restartInfo, newC, newO = self.multipleRestartForIterRobO(coeffs,maxRestarts,threshold)
            data['robOptInfo'] = {'robustArg':x.tolist(),'robustObj':robO,'info':restartInfo}

            self._iterationinfo.append(data)

            if(robO >= threshold): break

            q_ipo_new = self.recurrence(x,self._struct_q)
            ipoq = np.vstack((ipoq,q_ipo_new))
            cons = np.empty(0, "object")
            cons = np.append(cons, {'type': 'ineq', 'fun':fast_robustSampleV, 'jac':fast_robustSampleG,  'args':(ipoq, self.M, self.N)})

        if (len(self._iterationinfo) == maxIterations and self._iterationinfo[maxIterations-1]['robOptInfo']["robustObj"]<threshold):
            import json
            j = json.dumps(self._iterationinfo,indent=4, sort_keys=True)
            raise Exception(j+"\nCould not find a robust objective")
        self._pcoeff = np.array(self._iterationinfo[len(self._iterationinfo)-1]["pcoeff"])
        self._qcoeff = np.array(self._iterationinfo[len(self._iterationinfo)-1]["qcoeff"])

    def multipleRestartForIterRobO(self, coeffs, maxRestarts=10, threshold=0.2, solver="L-BFGS-B"):
        minx, restartInfo = [], []
        totaltime, norestarts = 0, 0

        X0 = self._scaler.drawSamples_scaled(maxRestarts)

        X, Y = [], []
        minrobO = np.inf
        for x0 in X0:
            x, robO, optstatus = self.restartRobO(x0, coeffs, threshold, solver)
            X.append(x)
            Y.append(robO)
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
        return minx, minrobO, restartInfo, X, Y

    def restartRobO(self, x0, coeffs, threshold, solver):
        ret = minimize(self.robustObj, x0, bounds=self.box, args = (coeffs,), method = solver, options={'maxiter': 1000,'ftol': 1e-4, 'disp': False})
        optstatus = {'message':ret.get('message').decode(), 'status':ret.get('status'), 'noOfIterations':ret.get('nit'), 'time':0}
        return ret.x, ret.fun, optstatus


    def robustObj(self, x, coeff):
        q_ipo = self.recurrence(x, self._struct_q)
        return np.dot(coeff[self.M:], q_ipo)
