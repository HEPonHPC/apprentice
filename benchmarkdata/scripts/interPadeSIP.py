import nlopt
import numpy as np
import sympy as sym
from apprentice import monomial
from apprentice import tools
from scipy.optimize import minimize
# from scipy.optimize import fmin_l_bfgs_b

def leastSqObj(coeff, ipo, Y, M=1, N=1, trainingsize=3):
    sum = 0
    for index in range(trainingsize):
        p_ipo = ipo[index][0]
        q_ipo = ipo[index][1]

        P = np.sum([coeff[i]*p_ipo[i] for i in range(M)])
        Q = np.sum([coeff[i]*q_ipo[i-M] for i in range(M,M+N)])

        sum += (Y[index] * Q - P)**2
    return sum

def computel1Term(coeff, M=1, N=1,l=0.001,p_penaltyIndexs=np.array([]), q_penaltyIndexs=np.array([])):
    l1Term = 0
    for index in p_penaltyIndexs:
        l1Term += l * coeff[M+N+index]
    for index in q_penaltyIndexs:
        l1Term += l * coeff[M+N+M+index]
    return l1Term

def leastSqObjWithPenalty(coeff, ipo, Y, M=1, N=1, trainingsize=3,l=0.001,
            p_penaltyIndexs=np.array([]), q_penaltyIndexs=np.array([])):
    sum = leastSqObj(coeff, ipo, Y, M, N, trainingsize)
    l1Term = computel1Term(coeff, M, N, l, p_penaltyIndexs, q_penaltyIndexs)
    return sum+l1Term

def abs1(coeff, index, pOrq="q", M=1, N=1):
    ret = -1
    if(pOrq == "p"):
        ret = coeff[M+N+index] - coeff[index]
    elif(pOrq == "q"):
        ret = coeff[M+N+M+index] - coeff[M+index]
    return ret

def abs2(coeff, index, pOrq="q", M=1, N=1):
    ret = -1
    if(pOrq == "p"):
        ret = coeff[M+N+index] + coeff[index]
    elif(pOrq == "q"):
        ret = coeff[M+N+M+index] + coeff[M+index]
    return ret

def coeffSetTo0(coeff, index, pOrq="q", M=1):
    ret = -1
    if(pOrq == "p"):
        ret = coeff[index]
    elif(pOrq == "q"):
        ret = coeff[M+index]
    return ret

def robustSample(coeff, q_ipo, M=1, N=1):
    return np.sum([coeff[i]*q_ipo[i-M] for i in range(M,M+N)])-1

def robustObj(x,coeff,struct_q,M,N):
    q_ipo = monomial.recurrence(x,struct_q)
    return np.sum([coeff[i]*q_ipo[i-M] for i in range(M,M+N)])

def createPenaltyIndexArr(p_penaltyBinArr, q_penaltyBinArr, dim, m, n):
    p_penaltyIndex = np.array([], dtype=np.int64)
    for index in range(m+1):
        if(p_penaltyBinArr[index] == 0):
            if(index == 0):
                p_penaltyIndex = np.append(p_penaltyIndex, 0)
            else:
                A = tools.numCoeffsPoly(dim, index-1)
                B = tools.numCoeffsPoly(dim, index)
                for i in range(A, B):
                    p_penaltyIndex = np.append(p_penaltyIndex, i)

    q_penaltyIndex = np.array([],dtype=np.int64)
    for index in range(n+1):
        if(q_penaltyBinArr[index] == 0):
            if(q_penaltyBinArr[index] == 0):
                if(index == 0):
                    q_penaltyIndex = np.append(q_penaltyIndex, 0)
                else:
                    A = tools.numCoeffsPoly(dim, index-1)
                    B = tools.numCoeffsPoly(dim, index)
                    for i in range(A, B):
                        q_penaltyIndex = np.append(q_penaltyIndex, i)

    return p_penaltyIndex, q_penaltyIndex


"""
Strategies:
0: LSQ with SIP and without penalty
1: LSQ with SIP and some coeffs set to 0 (using constraints)
2: LSQ with SIP, penaltyParam > 0 and all or some coeffs in L1 term

"""
def optimize(dim=1,trainingscale="1x",m=2,n=2,
    box=np.array([[-1,1]],dtype=np.float64), infilePath="../f11.txt",
    outfilePath="../f11.json", strategy = 0, penaltyParam = 0,
    penaltyBin = np.array([])):

    M = tools.numCoeffsPoly(dim, m)
    N = tools.numCoeffsPoly(dim, n)

    X, Y = tools.readData(infilePath)

    struct_p = monomial.monomialStructure(dim, m)
    struct_q = monomial.monomialStructure(dim, n)

    trainingsize = 0
    if(trainingscale == "1x"):
        trainingsize = M+N
    elif(trainingscale == "2x"):
        trainingsize = 2*(M+N)
    elif(trainingscale == "Cp"):
        trainingsize = len(X)

    ipo = np.empty((trainingsize,2),"object")
    for i in range(trainingsize):
        ipo[i][0] = monomial.recurrence(X[i,:],struct_p)
        ipo[i][1]= monomial.recurrence(X[i,:],struct_q)

    cons = np.empty(trainingsize, "object")
    for trainingIndex in range(trainingsize):
        q_ipo = ipo[trainingIndex][1]
        cons[trainingIndex] = {'type': 'ineq', 'fun':robustSample, 'args':(q_ipo,M,N)}

    p_penaltyIndex = []
    q_penaltyIndex = []
    if(strategy ==1 or strategy == 2):
        p_penaltyIndex, q_penaltyIndex = createPenaltyIndexArr(penaltyBin[0],penaltyBin[1], dim, m, n)

    coeff0 = []
    if(strategy == 0):
        coeffs0 = np.zeros((M+N))
    elif(strategy == 1):
        coeffs0 = np.zeros((M+N))
        for index in p_penaltyIndex:
            cons = np.append(cons,{'type': 'eq', 'fun':coeffSetTo0, 'args':(index, "p", M)})
        for index in q_penaltyIndex:
            cons = np.append(cons,{'type': 'eq', 'fun':coeffSetTo0, 'args':(index, "q", M)})
    elif(strategy == 2):
        coeffs0 = np.zeros(2*(M+N))
        for index in p_penaltyIndex:
            cons = np.append(cons,{'type': 'ineq', 'fun':abs1, 'args':(index, "p", M, N)})
            cons = np.append(cons,{'type': 'ineq', 'fun':abs2, 'args':(index, "p", M, N)})
        for index in q_penaltyIndex:
            cons = np.append(cons,{'type': 'ineq', 'fun':abs1, 'args':(index, "q", M, N)})
            cons = np.append(cons,{'type': 'ineq', 'fun':abs2, 'args':(index, "q", M, N)})

    maxIterations = 100
    iterationInfo = []
    for iter in range(1,maxIterations+1):
        data = {}
        data['iterationNo'] = iter
        ret = {}
        if(strategy == 2):
            ret = minimize(leastSqObjWithPenalty, coeffs0, args = (ipo,Y,M,N,trainingsize, penaltyParam,p_penaltyIndex,q_penaltyIndex),method = 'SLSQP', constraints=cons, options={'iprint': 0,'ftol': 1e-6, 'disp': False})
        else:
            ret = minimize(leastSqObj, coeffs0, args = (ipo,Y,M,N,trainingsize),method = 'SLSQP', constraints=cons, options={'iprint': 0,'ftol': 1e-6, 'disp': False})
        coeffs = ret.get('x')
        leastSq = ret.get('fun')
        data['leastSqObj'] = leastSq
        data['pcoeff'] = coeffs[0:M].tolist()
        data['qcoeff'] = coeffs[M:M+N].tolist()

        if(strategy == 2):
            lsqsplit = {}
            l1term = computel1Term(coeffs,M,N,penaltyParam,p_penaltyIndex,q_penaltyIndex)
            lsqsplit['l1term'] = l1term
            lsqsplit['l2term'] = leastSq - l1term
            data['leastSqSplit'] = lsqsplit


        x0 = np.array([(box[i][0]+box[i][1])/2 for i in range(dim)], dtype=np.float64)

        ret = minimize(robustObj, x0, bounds=box, args = (coeffs,struct_q,M,N),method = 'L-BFGS-B')
        x = ret.get('x')
        robO = ret.get('fun')
        data['robustArg'] = x.tolist()
        data['robustObj'] = robO
        iterationInfo.append(data)
        if(robO >= 0.02):
            break
        q_ipo_new = monomial.recurrence(x,struct_q)
        cons = np.append(cons,{'type': 'ineq', 'fun':robustSample, 'args':(q_ipo_new,M,N)})

    if(len(iterationInfo) == maxIterations and iterationInfo[maxIterations-1]["robustObj"]<0.02):
        raise Exception("Could not find a robust objective")
    jsonRet = {}
    jsonRet['box'] = box.tolist()
    jsonRet['strategy'] = strategy
    if(strategy ==1 or strategy==2):
        jsonRet['chosenppenalty'] = penaltyBin[0]
        jsonRet['chosenqpenalty'] = penaltyBin[1]
    if(strategy == 2):
        jsonRet['lambda'] = penaltyParam
    jsonRet['dim'] = dim
    jsonRet['M'] = M
    jsonRet['N'] = N
    jsonRet['trainingscale'] = trainingscale
    jsonRet['trainingsize'] = trainingsize
    jsonRet['pcoeff'] = iterationInfo[len(iterationInfo)-1]["pcoeff"]
    jsonRet['qcoeff'] = iterationInfo[len(iterationInfo)-1]["qcoeff"]
    jsonRet['m'] = m
    jsonRet['n'] = n
    jsonRet['iterationinfo'] = iterationInfo

    import json
    jsonStr = json.dumps(jsonRet,indent=4, sort_keys=True)
    return jsonStr



# Static for now
dim=1
trainingscale = '1x'
m=2
n=3
infilePath = "../f11_noise_0.1.txt"
# infilePath = "../f11.txt"
box = np.array([[-1,1]],dtype=np.float64)
json = optimize(dim,trainingscale,m,n, box, infilePath,
            strategy=2,penaltyBin=[[1,0,0],[1,0,0,0]], penaltyParam = 10**-1)
print(json)




# END
