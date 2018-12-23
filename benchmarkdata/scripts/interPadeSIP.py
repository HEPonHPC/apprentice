import nlopt
import numpy as np
import sympy as sym


def leastSqObj(coeff, interpolants, Y, M=1, N=1, trainingsize=3):
    sum = 0
    # print("coeff= ", coeff)
    # print("Y= ", Y)
    for index in range(trainingsize):
        p_interpolant = interpolants[index][0]
        q_interpolant = interpolants[index][1]

        P = np.sum([coeff[i]*p_interpolant[i] for i in range(M)])

        # print ("P-in = ", p_interpolant)
        # print ("Q-in = ", q_interpolant)
        # print("P = ",P)


        Q = np.sum([coeff[i]*q_interpolant[i-M] for i in range(M,M+N)])
        # print ((Y[index] * Q - P)**2)
        sum += (Y[index] * Q - P)**2

    return sum

def robustSample(coeff, interpolants, M=1, N=1, trainingIndex = 0):
    q_interpolant = interpolants[trainingIndex][1]
    return np.sum([coeff[i]*q_interpolant[i-M] for i in range(M,M+N)])-1

def optimize(dim,trainingscale,m,n, infilePath):
    from apprentice import tools
    M = tools.numCoeffsPoly(dim, m)
    N = tools.numCoeffsPoly(dim, n)
    K = 1 + M + N

    X, Y = tools.readData(infilePath)

    from apprentice import monomial
    struct_p = monomial.monomialStructure(dim, m)
    struct_q = monomial.monomialStructure(dim, n)

    trainingsize = 0
    if(trainingscale == "1x"):
        trainingsize = len(struct_p) +len(struct_q)
    elif(trainingscale == "2x"):
        trainingsize = 2*(len(struct_p) +len(struct_q))
    elif(trainingscale == "Cp"):
        trainingsize = len(X)

    # print(Y[0:trainingsize])
    # rec_p = np.array(monomial.recurrence(X[0:trainingsize,:],struct_p))
    # rec_q = np.array(monomial.recurrence(X[0:trainingsize,:],struct_q))

    interpolants = np.empty((trainingsize,2),"object")
    for i in range(trainingsize):
        # print(monomial.recurrence(X[i,:],struct_p))
        interpolants[i][0] = monomial.recurrence(X[i,:],struct_p)
        interpolants[i][1]= monomial.recurrence(X[i,:],struct_q)

    # print(np.c_[struct_p])
    cons = np.empty(trainingsize, "object")
    for trainingIndex in range(trainingsize):
        cons[trainingIndex] = {'type': 'ineq', 'fun':robustSample, 'args':(interpolants,M,N,trainingIndex)}


    coeffs0 = np.zeros((M+N))
    # print(leastSqObj(coeffs,interpolants,Y, M,N,trainingsize))
    from scipy.optimize import minimize
    ret = minimize(leastSqObj, coeffs0, args = (interpolants,Y,M,N,trainingsize),method = 'SLSQP', constraints=cons)
    print(ret)


# Static for now
dim=1
trainingsize = '1x'
m=2
n=2
infilePath = "../f11_noise_0.1.txt"
optimize(dim,trainingsize,m,n, infilePath)





# END
