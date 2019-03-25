import apprentice
import numpy as np
from timeit import default_timer as timer


def runRA(X, Y, fndesc, m, n, ts, outfile):
    dim = X[0].shape[0]
    M = apprentice.tools.numCoeffsPoly(dim,m)
    N = apprentice.tools.numCoeffsPoly(dim,n)
    totalcoeffsinRA = M+N
    if(ts == ".5x" or ts == "0.5x"):
        trainingsize = 0.5 * totalcoeffsinRA
    elif(ts == "1x"):
        trainingsize = totalcoeffsinRA
    elif(ts == "2x"):
        trainingsize = 2 * totalcoeffsinRA
    elif(ts == "Cp"):
         trainingsize = len(X)
    else: raise Exception("Training scale %s unknown"%(ts))

    if(trainingsize > len(X)):
        raise Exception("Not enough data for pdeg = %d, qdeg = %d and dim = %d. Require %d (%s) and only have %d"%(m,n,dim,trainingsize,ts,len(X)))

    train = range(trainingsize)
    from apprentice import RationalApproximationONB
    start = timer()
    ra = RationalApproximationONB(
    							X[train],
    							Y[train],
    							order=(m,n),
                                tol=-1
    )
    end = timer()

    radict = ra.asDict
    fittime = end-start
    radict["log"] = {"fittime":fittime}

    import json
    with open(outfile, "w") as f:
        json.dump(radict, f,indent=4, sort_keys=True)

    # ra1 = RationalApproximationONB(initDict=radict)
    # Y_pred = [ra1(x) for x in X]
    # print(Y_pred)


if __name__ == "__main__":

    import os, sys
    if len(sys.argv)!=7:
        print("Usage: {} infile fndesc m n trainingscale outfile".format(sys.argv[0]))
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print("Input file '{}' not found.".format(sys.argv[1]))

    try:
        X,Y = apprentice.tools.readData(sys.argv[1])
    except:
        DATA = apprentice.tools.readH5(sys.argv[1], [0])
        X, Y = DATA[0]


    runRA(
        X,
        Y,
        fndesc=sys.argv[2],
        m=int(sys.argv[3]),
        n=int(sys.argv[4]),
        ts=sys.argv[5],
        outfile=sys.argv[6]
    )
