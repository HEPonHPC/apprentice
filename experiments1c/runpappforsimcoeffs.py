import apprentice
import numpy as np
from timeit import default_timer as timer

def runPAforsimcoeffs(X, Y, fndesc, m, n, ts, outfile):
    import json
    dim = X[0].shape[0]
    M = apprentice.tools.numCoeffsPoly(dim,m)
    N = apprentice.tools.numCoeffsPoly(dim,n)
    totalcoeffsinRA = M+N
    padeg = 0
    pacoeffs = apprentice.tools.numCoeffsPoly(dim, padeg)
    while(pacoeffs < totalcoeffsinRA):
        padeg += 1
        pacoeffs = apprentice.tools.numCoeffsPoly(dim, padeg)

    if(ts == ".5x" or ts == "0.5x"):
        trainingsize = 0.5 * pacoeffs
    elif(ts == "1x"):
        trainingsize = pacoeffs
    elif(ts == "2x"):
        trainingsize = 2 * pacoeffs
    elif(ts == "Cp"):
         trainingsize = len(X)
    else: raise Exception("Training scale %s unknown"%(ts))

    if(trainingsize > len(X)):
        raise Exception("Not enough data for padeg = %d and dim = %d. Require %d (%s) and only have %d"%(padeg,dim,trainingsize,ts,len(X)))

    train = range(trainingsize)
    start = timer()
    pa = apprentice.PolynomialApproximation(
    							X[train],
    							Y[train],
    							order=padeg
            )
    end = timer()
    padict = pa.asDict
    fittime = end-start
    padict["log"] = {"fittime":fittime}
    import json
    with open(outfile, "w") as f:
        json.dump(padict, f,indent=4, sort_keys=True)


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


    runPAforsimcoeffs(
        X,
        Y,
        fndesc=sys.argv[2],
        m=int(sys.argv[3]),
        n=int(sys.argv[4]),
        ts=sys.argv[5],
        outfile=sys.argv[6]
    )
