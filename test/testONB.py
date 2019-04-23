import apprentice
import numpy as np


def mkApp(X,Y, M,N, tol):
    return apprentice.RationalApproximationONB(X=X, Y=Y, order=(M,N), tol=tol)

if __name__=="__main__":
    import sys
    fin, size = sys.argv[1].split(":")
    size=int(size)
    M=int(sys.argv[2])
    N=int(sys.argv[3])
    T = float(sys.argv[4])
    X, Y = apprentice.tools.readData(fin)
    R = mkApp(X[0:size],Y[0:size],M,N, T)
