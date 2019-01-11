def numCoeffsPoly(dim, order):
    """
    Number of coefficients a dim-dimensional polynomial of order order has.
    """
    ntok = 1
    r = min(order, dim)
    for i in range(r):
      ntok = ntok*(dim+order-i)/(i+1)
    return int(ntok)

def numCoeffsRapp(dim, order):
    """
    Number of coefficients a dim-dimensional polynomial of order order has.
    """
    return 1 + numCoeffsPoly(dim, order[0]) + numCoeffsPoly(dim, order[1])

def maxOrder(N, dim):
    """
    Utility function to find highest order polynomial
    for dataset with N dim-dimensional points.
    Intended to be used when constructing max ONB
    """
    from scipy.special import comb
    omax = 0
    while comb(dim + omax+1, omax+1) + 1 <= N: # The '+1' stands for a order 0 polynomial's dof
        omax+=1
    return omax


def readH5(fname, idx=[0], xfield="params", yfield="values"):
    """
    Read X,Y values etc from HDF5 file.
    By default, only the first object is read.
    The X and Y-value dataset names depend on the file of course, so we allow
    specifying wha to use. yfield can be values|errors with the test files.
    Returns a list of tuples of arrays : [ (X1, Y1), (X2, Y2), ...]
    The X-arrays are n-dimensional, the Y-arrays are always 1D
    """
    import numpy as np
    import h5py

    with h5py.File(fname, "r") as f:
        indexsize = f.get("index").size

    # A bit of logic here --- if idx is passed an empty list, ALL data is read from file.
    # Otherwise we need to check that we are not out of bounds.

    # pnames = [p for p in f.get(xfield).attrs["names"]]
    if len(idx)>0:
        assert(max(idx) <= indexsize)
    else:
        idx=[i for i in range(indexsize)]

    ret = []
    f = h5py.File(fname, "r")

    # Read parameters
    _X=np.array(f.get(xfield))

    # Read y-values
    for i in idx:
        _Y=np.atleast_1d(f.get(yfield)[i])
        USE = np.where( (~np.isinf(_Y))  & (~np.isnan(_Y)) )
        ret.append([ _X[USE], _Y[USE] ])

    f.close()

    return ret


def readData(fname, delimiter=","):
    """
    Read CSV formatted data. The last column is interpreted as
    function values while all other columns are considered
    parameter points.
    """
    import os
    if not os.path.exists(fname): raise Exception("File {} not found".format(fname))
    import numpy as np
    D = np.loadtxt(fname, delimiter=delimiter)
    X=D[:,0:-1]
    Y=D[:,-1]
    return X, Y

def readApprentice(fname):
    """
    Read an apprentice JSON file. We abuse try except here to
    figure out whether it's a rational or polynomial approximation.
    """
    import apprentice
    import os
    if not os.path.exists(fname): raise Exception("File {} not found".format(fname))
    try:
        app = apprentice.RationalApproximation(fname=fname)
    except:
        app = apprentice.PolynomialApproximation(fname=fname)
    return app
