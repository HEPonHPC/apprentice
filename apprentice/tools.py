def numNonZeroCoeff(app, threshold=1e-6):
    """
    Determine the number of non-zero coefficients for an approximation app.
    """
    n=0
    for p in app._pcoeff:
        if abs(p)>threshold: n+=1

    if hasattr(app, '_qcoeff'):
        for q in app._qcoeff:
            if abs(q)>threshold: n+=1

    return n


def numNLPoly(dim, order):
    if order <2: return 0
    else:
        return numCoeffsPoly(dim, order) - numCoeffsPoly(dim, 1)

def numNL(dim, order):
    """
    Number of non-linearities.
    """
    m, n = order
    if n ==0 : return numNLPoly(dim, m)
    if m <2 and n <2: return 0
    elif m<2 and n>=2:
        return numCoeffsPoly(dim, n) - numCoeffsPoly(dim, 1)
    elif n<2 and m>=2:
        return numCoeffsPoly(dim, m) - numCoeffsPoly(dim, 1)
    else:
        return numCoeffsRapp(dim, order) -  numCoeffsRapp(dim, (1,1))

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
    Number of coefficients a dim-dimensional rational approximation of order (m,n) has.
    """
    return numCoeffsPoly(dim, order[0]) + numCoeffsPoly(dim, order[1])

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
    specifying what to use. yfield can be values|errors with the test files.
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

# TODO rewrite such that yfield is a list of datasetnames, e.g. yfield=["values", "errors"]

def readH52(fname, idx=[0], xfield="params", yfield1="values", yfield2="errors"):
    """
    Read X,Y, erros values etc from HDF5 file.
    By default, only the first object is read.
    The X and Y-value dataset names depend on the file of course, so we allow
    specifying what to use. yfield can be values|errors with the test files.
    Returns a list of tuples of arrays : [ (X1, Y1, E1), (X2, Y2, E2), ...]
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
        _Y=np.atleast_1d(f.get(yfield1)[i])
        _E=np.atleast_1d(f.get(yfield2)[i])
        USE = np.where( (~np.isinf(_Y))  & (~np.isnan(_Y)) & (~np.isinf(_E))& (~np.isnan(_E)) )
        ret.append([ _X[USE], _Y[USE], _E[USE] ])

    f.close()

    return ret

def readPnamesH5(fname, xfield):
    """
    Get the parameter names from the hdf5 files params dataset attribute
    """
    import numpy as np
    import h5py

    with h5py.File(fname, "r") as f:
        pnames = [p.astype(str) for p in f.get(xfield).attrs["names"]]

    return pnames

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
    USE = np.where( (~np.isinf(Y))  & (~np.isnan(Y)) )
    return X[USE], Y[USE]

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

def getPolyGradient(coeff, X, dim=2, n=2):
    from apprentice import monomial
    import numpy as np
    struct_q = monomial.monomialStructure(dim, n)
    grad = np.zeros(dim,dtype=np.float64)

    for coord in range(dim):
        """
        Partial derivative w.r.t. coord
        """
        der = [0.]
        if dim==1:
            for s in struct_q[1:]: # Start with the linear terms
                der.append(s*X[0]**(s-1))
        else:
            for s in struct_q[1:]: # Start with the linear terms
                if s[coord] == 0:
                    der.append(0.)
                    continue
                term = 1.0
                for i in range(len(s)):
                    # print(s[i])
                    if i==coord:
                        term *= s[i]
                        term *= X[i]**(s[i]-1)
                    else:
                        term *= X[i]**s[i]
                der.append(term)
        grad[coord] = np.dot(der, coeff)
    return grad

def possibleOrders(N, dim, mirror=False):
    """
    Utility function to find all possible polynomials
    orders for dataset with N points in N dimension
    """
    from scipy.special import comb
    omax = 0
    while comb(dim + omax+1, omax+1) + 1 <= N: # The '+1' stands for a order 0 polynomial's dof
        omax+=1

    combs = []
    for m in reversed(range(omax+1)):
        for n in reversed(range(m+1)):
            if comb(dim + m, m) + comb(dim+n,n) <= N:
                combs.append((m,n))

    if mirror:
        temp=[tuple(reversed(i)) for i in combs]
        for t in temp:
            if not t in combs:
                combs.append(t)
    return combs
