
import numpy as np

def chol_update(L, x, in_place=True):
    """
    Update the Lower factor of a cholesky decomposition, with a full vector x,
    such that the results M satisfis M M* = L L* + x x*, and * signifies the transpose.
    """
    x = np.array(x).flatten()
    n = size = x.size

    assert L.shape == (size, size), "L must be a square matrix of size same as x"

    if in_place:
        out_L = L
        out_x = x
    else:
        out_L = L.copy()
        out_x = x.copy()

    for k in range(n):
        r = np.sqrt(out_L[k, k]**2 + out_x[k]**2)
        c = r / out_L[k, k]
        s = out_x[k] / out_L[k, k]
        out_L[k, k] = r
        if k < n-1:
            out_L[k+1: , k] = (out_L[k+1: , k] + s*out_x[k+1: ]) / c
            #
            out_x[k+1: ] = c * out_x[k+1: ] - s * out_L[k+1: , k]

    return out_L, out_x

def chol_downdate(L, x,in_place=True):
    """
    Reverse of chol_update
    """
    x = np.array(x).flatten()
    n = size = x.size

    assert L.shape == (size, size), "L must be a square matrix of size same as x"

    if in_place:
        out_L = L
        out_x = x
    else:
        out_L = L.copy()
        out_x = x.copy()

    for k in range(n):
        r = np.sqrt(out_L[k, k]**2 - out_x[k]**2)
        c = r / out_L[k, k]
        s = out_x[k] / out_L[k, k]
        out_L[k, k] = r
        if k < n-1:
            out_L[k+1: , k] = (out_L[k+1: , k] - s*out_x[k+1: ]) / c
            out_x[k+1: ] = c * out_x[k+1: ] - s * out_L[k+1: , k]
    return out_L, out_x

def cholupdate(R, x,in_place=True):
    if in_place:
        out_R = R
        out_x = x
    else:
        out_R = R.copy()
        out_x = x.copy()
    import choldate
    choldate.cholupdate(out_R, out_x)
    return out_R,out_x

def choldowndate(R,x,in_place=True):
    if in_place:
        out_R = R
        out_x = x
    else:
        out_R = R.copy()
        out_x = x.copy()
    import choldate
    choldate.choldowndate(out_R, out_x)
    return out_R,out_x

def chol_diag_update(L, d, in_place=True,usecholdate=True):
    """
    Take a lower diagonal factor (from Cholesky), and return updated factorization
    of L L^T + diag(d), by applying a sequence of rank one updates
    """
    d = np.array(d).flatten()
    size = d.size
    assert L.shape == (size, size), "L must be a square matrix of size same as x"
    assert np.all(d>=0), "This function supports positive diagonal udpate only"

    if in_place:
        out = L
    else:
        out = L.copy()

    if usecholdate:
        out = out.transpose()

    d_sqrt = np.sqrt(d)

    x = np.empty(size)
    for ind in range(size):
        x[:] = 0.0
        x[ind] = d_sqrt[ind]
        if usecholdate:
            out, _ = cholupdate(out, x, in_place=True)
        else:
            out, _ = chol_update(out, x, in_place=True)

    if usecholdate:
        out = out.transpose()
    return out

def chol_diag_downdate(L, d, in_place=True,usecholdate=True):
    """
    Reverse chol_diag_update
    """
    d = np.array(d).flatten()
    size = d.size
    assert L.shape == (size, size), "L must be a square matrix of size same as x"
    assert np.all(d>=0), "This function supports positive diagonal udpate only"

    if in_place:
        out = L
    else:
        out = L.copy()

    if usecholdate:
        out = out.transpose()

    d_sqrt = np.sqrt(d)

    x = np.empty(size)
    for ind in range(size):
        x[:] = 0.0
        x[ind] = d_sqrt[ind]
        if usecholdate:
            out, _ = choldowndate(out, x, in_place=True)
        else:
            out, _ = chol_downdate(out, x, in_place=True)
    if usecholdate:
        out = out.transpose()
    return out