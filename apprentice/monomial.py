import numpy as np
# from numba import jit, njit

# @njit
def mono_next_grlex(m, x):
    #  Author:
    #
    #    John Burkardt
    #
    #     TODO --- figure out the licensing thing https://people.sc.fsu.edu/~jburkardt/py_src/monomial/monomial.html

    #  Find I, the index of the rightmost nonzero entry of X.
    i = 0
    for j in range(m, 0, -1):
        if 0 < x[j-1]:
            i = j
            break

    #  set T = X(I)
    #  set X(I) to zero,
    #  increase X(I-1) by 1,
    #  increment X(M) by T-1.
    if i == 0:
        x[m-1] = 1
        return x
    elif i == 1:
        t = x[0] + 1
        im1 = m
    elif 1 < i:
        t = x[i-1]
        im1 = i - 1

    x[i-1] = 0
    x[im1-1] = x[im1-1] + 1
    x[m-1] = x[m-1] + t - 1

    return x

def genStruct(dim, mnm):
    while True:
        yield mnm
        mnm =  mono_next_grlex(dim, mnm)

from functools import lru_cache
@lru_cache(maxsize=32)
def monomialStructure(dim, order):
    import numpy as np
    import copy
    from apprentice import tools
    ncmax = tools.numCoeffsPoly(dim, order)
    gen = genStruct(dim, np.zeros(dim))
    structure = np.array([ copy.copy(next(gen)) for _ in range(ncmax)], dtype=int)
    # Dimension one requires some extra treatment when returning ---writing out is fine
    if dim==1:
        return structure.ravel()
    return structure

def recurrence1D(X, structure):
    return X**structure

def recurrence(X, structure):
    """
    Create the parameter combination vector for a particular structure,
    or in more mathy terms, the recurrence relation for X in a monomial basis
    structure.
    """
    return np.prod(X**structure, axis=1, dtype=np.float64)


def recurrence2(X, structure, nnz):
    temp = np.ones((len(structure), len(X)))
    np.power(X, structure, where=nnz, out=(temp))
    return np.prod(temp, axis=1)

def vandermonde(params, order):
    """
    Construct the Vandermonde matrix.
    """
    import numpy as np
    try:
        dim = len(params[0])
    except:
        dim = 1

    from apprentice import tools
    s = monomialStructure(dim, order)
    if len(params[0]) == 1:
        V = np.zeros((len(params), tools.numCoeffsPoly(dim, order)), dtype=np.float64)
        for a, p in enumerate(params): V[a]=recurrence1D(p, s)
        return V
    else:
        V = np.ones((tools.numCoeffsPoly(dim, order), *params.shape), dtype=np.float64)
        np.power(params, s[:, np.newaxis], out=(V), where=s[:, np.newaxis]>0)
        return np.prod(V, axis=2).T

if __name__=="__main__":
    print(monomialStructure(2,3))
