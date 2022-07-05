"""
Utility functions for building and working with monomials
"""
import numpy as np


def mono_next_grlex(x: list[int]) -> list[int]:
    """
    Return next monomial. This is a deterministic procedure.

    :param x: current monomial
    :type x: list
    :return: next monomial
    :rtype: list

    :Example:

        ``mono_next_grlex([0,0,0])`` returns ``[0,0,1]``

        ``[0,0,0]`` corresponds to :math:`x^0y^0z^0`

        ``[0,0,1]`` corresponds to :math:`x^0y^0z^1`

    """
    m = len(x)
    #  Original Author: John Burkardt https://people.sc.fsu.edu/~jburkardt/py_src/monomial/monomial.html

    #  Find I, the index of the rightmost nonzero entry of X.
    i = 0
    for j in range(m, 0, -1):
        if 0 < x[j - 1]:
            i = j
            break

    #  set T = X(I)
    #  set X(I) to zero,
    #  increase X(I-1) by 1,
    #  increment X(M) by T-1.
    if i == 0:
        x[m - 1] = 1
        return x
    elif i == 1:
        t = x[0] + 1
        im1 = m
    elif 1 < i:
        t = x[i - 1]
        im1 = i - 1

    x[i - 1] = 0
    x[im1 - 1] = x[im1 - 1] + 1
    x[m - 1] = x[m - 1] + t - 1

    return x


def genStruct(mnm: list[int]) -> list[int]:
    """
    Generator for mono_next_grlex.
    Allows for a deterministic generation of monomial terms.

    :param mnm: zero array of length dimension
    :type mnm: list
    :return: generated structure
    :rtype: list

    :Example:

    .. code-block:: python

        g = genStruct(np.zeros(10))
        for i in range(10):
            print(next(g))

        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
        [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
        [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
        [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
        [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
        [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
        [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
        [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
        [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]

    """
    while True:
        yield mnm
        mnm = mono_next_grlex(mnm)


from functools import lru_cache


@lru_cache(maxsize=32)
def monomialStructure(dim: int, order: int):
    """
    Generate the monomial structure of a dim-dimensional polynomial of order order.
    The rows are the terms. The columns are the dimensions.

    :param dim: dimension monomial
    :type dim: int
    :param order: order of monomial
    :type order: int
    :return: array of terms (rows)
    :rtype: numpy.ndarray

    :Example:

        ``[2,3,4]`` means :math:`x^2y^3z^4`

        ``[[0 0],[0 1],[1 0],...]`` means :math:`x^0y^0, x^0y^1, x^1y^0,...`

    """
    import numpy as np
    from apprentice.tools import numCoeffsPoly
    ncmax = numCoeffsPoly(dim, order)
    gen = genStruct(np.zeros(dim, dtype=int))
    structure = np.empty((ncmax, dim), dtype=int)
    for i in range(ncmax):
        structure[i] = next(gen)
    if dim == 1:
        return structure.ravel()
    return structure


def recurrence1D(x: float, structure) -> list[float]:
    """
    Evaluate recurrence relation for a one-dimensional polynomial
    structure.

    :param x: parameter value
    :type x: float
    :param structure: monomial structure (1D)
    :type structure: numpy.ndarray
    :return:  individual terms of f(x) without multiplication
        by the coefficients with :math:`f(x) = ax^0 + bx^1 + cx^2 +dx^3`
    :rtype: numpy.ndarray

    :Exaxmple:

        let ``x = 0.1`` and the polynomial order be 3, then
        ``structure = [0,1,2,3]`` and therefore return ``[0.1^0, 0.1^1, 0.1^2, 0.1^3]``
        i.e. the individual terms of f(x) without multiplication
        by the coefficients with :math:`f(x) = ax^0 + bx^1 + cx^2 +dx^3`

    """
    return x ** structure


def recurrence(x: list[float], structure) -> list[float]:
    """
    Generalised version of recurrence1D for multidimensional problems.

    :param x: array of parameter values
    :type x: list
    :param structure: monomial structure (>=2D)
    :type structure: numpy.ndarray
    :return:  individual terms of f(x) WITHOUT multiplication
        by the coefficients
    :rtype: numpy.ndarray

    """
    assert (len(x) == structure.shape[1])
    return np.prod(x ** structure, axis=1, dtype=np.float64)


def recurrence2(x: list[float], structure, nnz=slice(None, None, None)) -> list[float]:
    """
    Efficient version of recurrence with beforehand knowledge of nonzero (nnz)
    structure elements.

    :param  x: array of parameter values
    :type x: list
    :param structure: monomial structure (>=2D)
    :type structure: numpy.ndarray
    :param nnz: nonzero structure elements
    :type nnz: list
    :return:  individual terms of f(x) WITHOUT multiplication
        by the coefficients
    :rtype: numpy.ndarray

    :Example:

    .. code-block:: python

        struct = monomialStructure(5,3)
        nnz = struct>0
        recurrence2([0.1,0.2,0.3,0.4,0.5], struct, nnz)

    """
    assert (len(x) == structure.shape[1])
    temp = np.ones((len(structure), len(x)))
    np.power(x, structure, where=nnz, out=(temp))
    return np.prod(temp, axis=1, dtype=np.float64)


def vandermonde(params, order):
    """
    Construct the Vandermonde matrix for polynomial of order order

    :param params: array of parameter values
    :type params: list
    :param order: order of monomial
    :type order: int
    :return: array of shape (len(params), numcoeffs(dim, order).
        The row i contains the recurrence relations obtained for param[i].
    :rtype: numpy.ndarray

    """
    import numpy as np
    try:
        dim = len(params[0])
    except:
        dim = 1

    from apprentice import tools
    s = monomialStructure(dim, order)
    if dim == 1:
        V = np.zeros((params.shape[0], tools.numCoeffsPoly(dim, order)), dtype=np.float64)
        for a, p in enumerate(params): V[a] = recurrence1D(p, s)
        return V
    else:
        V = np.ones((tools.numCoeffsPoly(dim, order), *params.shape), dtype=np.float64)
        np.power(params, s[:, np.newaxis], out=(V), where=s[:, np.newaxis] > 0)
        return np.prod(V, axis=2).T
