import unittest
from apprentice.function import Function
from apprentice.polyset import PolySet
from apprentice.polynomialapproximation import PolynomialApproximation
from apprentice.leastsquares import LeastSquares

from apprentice.util import Util

import numpy as np

def get_rdm_poly(dim, order, n=1):
    import numpy as np
    NC = 2*Util.num_coeffs_poly(dim, order)

    X = np.random.random((NC, dim))

    return [PolynomialApproximation.from_interpolation_points(X, np.random.random(NC), m=np.random.randint(0,order+1)) for _ in range(n)]


class TestFunction(unittest.TestCase):


    def test_fromApproximations(self):
        DIM     =   3
        OMAX    =   3
        NOBJS   =   2
        NPOINTS =  20

        APPR = get_rdm_poly(DIM,OMAX,NOBJS)
        ps = PolySet.from_surrogates(APPR)

        DATA = np.random.random(NOBJS)
        ERRS = np.random.random(NOBJS)

        PRF = np.ones(NOBJS)

        A = LeastSquares(DIM, None, DATA, APPR, ERRS, NOBJS, APPR)#, fixed=[[0,0.5]])
        B = LeastSquares(DIM, None, DATA,   ps, ERRS, NOBJS, ps) ## Should be the faster one

        # print(A([0,0]), B([0.5,0,0]))
        # print(B.gradient([0.5,0,0]))
        # print(A.gradient([0,0]) - B.gradient([0.5,0,0])) # Works!

        # print( np.sum( B.hessian([0.5,0,0]) ))# - A.hessian([0,0,0,0]) ) )
        # print( np.sum( A.hessian([0,0]) ))# - A.hessian([0,0,0,0]) ) )
        # from IPython import embed
        # embed()

        X = np.random.random((NPOINTS, DIM))
        for i in range(NPOINTS):
            x = X[i]


            self.assertAlmostEqual(A(x), B(x))

    def test_fromApproximations_1D(self):
        DIM     =   1
        OMAX    =   3
        NOBJS   =   2
        NPOINTS =  20

        APPR = get_rdm_poly(DIM,OMAX,NOBJS)
        ps = PolySet.from_surrogates(APPR)

        DATA = np.random.random(NOBJS)
        ERRS = np.random.random(NOBJS)

        PRF = np.ones(NOBJS)

        A = LeastSquares(DIM, None, DATA, APPR, ERRS, NOBJS, APPR)#, fixed=[[0,0.5]])
        B = LeastSquares(DIM, None, DATA,   ps, ERRS, NOBJS, ps) ## Should be the faster one

        X = np.random.random((NPOINTS, DIM))
        for i in range(NPOINTS):
            x = X[i]


            self.assertAlmostEqual(A(x), B(x))

if __name__ == "__main__":
    unittest.main()

