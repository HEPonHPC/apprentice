import unittest
from apprentice.function import Function
from apprentice.polyset import PolySet
from apprentice.polynomialapproximation import PolynomialApproximation

from apprentice.util import Util

def get_rdm_poly(dim, order, n=1):
    import numpy as np
    NC = 2*Util.num_coeffs_poly(dim, order)

    X = np.random.random((NC, dim))

    return [PolynomialApproximation.from_interpolation_points(X, np.random.random(NC), m=np.random.randint(0,order+1)) for _ in range(n)]

#TODO test with rationals
# def get_rdm_rapp(dim, order, n=1):
    # import numpy as np
    # NC = 2*Util.num_coeffs_rapp(dim, order)

    # X = np.random.random((NC, dim))

    # return [RationalApproximation.from_interpolation_points(X, np.random.random(NC), m=order) for _ in range(n)]

class TestFunction(unittest.TestCase):


    def test_fromApproximations(self):
        DIM     =   5
        OMAX    =   6
        NOBJS   = 100
        NPOINTS =  20

        APPR = get_rdm_poly(DIM,OMAX,NOBJS)

        ps = PolySet.from_surrogates(APPR)

        import numpy as np
        X = np.random.random((NPOINTS, DIM))
        for i in range(NPOINTS):
            x = X[i]
            V = ps.vals(x)
            G = ps.grads(x)
            H = ps.hessians(x)
            for j in range(NOBJS):
                self.assertAlmostEqual(V[j], APPR[j](x))

                gg = APPR[j].gradient(x)
                hh = APPR[j].hessian(x)

                for d in range(DIM):
                    self.assertAlmostEqual(G[j][d], gg[d])

                    for t in range(DIM):
                        self.assertAlmostEqual(H[d][t][j], hh[d][t])


if __name__ == "__main__":
    unittest.main()
