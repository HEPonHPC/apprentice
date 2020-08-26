import apprentice

import unittest

import numpy as np

def readAppSet(fname):
    return apprentice.AppSet(fname)

def mkFteam(ndim, coeff, sdict):
    FT = apprentice.FunctionalApprox(ndim, m=3, n=0)
    FT.setCoefficients(coeff)
    FT.setScaler(sdict)
    return FT


class MyTest(unittest.TestCase):
    def test(self):
        aps = readAppSet("test-restructure/la_30.json")
        ft = mkFteam(2, aps._PC, aps._SCLR.asDict)
        self.assertEqual(np.sum(aps.vals([0,1])-ft.vals([0,1]).ravel()), 0)


if __name__ == '__main__':

    unittest.main()
