import apprentice
import time
import unittest
import math
import numpy as np

def readAppSet(fname):
    return apprentice.AppSet(fname)


class PolynomialTestSingle(unittest.TestCase):
    def test(self):
        aps = readAppSet("test-restructure/la_30.json")
        ft = apprentice.FunctionalApprox(2, m=3, n=0)
        ft.setCoefficients(aps._PC)
        ft.setScaler(aps._SCLR.asDict)
        self.assertEqual(np.sum(aps.vals([0,1])-ft.vals([0,1]).ravel()), 0)

class RationalTestSingle(unittest.TestCase):
    def test(self):
        aps = readAppSet("test-restructure/sip_13.json")
        ft = apprentice.FunctionalApprox(2, m=1, n=3)
        ft.setCoefficients(aps._PC, aps._QC)
        ft.setScaler(aps._SCLR.asDict)
        self.assertEqual(np.sum(aps.vals([0,1]) - ft.vals([0,1]).ravel()), 0)

class PolynomialTestMulti(unittest.TestCase):
    def test(self):
        aps = readAppSet("test-restructure/la_30.json")
        ft = apprentice.FunctionalApprox(2, m=3, n=0)
        ft.setCoefficients(aps._PC)
        ft.setScaler(aps._SCLR.asDict)

        NTEST=1000
        X = np.random.rand(NTEST,2)

        AR = np.zeros((1793,NTEST))
        for num, x in enumerate(X):
            AR[:,num] = aps.vals(x)
        assert abs(np.sum(AR-ft.vals(X))) < 1e-9


class RationalTestMulti(unittest.TestCase):
    def test(self):
        aps = readAppSet("test-restructure/sip_31.json")
        ft = apprentice.FunctionalApprox(2, m=3, n=1)
        ft.setCoefficients(aps._PC, aps._QC)
        ft.setScaler(aps._SCLR.asDict)

        NTEST=1000
        X = np.random.rand(NTEST,2)

        AR = np.zeros((254,NTEST))
        for num, x in enumerate(X):
            AR[:,num] = aps.vals(x)
        assert abs(np.sum(AR-ft.vals(X))) < 1e-9

if __name__ == '__main__':

    unittest.main()
