"""
Test Monomials
"""
import unittest
import numpy as np

from apprentice.monomial import mono_next_grlex
from apprentice.monomial import monomialStructure
from apprentice.monomial import recurrence1D
from apprentice.monomial import recurrence
from apprentice.monomial import vandermonde

class TestMonomial(unittest.TestCase):
    def test_mono_next_grlex(self):
        """
        Test the monomial grlex function
        """
        self.assertEqual(mono_next_grlex([0,0,0]),[0,0,1])
        self.assertEqual(mono_next_grlex([0,1,0]),[1,0,0])
        self.assertEqual(mono_next_grlex([1,0,0]),[0,0,2])

    def test_monomialStructure(self):
        """
        Test the monomial structure
        """
        self.assertEqual(list(monomialStructure(1,2)) , [0,1,2])
        self.assertEqual(list(monomialStructure(2,2).ravel()) , [0,0,0,1,1,0,0,2,1,1,2,0])

        bigstr = monomialStructure(20,4)
        self.assertEqual(bigstr.shape, (10626, 20))


    def test_recurrence1D(self):
        """
        Tes one dimensional recurrence structure for order 0 to 3
        """
        for o in range(4):
            struct = monomialStructure(1,o)
            self.assertEqual(np.sum(recurrence1D(0, struct)), 1)
            self.assertEqual(np.sum(recurrence1D(1, struct)), len(struct))

    def test_recurrence(self):
        """
        Tes d>1 dimensional recurrence structure for order 4 (tests for dimensions 2 to 20)
        """
        for d in range(2,20):
            struct = monomialStructure(d,4)
            self.assertEqual(np.sum(recurrence(np.zeros(d), struct)), 1)
            self.assertEqual(np.sum(recurrence(np.ones(d) , struct)), len(struct))

    def test_vandermonde(self):
        """
        Test Vandermonde matrix construction for polynomial of order 3
        """
        V = vandermonde(np.array([[0,0],[1,1]]), 3)
        struct = monomialStructure(2,3)
        self.assertEqual(list(V[0]), list(recurrence(np.zeros(2), struct)))
        self.assertEqual(list(V[1]), list(recurrence(np.ones(2), struct)))


if __name__ == "__main__":
    unittest.main();
