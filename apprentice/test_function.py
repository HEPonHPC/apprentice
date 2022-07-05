import unittest
from apprentice.function import Function
from apprentice.polynomialapproximation import PolynomialApproximation

class SimpleFunction(Function):
    def objective(self, x):
        return x**2

from apprentice.surrogatemodel import SurrogateModel
class Dummy(SurrogateModel):
    def __repr__(self):
        return "I am a dummy"
    def f_x(self, x):
        return 2*x



class TestFunction(unittest.TestCase):

    def test_empty(self):
        fn = Function.mk_empty(3)
        self.assertEqual(fn.dim,3)


    def test_simple(self):
        sf = SimpleFunction.mk_empty(1)
        self.assertEqual(sf(3), 9)

    def test_fromSpace(self):
        sf = SimpleFunction.from_space( ([0,2],[4,7]))
        self.assertEqual(sf.dim, 2)

    def test_fromApproximations(self):
        APPR = []

        APPR.append(PolynomialApproximation(2))
        APPR.append(PolynomialApproximation(2))
        APPR.append(PolynomialApproximation(2))
        APPR.append(PolynomialApproximation(2))
        APPR.append(PolynomialApproximation(2))


        sf = Function.from_surrogates( APPR )
        self.assertEqual(sf.dim, 2)

        with self.assertRaises(Exception) as context:
            sf([0,0])

        self.assertTrue("The function objective must be implemented in the derived class" in str(context.exception))


    def test_mixed(self):
        APPR = []

        APPR.append(PolynomialApproximation(2))
        APPR.append(Dummy(2))
        sf = Function.from_surrogates( APPR )
        self.assertEqual(sf.dim, 2)

        print(Dummy(2))

        d = Dummy(2)
        print(d(3))



    def test_fixed(self):
        sf = SimpleFunction.from_space( ([0,2],[4,7]), fixed=[[0,14]] ) # Fix the first dimension to the value 14
        self.assertEqual(sf([999])[0], 196)




if __name__ == "__main__":
    unittest.main()
