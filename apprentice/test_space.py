import unittest
from apprentice.space import Space

class TestSpace(unittest.TestCase):

    def test_sth(self):
        S = Space(2, [0,0], [2,3], pnames=["parA", "parB"])

    def test_mkSubSpace(self):
        S = Space(3, [0,0,1], [2,3,4], pnames=["parA", "parB", "parC"])
        S2 = S.mkSubSpace([0,2])
        self.assertEqual(S2.dim,2)

    def test_sample(self):
        S = Space(3, [0,0,1], [2,3,4], pnames=["parA", "parB", "parC"])

    def test_center(self):
        S = Space(3, [0,0,1], [2,3,4])
        self.assertEqual(S.center, [1,1.5,2.5])

    def test_fromList(self):
        data = [(0,1),(10,100)]
        S = Space.fromList(data)
        self.assertEqual(S.dim, 2)


if __name__ == "__main__":
    unittest.main()
