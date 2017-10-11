import unittest
import numpy as np

from interpolate import interpolate2m

class TestMode(unittest.TestCase):

    def setUp(self):
        pass

    def test_interpolation(self):
        array = np.array([[0, 1, 1.5, 2],
                          [1, 2, 3,   2],
                          [2, 2, 3,   4]])

        # test interpolation in the center of the cells
        self.assertEqual(interpolate2m(np.array([0.5]), np.array([0.5]), array), 1.0)
        self.assertEqual(interpolate2m(np.array([1.5]), np.array([0.5]), array), 1.875)
        self.assertEqual(interpolate2m(np.array([2.5]), np.array([0.5]), array), 2.125)

        self.assertEqual(interpolate2m(np.array([0.5]), np.array([1.5]), array), 1.75)
        self.assertEqual(interpolate2m(np.array([1.5]), np.array([1.5]), array), 2.5)
        self.assertEqual(interpolate2m(np.array([2.5]), np.array([1.5]), array), 3)

        # test interpolation on nodes of the gird
        self.assertEqual(interpolate2m(np.array([0]), np.array([0]), array), 0.0)
        self.assertEqual(interpolate2m(np.array([1]), np.array([0]), array), 1.0)
        self.assertEqual(interpolate2m(np.array([2]), np.array([0]), array), 1.5)
        self.assertEqual(interpolate2m(np.array([3]), np.array([0]), array), 2.0)

        self.assertEqual(interpolate2m(np.array([0]), np.array([1]), array), 1.0)
        self.assertEqual(interpolate2m(np.array([1]), np.array([1]), array), 2.0)
        self.assertEqual(interpolate2m(np.array([2]), np.array([1]), array), 3.0)
        self.assertEqual(interpolate2m(np.array([3]), np.array([1]), array), 2.0)

        self.assertEqual(interpolate2m(np.array([0]), np.array([2]), array), 2.0)
        self.assertEqual(interpolate2m(np.array([1]), np.array([2]), array), 2.0)
        self.assertEqual(interpolate2m(np.array([2]), np.array([2]), array), 3.0)
        self.assertEqual(interpolate2m(np.array([3]), np.array([2]), array), 4.0)

        # test interpolation between nodes of the grid
        self.assertEqual(interpolate2m(np.array([.5]), np.array([0]), array), .5)
        self.assertEqual(interpolate2m(np.array([1.5]), np.array([0]), array), 1.25)
        self.assertEqual(interpolate2m(np.array([2.5]), np.array([0]), array), 1.75)

        self.assertEqual(interpolate2m(np.array([.5]), np.array([1]), array), 1.5)
        self.assertEqual(interpolate2m(np.array([1.5]), np.array([1]), array), 2.5)
        self.assertEqual(interpolate2m(np.array([2.5]), np.array([1]), array), 2.5)

        self.assertEqual(interpolate2m(np.array([.5]), np.array([2]), array), 2.0)
        self.assertEqual(interpolate2m(np.array([1.5]), np.array([2]), array), 2.5)
        self.assertEqual(interpolate2m(np.array([2.5]), np.array([2]), array), 3.5)

        self.assertEqual(interpolate2m(np.array([0]), np.array([0.5]), array), 0.5)
        self.assertEqual(interpolate2m(np.array([0]), np.array([1.5]), array), 1.5)

        self.assertEqual(interpolate2m(np.array([2]), np.array([0.5]), array), 2.25)
        self.assertEqual(interpolate2m(np.array([2]), np.array([1.5]), array), 3)

        self.assertEqual(interpolate2m(np.array([3]), np.array([0.5]), array), 2)
        self.assertEqual(interpolate2m(np.array([3]), np.array([1.5]), array), 3)

        # test interpolation outside 
        array = np.array([[0, 0, 0, 0],
                          [1, 1, 1, 1],
                          [2, 2, 2, 2]])

        self.assertEqual(interpolate2m(np.array([-0.5]), np.array([-0.5]), array), 0.0)
        self.assertEqual(interpolate2m(np.array([1.5]), np.array([-0.5]), array), 0.0)
        self.assertEqual(interpolate2m(np.array([3.5]), np.array([-0.5]), array), 0.0)

        self.assertEqual(interpolate2m(np.array([-0.5]), np.array([1]), array), 1)
        self.assertEqual(interpolate2m(np.array([3.5]), np.array([1]), array), 1)

        self.assertEqual(interpolate2m(np.array([-0.5]), np.array([2.5]), array),2.0)
        self.assertEqual(interpolate2m(np.array([1.5]), np.array([2.5]), array), 2.0)
        self.assertEqual(interpolate2m(np.array([3.5]), np.array([2.5]), array), 2.0)

        array = np.array([[0, 1, 2, 3],
                          [0, 1, 2, 3],
                          [0, 1, 2, 3]])

        self.assertEqual(interpolate2m(np.array([-0.5]), np.array([-0.5]), array), 0.0)
        self.assertEqual(interpolate2m(np.array([-0.5]), np.array([-0.5]), array), 0.0)
        self.assertEqual(interpolate2m(np.array([-0.5]), np.array([2.5]), array), 0.0)

        self.assertEqual(interpolate2m(np.array([1.5]), np.array([-0.5]), array), 1.5)
        self.assertEqual(interpolate2m(np.array([1.5]), np.array([2.5]), array), 1.5)

        self.assertEqual(interpolate2m(np.array([3.5]), np.array([-0.5]), array), 3)
        self.assertEqual(interpolate2m(np.array([3.5]), np.array([1]), array), 3)
        self.assertEqual(interpolate2m(np.array([3.5]), np.array([2.5]), array), 3)

if __name__=='__main__':
    unittest.main()
