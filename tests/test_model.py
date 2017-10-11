import unittest
import matplotlib.pylab as plt
import numpy as np

from interpolate import (interpolate2m,
                         interpolate2m_vect,
                         interpolate,
                         interpolate_harmonic)

class TestInterpolation(unittest.TestCase):

    def setUp(self):
        pass

    def test_interplate(self):

        # test simple interpolation onto grid nodes

        mxx = np.array([0,1,0,1,0,1])#, .5, .5])
        myy = np.array([0,0,1,1,2,2])#, .5,1.5])
        v   = np.array([0,1,0,1,0,1])#, .5, .5])

        result = interpolate(mxx, myy, 3, 2, (v,))[0]
        self.assertTrue(np.array_equal(result,
                                       np.array([[0,1],
                                                 [0,1],
                                                 [0,1]])))


        # test interpolation iside grid cells

        mxx = np.array([0.25, 0.25, 0.25, 0.75, 0.75, 0.75])
        myy = np.array([0.25, 1,    1.75, 0.25, 1,    1.75])
        v   = np.array([0,   1,     2,    1,    2,    3])
        result = interpolate(mxx, myy, 3, 2, (v,))[0]
        self.assertTrue(np.array_equal(result,
                                       np.array([[0,1],
                                                 [1,2],
                                                 [2,3]])))

        mxx = np.array([ 0.25, 0.75, 0.25, 0.75,-0.25,1.25])
        myy = np.array([-0.25,-0.25, 2.25, 2.25,1, 1])
        v   = np.array([ 0,   1, 2, 3, 4, 5])
        result = interpolate(mxx, myy, 3, 2, (v,))[0]
        self.assertTrue(np.array_equal(result,
                                       np.array([[0.,1.],
                                                 [4.,5.],
                                                 [2.,3.]])))

        # test complicated interpolation
        np.random.seed(2)
        i_res, j_res = 3,5
        mxx = np.random.uniform(0, j_res, 1500)-.5
        myy = np.random.uniform(0, i_res, 1500)-.5
        v = mxx + myy
        result = interpolate(mxx, myy, i_res, j_res, (v,))[0]

        result_int = np.array([[0,1,2,3,4],
                               [1,2,3,4,5],
                               [2,3,4,5,6]])
        diff = result - result_int
        self.assertTrue((diff<0.09).all(),
                        ' Interpolated gird should differ from ideal less then 0.09' )

        #plt.imshow(result, interpolation="none")
        #plt.scatter(mxx, myy, s=25, c=v, edgecolors='black')
        #plt.show()
        ##print (result)

    def test_interpolate_harmonic(self):
        # test simple interpolation onto grid nodes

        mxx = np.array([0,1,0,1,0,1])#, .5, .5])
        myy = np.array([0,0,1,1,2,2])#, .5,1.5])
        v   = np.array([0,1,0,1,0,1])#, .5, .5])

        result = interpolate_harmonic(mxx, myy, 3, 2, v)
        self.assertTrue(np.array_equal(result,
                                       np.array([[0,1],
                                                 [0,1],
                                                 [0,1]])))

        # test interpolation iside grid cells

        mxx = np.array([0.25, 0.25, 0.25, 0.75, 0.75, 0.75])
        myy = np.array([0.25, 1,    1.75, 0.25, 1,    1.75])
        v   = np.array([0,   1,     2,    1,    2,    3])
        result = interpolate_harmonic(mxx, myy, 3, 2, v)
        self.assertTrue(np.array_equal(result,
                                       np.array([[0,1],
                                                 [1,2],
                                                 [2,3]])))

        mxx = np.array([ 0.25, 0.75, 0.25, 0.75,-0.25,1.25])
        myy = np.array([-0.25,-0.25, 2.25, 2.25,1, 1])
        v   = np.array([ 0,   1, 2, 3, 4, 5])
        result = interpolate_harmonic(mxx, myy, 3, 2, v)
        self.assertTrue(np.array_equal(result,
                                       np.array([[0.,1.],
                                                 [4.,5.],
                                                 [2.,3.]])))

        # test complicated interpolation
        np.random.seed(1)
        i_res, j_res = 3,5
        mxx = np.random.uniform(0, j_res, 1500)-.5
        myy = np.random.uniform(0, i_res, 1500)-.5
        v = mxx + myy
        result = interpolate_harmonic(mxx, myy, i_res, j_res, v)

        result_int = np.array([[0,1,2,3,4],
                               [1,2,3,4,5],
                               [2,3,4,5,6]])
        diff = result - result_int

        self.assertTrue((diff<0.3).all(),
                        ' Interpolated gird should differ from ideal less then 0.3' )

    def test_interpolate2m_vect(self):
        array = np.array([[0, 1, 1.5, 2],
                          [1, 2, 3,   2],
                          [2, 2, 3,   4]])

        # test interpolation in the center of the cells
        self.assertEqual(interpolate2m_vect(np.array([0.5]), np.array([0.5]), array), 1.0)
        self.assertEqual(interpolate2m_vect(np.array([1.5]), np.array([0.5]), array), 1.875)
        self.assertEqual(interpolate2m_vect(np.array([2.5]), np.array([0.5]), array), 2.125)

        self.assertEqual(interpolate2m_vect(np.array([0.5]), np.array([1.5]), array), 1.75)
        self.assertEqual(interpolate2m_vect(np.array([1.5]), np.array([1.5]), array), 2.5)
        self.assertEqual(interpolate2m_vect(np.array([2.5]), np.array([1.5]), array), 3)

        # test interpolation on nodes of the gird
        self.assertEqual(interpolate2m_vect(np.array([0]), np.array([0]), array), 0.0)
        self.assertEqual(interpolate2m_vect(np.array([1]), np.array([0]), array), 1.0)
        self.assertEqual(interpolate2m_vect(np.array([2]), np.array([0]), array), 1.5)
        self.assertEqual(interpolate2m_vect(np.array([3]), np.array([0]), array), 2.0)

        self.assertEqual(interpolate2m_vect(np.array([0]), np.array([1]), array), 1.0)
        self.assertEqual(interpolate2m_vect(np.array([1]), np.array([1]), array), 2.0)
        self.assertEqual(interpolate2m_vect(np.array([2]), np.array([1]), array), 3.0)
        self.assertEqual(interpolate2m_vect(np.array([3]), np.array([1]), array), 2.0)

        self.assertEqual(interpolate2m_vect(np.array([0]), np.array([2]), array), 2.0)
        self.assertEqual(interpolate2m_vect(np.array([1]), np.array([2]), array), 2.0)
        self.assertEqual(interpolate2m_vect(np.array([2]), np.array([2]), array), 3.0)
        self.assertEqual(interpolate2m_vect(np.array([3]), np.array([2]), array), 4.0)

        # test interpolation between nodes of the grid
        self.assertEqual(interpolate2m_vect(np.array([.5]), np.array([0]), array), .5)
        self.assertEqual(interpolate2m_vect(np.array([1.5]), np.array([0]), array), 1.25)
        self.assertEqual(interpolate2m_vect(np.array([2.5]), np.array([0]), array), 1.75)

        self.assertEqual(interpolate2m_vect(np.array([.5]), np.array([1]), array), 1.5)
        self.assertEqual(interpolate2m_vect(np.array([1.5]), np.array([1]), array), 2.5)
        self.assertEqual(interpolate2m_vect(np.array([2.5]), np.array([1]), array), 2.5)

        self.assertEqual(interpolate2m_vect(np.array([.5]), np.array([2]), array), 2.0)
        self.assertEqual(interpolate2m_vect(np.array([1.5]), np.array([2]), array), 2.5)
        self.assertEqual(interpolate2m_vect(np.array([2.5]), np.array([2]), array), 3.5)

        self.assertEqual(interpolate2m_vect(np.array([0]), np.array([0.5]), array), 0.5)
        self.assertEqual(interpolate2m_vect(np.array([0]), np.array([1.5]), array), 1.5)

        self.assertEqual(interpolate2m_vect(np.array([2]), np.array([0.5]), array), 2.25)
        self.assertEqual(interpolate2m_vect(np.array([2]), np.array([1.5]), array), 3)

        self.assertEqual(interpolate2m_vect(np.array([3]), np.array([0.5]), array), 2)
        self.assertEqual(interpolate2m_vect(np.array([3]), np.array([1.5]), array), 3)

        # test interpolation outside 
        array = np.array([[0, 0, 0, 0],
                          [1, 1, 1, 1],
                          [2, 2, 2, 2]])

        self.assertEqual(interpolate2m_vect(np.array([-0.5]), np.array([-0.5]), array), 0.0)
        self.assertEqual(interpolate2m_vect(np.array([1.5]), np.array([-0.5]), array), 0.0)
        self.assertEqual(interpolate2m_vect(np.array([3.5]), np.array([-0.5]), array), 0.0)

        self.assertEqual(interpolate2m_vect(np.array([-0.5]), np.array([1]), array), 1)
        self.assertEqual(interpolate2m_vect(np.array([3.5]), np.array([1]), array), 1)

        self.assertEqual(interpolate2m_vect(np.array([-0.5]), np.array([2.5]), array),2.0)
        self.assertEqual(interpolate2m_vect(np.array([1.5]), np.array([2.5]), array), 2.0)
        self.assertEqual(interpolate2m_vect(np.array([3.5]), np.array([2.5]), array), 2.0)

        array = np.array([[0, 1, 2, 3],
                          [0, 1, 2, 3],
                          [0, 1, 2, 3]])

        self.assertEqual(interpolate2m_vect(np.array([-0.5]), np.array([-0.5]), array), 0.0)
        self.assertEqual(interpolate2m_vect(np.array([-0.5]), np.array([-0.5]), array), 0.0)
        self.assertEqual(interpolate2m_vect(np.array([-0.5]), np.array([2.5]), array), 0.0)

        self.assertEqual(interpolate2m_vect(np.array([1.5]), np.array([-0.5]), array), 1.5)
        self.assertEqual(interpolate2m_vect(np.array([1.5]), np.array([2.5]), array), 1.5)

        self.assertEqual(interpolate2m_vect(np.array([3.5]), np.array([-0.5]), array), 3)
        self.assertEqual(interpolate2m_vect(np.array([3.5]), np.array([1]), array), 3)
        self.assertEqual(interpolate2m_vect(np.array([3.5]), np.array([2.5]), array), 3)

    def test_interpolate2m(self):
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
