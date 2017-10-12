import unittest
import numpy as np

from view import Matplot


class TestInterpolation(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_and_set_parameters(self):
        view = Matplot()
        self.assertTrue(view.dx is None)

        self.assertRaises(KeyError, Matplot, D=1)

        self.assertEqual(view.dx, None)
        view.dx=2
        self.assertEqual(view.dx, 2)

        array = np.array([[1,1,1,1,1],
                          [2,2,2,2,1],
                          [2,3,3,2,1],
                          [3,3,4,4,2]])

        Vy = np.array([[1,2,3,3,0],
                       [1,2,3,3,0],
                       [1,2,3,3,0],
                       [1,2,3,3,0]])

        Vx = np.array([[1,1,1,1,1],
                       [2,2,2,2,2],
                       [3,3,3,3,3],
                       [0,0,0,0,0]])

        view = Matplot(width = 10, height = 10, git = 'git', i_res=4, j_res=5, dx=4, dy=4, figname='/tmp/' )
        p = {'T' : 1,
             'step' : 1,
             'eta_n' : array,
             'Vx' : Vx,
             'Vy' : Vy,
             'e_xx' : array,
             'e_xy' : array,
             's_xx' : array,
             's_xy' : array,
             'mu_n' : array,
             'w' : array,
             'sii' : array,
             'P' : array,

            }
        view.plot12(parameters=p)





