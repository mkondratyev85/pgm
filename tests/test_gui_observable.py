import unittest
import numpy as np

from gui.observable import Observable

class TestObservable(unittest.TestCase):

    def test_list(self):

        list = Observable([1,2,3])

        self.assertEqual(list[0], 1)
        self.assertEqual(list[1], 2)
        self.assertEqual(list[2], 3)

        for i, val in enumerate(list):
            self.assertEqual(i+1, val)

        list.append(4)
        self.assertEqual(len(list),4)
        self.assertEqual(list[-1],4)

        list[3]=0
        self.assertEqual(list[-1],0)

    def test_dict(self):
        dict = Observable({'a':1, 'b':2, 'c':3})
        self.assertEqual(dict['a'] , 1)
        self.assertEqual(dict['b'] , 2)
        self.assertEqual(dict['c'] , 3)

        for i, key in enumerate(dict):
            self.assertEqual(dict[key], i+1)

        dict['d'] = 4
        self.assertEqual(dict['d'], 4)

        dict['a'] = 1111
        self.assertEqual(dict['a'], 1111)
