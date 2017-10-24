import unittest
import numpy as np

from gui.model import Model
from gui.observable import Observable

class TestGuiModel(unittest.TestCase):

    def test_save(self):
        array = np.array([1,2,2,3])
        materials = Observable([])
        boundaries = Observable({'topvar':'1',
                                 'bottomvar':'1',
                                 'leftvar':'1',
                                 'rightvar':'1'})
        moving_cells = Observable([1,2,3])
        m = Model(array, materials, boundaries, moving_cells)
        m.save_to_file('/tmp/test')

        m.load_from_file('/tmp/test')

    def test_load(self):
        pass
