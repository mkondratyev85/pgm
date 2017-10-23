from .model import Model
from .view import View
from .observable import Observable


class Controller(object):

    def __init__(self):
        array = None
        materials = Observable([])
        boundaries = Observable({'topvar': '1',
                                 'bottomvar': '1',
                                 'leftvar': '1',
                                 'rightvar': '1'})
        moving_cells = Observable([1, 2, 3])

        self.Model = Model(array, materials, boundaries, moving_cells)
        self.View = View(fload=self.fload, fsave=self.fsave, fadd=self.fadd,
                         array=array, materials=materials, boundaries=boundaries, moving_cells=moving_cells)

    def run(self):
        self.View.main_loop()

    def fload(self, fname):
        print('fload')

    def fsave(self, fname):
        print('fsave')

    def fadd(self, fname):
        print('fadd')
