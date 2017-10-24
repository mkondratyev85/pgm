import numpy as np

from .model import Model
from .view import View
from .observable import Observable

class Controller(object):

    def __init__(self):
        self.array = Observable(np.zeros((2,2)))
        self.materials = Observable([])
        self.boundaries = Observable({'topvar': '1',
                                 'bottomvar': '1',
                                 'leftvar': '1',
                                 'rightvar': '1'})
        self.moving_cells = Observable([])

        self.Model = Model(self.array, self.materials, self.boundaries, self.moving_cells)
        self.View = View(fload=self.fload, fsave=self.fsave, fadd=self.fadd,
                         array=self.array, materials=self.materials, boundaries=self.boundaries, moving_cells=self.moving_cells)

        # bind variables
        self.materials.bind(self.View.update_lb_materials)
        self.moving_cells.bind(self.View.update_moving_cells_list)

    def run(self):
        self.View.main_loop()

    def fload(self, fname):
        self.Model.load_from_file(fname[:-3])
        self.View.redraw_canvas()

    def fsave(self, fname):
        self.Model.save_to_file(fname)

    def fadd(self, fname):
        self.Model.add_image(fname)
        self.View.redraw_canvas()
