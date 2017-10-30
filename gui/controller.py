import numpy as np
from tkinter import StringVar

from .model import Model
from .view import View
from .observable import Observable

class Controller(object):

    def __init__(self):
        self.array = Observable(np.zeros((2,2)))
        self.materials = Observable([])
        self.boundaries = Observable({})
        self.boundaries["left_bound"] = 'sleep'
        self.boundaries["top_bound"] = 'sleep'
        self.boundaries["right_bound"] = 'sleep'
        self.boundaries["bottom_bound"] = 'sleep'

        self.moving_cells = Observable([])

        self.Model = Model(self.array, self.materials, self.boundaries, self.moving_cells)
        self.View = View(fload=self.fload, fsave=self.fsave, fadd=self.fadd,
                         array=self.array, materials=self.materials, boundaries=self.boundaries, moving_cells=self.moving_cells)

        # bind variables
        self.materials.bind(self.View.update_lb_materials)
        self.moving_cells.bind(self.View.update_moving_cells_list)
        # self.boundaries.bind(self.View.update_boundaries_from_outside)
        self.boundaries["left_bound"] = 'sleep'
        self.boundaries["top_bound"] = 'sleep'
        self.boundaries["right_bound"] = 'sleep'
        self.boundaries["bottom_bound"] = 'sleep'

    def run(self):
        self.View.main_loop()

    def fload(self, fname):
        self.Model.load_from_file(fname[:-3])

        self.View.bottomvar.set(self.boundaries['bottom_bound'])
        self.View.topvar.set(self.boundaries['top_bound'])
        self.View.leftvar.set(self.boundaries['left_bound'])
        self.View.rightvar.set(self.boundaries['right_bound'])

        self.View.redraw_canvas()

    def fsave(self, fname):
        self.boundaries['top_bound'] = self.View.topvar.get()
        self.boundaries['bottom_bound'] = self.View.bottomvar.get()
        self.boundaries['left_bound'] = self.View.leftvar.get()
        self.boundaries['right_bound'] = self.View.rightvar.get()
        print(self.boundaries)
        self.Model.save_to_file(fname)

    def fadd(self, fname):
        self.Model.add_image(fname)
        self.View.redraw_canvas()
