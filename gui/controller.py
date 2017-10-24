import numpy as np
from matplotlib import pylab as plt

from .model import Model
from .view import View
from .observable import Observable
from .materials import materials as materials_

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
        print('fload')

    def fsave(self, fname):
        print('fsave')

    def fadd(self, fname):
        """ load image as png or npy format"""

        image = plt.imread(fname)
        image = image[:,:,0]*100+image[:,:,1]*10 + image[:,:,2]
        image_i, image_j = image.shape
        uniqe, vals = np.unique(image,return_inverse=True)
        self.array.set(vals.reshape((image_i, image_j)))
        self.View.redraw_canvas()

        materials = []
        for (i,item) in enumerate(uniqe):
            self.array[self.array==item] = i
            materials.append({"name":"default",
                              "rho":materials_["default"]["rho"],
                              "eta":materials_["default"]["eta"],
                              "mu":materials_["default"]["mu"],
                              "C":materials_["default"]["C"],
                              "sinphi":materials_["default"]["sinphi"], })
        self.materials.set(materials)
