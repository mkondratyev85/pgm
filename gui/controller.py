import numpy as np
from matplotlib import pylab as plt

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
        self.moving_cells = Observable([1, 2, 3])

        self.Model = Model(self.array, self.materials, self.boundaries, self.moving_cells)
        self.View = View(fload=self.fload, fsave=self.fsave, fadd=self.fadd,
                         array=self.array, materials=self.materials, boundaries=self.boundaries, moving_cells=self.moving_cells)

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
