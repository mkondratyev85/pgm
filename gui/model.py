import numpy as np
from matplotlib import pylab as plt
import pickle

from .template import template
from .materials import materials as materials_

class Model(object):

    def __init__(self, array, materials, boundaries, moving_cells):
        self.array = array
        self.materials = materials
        self.boundaries = boundaries
        self.moving_cells = moving_cells

    def save_to_file(self, fname):
        materials = self.materials
        rho_list = [material['rho'] for  material in materials]
        eta_list = [material['eta'] for  material in materials]
        mu_list = [material['mu'] for  material in materials]
        C_list = [material['C'] for  material in materials]
        sinphi_list = [material['sinphi'] for  material in materials]
        context={ "fname": "%s" % fname,
                  "rho_list" : '%s' % rho_list,
                  "eta_list" : '%s' % eta_list,
                  "mu_list" : '%s' % mu_list,
                  "C_list" : '%s' % C_list,
                  "sinphi_list" : '%s' % sinphi_list,
                  "top" : self.boundaries['topvar'],
                  "bottom" : self.boundaries['bottomvar'],
                  "left" : self.boundaries['leftvar'],
                  "right" : self.boundaries['rightvar'],
                  'moving_cells' : self.moving_cells,
                  "p":"{",
                  "p2":"}"
                }
        with  open(f'{fname}.py' ,'w') as myfile:
              myfile.write(template.format(**context))
        np.save("%s" % (fname), self.array)
        print(materials.get())
        with open(f'{fname}.pickle', 'wb') as f:
            pickle.dump(materials.get(), f)
            pickle.dump(self.boundaries.get(), f)
            pickle.dump(self.moving_cells.get(), f)

    def load_from_file(self, fname):
        with open(f'{fname}.pickle', 'rb') as f:
            materials = pickle.load(f)
            boundaries = pickle.load(f)
            moving_cells = pickle.load(f)
        array = np.load(f'{fname}.npy')
        self.materials.set(materials)
        self.boundaries.set(boundaries)
        self.moving_cells.set(moving_cells)
        self.array.set(array)

    def add_image(self, fname):
        """ load image as png or npy format"""
        image = plt.imread(fname)
        image = image[:,:,0]*100+image[:,:,1]*10 + image[:,:,2]
        image_i, image_j = image.shape
        uniqe, vals = np.unique(image,return_inverse=True)
        self.array.set(vals.reshape((image_i, image_j)))

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
