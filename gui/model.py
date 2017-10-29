import sys
import importlib
import os

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

    def save_to_file2(self, filename):
        with open(f'{filename}', 'w') as myfile:
            # write materials
            myfile.write('materials = [\n')
            for material in self.materials:
                myfile.write(f'    {material},\n')
            myfile.write(']\n\n')

            # write boundaries
            myfile.write('boundaries = {\n')
            for boundary in self.boundaries:
                myfile.write(f"    '{boundary}' : '{self.boundaries[boundary]}',\n")
            myfile.write('}\n\n')

            # write cells
            myfile.write('moving_cells = [\n')
            for cell in self.moving_cells:
                myfile.write(f'    {cell},\n')
            myfile.write(']\n\n')

        # save array
        np.save("%s" % (filename[:-3]), self.array)

        self.open_file2(filename)

    def open_file2(self, filename):
        print(filename)
        sys.path.append(os.path.dirname(f'{filename}.py'))
        modulename = os.path.splitext(os.path.basename(filename))[0]
        imported = importlib.import_module(modulename)
        sys.path.pop()

        array = np.load(f'{filename}.npy')
        self.materials.set(imported.materials)
        self.boundaries.set(imported.boundaries)
        self.moving_cells.set(imported.moving_cells)
        self.array.set(array)


    def save_to_file(self, fname):
        self.save_to_file2(fname)
        return

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
                  "top" : self.boundaries['top_bound'],
                  "bottom" : self.boundaries['bottom_bound'],
                  "left" : self.boundaries['left_bound'],
                  "right" : self.boundaries['right_bound'],
                  'moving_cells' : self.moving_cells,
                  "p":"{",
                  "p2":"}"
                }
        with  open(f'{fname}.py' ,'w') as myfile:
              myfile.write(template.format(**context))
        np.save("%s" % (fname), self.array)
        with open(f'{fname}.pickle', 'wb') as f:
            pickle.dump(materials.get(), f)
            pickle.dump(self.boundaries.get(), f)
            pickle.dump(self.moving_cells.get(), f)

    def load_from_file(self, fname):
        self.open_file2(fname)
        return
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
