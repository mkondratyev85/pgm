import sys
import importlib
import os

import numpy as np
from matplotlib import pylab as plt

from .template import template
from .materials import materials as materials_

class Model(object):

    def __init__(self, array, materials, boundaries, moving_cells):
        self.array = array
        self.materials = materials
        self.boundaries = boundaries
        self.moving_cells = moving_cells

    def save_to_file(self, filename):
        with open(f'{filename}', 'w') as myfile:
            # write settings
            myfile.write(template)

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

    def load_from_file(self, filename):
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
