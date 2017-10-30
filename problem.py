import sys
import importlib
import os
import pickle
import numpy as np
import random

class Problem(object):

    settings = {}

    def __init__(self, default_settings_filename = None):

        # set defaults
        if default_settings_filename:
            self.load_settings_from_file(default_settings_filename)

    def __getitem__(self, index):
        return self.settings[index]

    def load_from_file(self, filename):
        sys.path.append(os.path.dirname(filename))
        modulename = os.path.splitext(os.path.basename(filename))[0]
        imported = importlib.import_module(modulename)
        sys.path.pop()

        settings = imported.settings
        materials = imported.materials
        boundaries = imported.boundaries
        moving_cells = imported.moving_cells

        self.set_settings_from_dictionary(settings=settings,
                                          check_for_None=False)
        return materials, boundaries, moving_cells

    def set_settings_from_dictionary(self,
                                     settings=None,
                                     check_for_None=True):

        self.settings.update(settings)

    def load_model(self, filename):
        # materials, boundaries, moving_cells  = self.load_problem_description(filename)
        materials, boundaries, moving_cells  = self.load_from_file(filename)
        mxx, myy, values, moving_cells_index_list = self.load_image(filename, moving_cells)

        rho_key = np.asarray([material['rho'] for  material in materials])
        eta_key = np.asarray([material['eta'] for  material in materials])
        mu_key = np.asarray([material['mu'] for  material in materials])
        C_key = np.asarray([material['C'] for  material in materials])
        sinphi_key = np.asarray([material['sinphi'] for  material in materials])

        m_cat = np.copy(values)
        m_rho = rho_key[values]
        m_eta = eta_key[values]
        m_mu = mu_key[values]
        m_C = C_key[values]
        m_sinphi = sinphi_key[values]

        m_s_xx = np.zeros(np.shape(mxx))
        m_s_xy = np.zeros(np.shape(mxx))
        m_e_xx = np.zeros(np.shape(mxx))
        m_e_xy = np.zeros(np.shape(mxx))
        m_P    = np.zeros(np.shape(mxx))

        top_bound = boundaries['top_bound']
        bottom_bound = boundaries['bottom_bound']
        left_bound = boundaries['left_bound']
        right_bound = boundaries['right_bound']

        self.settings["mxx"] = mxx
        self.settings["myy"] = myy
        self.settings["m_cat"] = m_cat
        self.settings["m_rho"] = m_rho
        self.settings["m_eta"] = m_eta
        self.settings["m_mu"] = m_mu
        self.settings["m_C"] = m_C
        self.settings["m_sinphi"] = m_sinphi
        self.settings["top_bound"] = top_bound
        self.settings["bottom_bound"] = bottom_bound
        self.settings["left_bound"] = left_bound
        self.settings["right_bound"] = right_bound
        self.settings["m_s_xx"] = m_s_xx
        self.settings["m_s_xy"] = m_s_xy
        self.settings["m_e_xx"] = m_e_xx
        self.settings["m_e_xy"] = m_e_xy
        self.settings["m_P"] = m_P
        self.settings['moving_points_index_list'] = moving_cells_index_list
        self.settings['markers_index_list'] = [ index for index, Vx, Vy in moving_cells_index_list]

    def load_image(self, fname, moving_cells):
        image = np.load(f'{fname[:-3]}.npy')
        image_i, image_j = image.shape

        j_res = self['j_res']
        i_res = self['i_res']
        marker_density = self['pdensity']

        # markers
        mxx = []
        myy = []
        for x in range(j_res-1):
            for y in range(i_res-1):
                for _ in range(marker_density):
                    mxx.append(x+np.random.uniform(0,.5,1))
                    myy.append(y+np.random.uniform(0,.5,1))

                    mxx.append(x+np.random.uniform(0,.5,1))
                    myy.append(y+np.random.uniform(.5,1,1))

                    mxx.append(x+np.random.uniform(.5,1,1))
                    myy.append(y+np.random.uniform(.5,1,1))

                    mxx.append(x+np.random.uniform(.5,1,1))
                    myy.append(y+np.random.uniform(0,.5,1))

        mxx = np.asarray(mxx)
        myy = np.asarray(myy)

        moving_cells_index_list = []
        moving_cells_coordinates_list = [(xy) for xy, VxVy in moving_cells]

        # TODO: Refactor following block to be inside previous cascade of for loops
        mj = (mxx*image_j/(j_res-1)).astype(int)
        mi = (myy*image_i/(i_res-1)).astype(int)
        values = np.zeros(np.shape(mxx))
        for idx in range(len(mxx)):
            j,i = mj[idx], mi[idx]
            values[idx] = image[i,j]
            if (j,i) in moving_cells_coordinates_list:
                idx_ = moving_cells_coordinates_list.index((j,i))
                _, (Vx, Vy) = moving_cells[idx_]
                moving_cells_index_list.append((idx, Vx, Vy))

        if moving_cells_index_list:
            moving_cells_index_list = [random.choice(moving_cells_index_list)
                                                         for _ in range(5)]

        values = values.astype(int)

        return mxx, myy, values, moving_cells_index_list
