import sys, os, importlib
import subprocess
import pickle
import random

import numpy as np
import matplotlib.pylab as plt
from scipy import sparse

from view import Matplot
from model import Model
from largetime import Time


average = lambda x: (x[:-1,:-1] + x[1:,:-1] +x[1:,1:] +x[:-1,1:])/4.0

class PGM:
    def __init__(self, model_prop, T = None, Step = None, figname=''):
        self.height, self.width  = int(model_prop['width']), int(model_prop['height'])
        self.j_res, self.i_res = int(model_prop['j_res']), int(model_prop['i_res'])
        self.dx = self.width / (self.j_res-1)
        self.dy = self.height /( self.i_res-1)
        self.dx2, self.dy2 = self.dx**2, self.dy**2

        self.gx_0, self.gy_0 = model_prop['gx_0'], model_prop['gy_0']
        self.p0cell  =  model_prop['p0cell'] # Pressure condition in one cell (i==1 && j==2)

        # define grid for messing with indexes
        self.j      = np.linspace(0,self.j_res-1,self.j_res).astype('int')
        self.i      = np.linspace(0,self.i_res-1,self.i_res).astype('int')
        self.jj,self.ii  = np.meshgrid(self.j,self.i)

        self.mxx = model_prop["mxx"]
        self.myy = model_prop["myy"]
        self.m_cat = model_prop["m_cat"]
        self.m_rho = model_prop["m_rho"]
        self.m_eta = model_prop["m_eta"]
        self.m_C = model_prop["m_C"]
        self.m_mu = model_prop["m_mu"]
        self.m_sinphi = model_prop["m_sinphi"]
        self.m_s_xx = model_prop["m_s_xx"]
        self.m_s_xy = model_prop["m_s_xy"]
        self.m_e_xx = model_prop["m_e_xx"]
        self.m_e_xy = model_prop["m_e_xy"]
        self.m_P    = model_prop["m_P"]
        self.right_bound = model_prop["right_bound"]
        self.left_bound = model_prop["left_bound"]
        self.top_bound = model_prop["top_bound"]
        self.bottom_bound = model_prop["bottom_bound"]

        self.T = T
        self.Step = T

        self.kbond   = 4*self.m_eta.min()/((self.dx+self.dy)**2)
        self.kcont   = 2*self.m_eta.min()/(self.dx+self.dy)
        self.label = subprocess.check_output(["git", "describe", "--always"])[:-1].decode('utf-8')

        self.figname=figname
        self.view = Matplot(width = self.width,
                            height = self.height,
                            i_res = self.i_res,
                            j_res = self.j_res,
                            dx = self.dx,
                            dy = self.dy,
                            git = self.label,
                            figname = self.figname )

        self.model = Model(model_prop)
        self.maxT = Time(model_prop['MaxT'])

    def run(self, MaxT = None):
        while True:
            for iteration in self.model.make_step(maxT = self.maxT):
                print(iteration['step'])
                if iteration['step']>50:
                    self.model.gx_0 = 0
                    self.model.gy_0 = 0
                # if iteration['step'] % 5:
                #     continue
                self.view.plot12(iteration)

def load_settings(fname):
    sys.path.append(os.path.dirname(fname))
    mname = os.path.splitext(os.path.basename(fname))[0]
    imported = importlib.import_module(mname)
    sys.path.pop()
    return imported.config

def load_image(fname, i_res, j_res, marker_density, moving_cells):
    image = np.load(f'{fname}.npy')
    image_i, image_j = image.shape

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
                                                     for _ in range(10)]

    values = values.astype(int)

    return mxx, myy, values, moving_cells_index_list

def load_model_properties(fname):
    with open(f'{fname}.pickle', 'rb') as f:
        materials = pickle.load(f)
        boundaries = pickle.load(f)
        moving_cells = pickle.load(f)
    return materials, boundaries, moving_cells

def load_model(fname, i_res=None, j_res=None, pdensity=None):
    model_prop = load_settings(fname)
    i_res, j_res = model_prop['i_res'], model_prop['j_res']
    pdensity = model_prop['pdensity']

    materials, boundaries, moving_cells  = load_model_properties(fname)
    mxx, myy, values, moving_cells_index_list = load_image(fname, i_res, j_res, pdensity, moving_cells)

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

    model_prop2 = {
                  "mxx": mxx,
                  "myy": myy,
                  "m_cat": m_cat,
                  "m_rho": m_rho,
                  "m_eta": m_eta,
                  "m_mu": m_mu,
                  "m_C": m_C,
                  "m_sinphi": m_sinphi,

                  "top_bound": top_bound,
                  "bottom_bound": bottom_bound,
                  "left_bound": left_bound,
                  "right_bound": right_bound,

                  "m_s_xx": m_s_xx,
                  "m_s_xy": m_s_xy,
                  "m_e_xx": m_e_xx,
                  "m_e_xy": m_e_xy,
                  "m_P": m_P,

                  'moving_points_index_list' : moving_cells_index_list,
                  'markers_index_list' : [ index for index, Vx, Vy in moving_cells_index_list]
                  }
    model_prop.update(model_prop2)

    return model_prop
