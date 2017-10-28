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
