import numpy as np
import matplotlib.pylab as plt
from scipy import sparse
import subprocess

from view import Matplot
from model import Model

average = lambda x: (x[:-1,:-1] + x[1:,:-1] +x[1:,1:] +x[:-1,1:])/4.0

class PGM:
    def __init__(self, width, height, j_res, i_res, gx_0, gy_0, model_prop, T = None, Step = None, p0cell=0,figname=''):
        self.height, self.width  = int(width), int(height)
        self.j_res, self.i_res = int(j_res), int(i_res)
        self.dx = self.width / (self.j_res-1)
        self.dy = self.height /( self.i_res-1)
        self.dx2, self.dy2 = self.dx**2, self.dy**2

        self.gx_0, self.gy_0 = gx_0, gy_0
        self.p0cell  =  p0cell # Pressure condition in one cell (i==1 && j==2)

        # define grid for messing with indexes
        self.j      = np.linspace(0,j_res-1,j_res).astype('int')
        self.i      = np.linspace(0,i_res-1,i_res).astype('int')
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
                            i_res = i_res,
                            j_res = j_res,
                            dx = self.dx,
                            dy = self.dy,
                            git = self.label,
                            figname = self.figname )

    def init_(self, parameters):
        self.model = Model(parameters)

    def run_(self, MaxT = None):
        while True:
            for iteration in self.model.make_step(maxT = MaxT):
                print(iteration['step'])
                if iteration['step'] % 10:
                    continue
                self.view.plot12(iteration)
