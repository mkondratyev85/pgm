import numpy as np
import pickle
import matplotlib.pylab as plt
from scipy import sparse
import subprocess

from view import Matplot
from model import Model

import random

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
                if iteration['step']>50:
                    self.model.gx_0 = 0
                    self.model.gy_0 = 0
                # if iteration['step'] % 5:
                #     continue
                self.view.plot12(iteration)


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

            # # add special moving particle
            # x_ = round((image_j/j_res)*x - .5)
            # y_ = round((image_i/i_res)*y - .5)
            # # x_ = round(x_)
            # # y_ = round(y_)
            # # print(x_, y_)
            #
            # if (x_,y_) in moving_cells_coordinates_list:
            #     index = moving_cells_coordinates_list.index((x_,y_))
            #     _, (Vx, Vy) = moving_cells[index]
            #     mxx.append(np.asarray([x-.5]))
            #     myy.append(np.asarray([y-.5]))
            #     # print (mxx[-2:-1])
            #     # print (myy[-2:-1])
            #     index = len(mxx) -1
            #     moving_cells_index_list.append((index, Vx, Vy))
            #     # print(index)
    # plt.scatter(mxx,myy,s=1)
    # plt.show()

    # def get_points_around():
    #     points = []
    #     for point_x, point_y in zip(mxx,myy):
    #         point_x_ = int((point_x[0]))
    #         point_y_ = int((point_y[0]))
    #         if (point_x_, point_y_) == (x,y):
    #             points.append((np.array(point_x),np.array(point_y)))
    #     return points
    #
    def plotVelocities():
        Vx_, Vy_ = np.zeros((j_res, i_res)), np.zeros((j_res, i_res))
        for index, Vx, Vy in moving_cells_index_list:
            x,y = mxx[index][0], myy[index][0]
            i,j = int(y), int(x)
            Vx_[i,j] = 1
            Vy_[i,j] = 1

        plt.subplot(1,2,1)
        plt.imshow(Vx_)
        plt.subplot(1,2,2)
        plt.imshow(Vy_)
        plt.show()


    # for (x,y),(Vx,Vy) in moving_cells:
    #     points = get_points_around()#(mxx, myy), (x,y), (j_res, i_res))
    #     point = random.choice(points)
    #     points_ = list(zip(mxx,myy))
    #     index = points_.index(point)
    #     moving_cells_index_list.append((index, Vx, Vy))


        # x_ = ((x+.5) / image_j) * (j_res)
        # y_ = ((y+.5) / image_i) * (i_res)
        # print (x, image_j, j_res, x_)
        # mxx.append(np.asarray([x_]))
        # myy.append(np.asarray([y_]))
        # index = len(mxx) -1
        # moving_cells_index_list.append((index, Vx, Vy))

    # plotVelocities()

    # print (moving_cells_index_list)



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

    moving_cells_index_list = [ random.choice(moving_cells_index_list) for _ in range(10)]

    # plotVelocities()

    values = values.astype(int)


    return mxx, myy, values, moving_cells_index_list

def load_model_properties(fname):
    with open(f'{fname}.pickle', 'rb') as f:
        materials = pickle.load(f)
        boundaries = pickle.load(f)
        moving_cells = pickle.load(f)
    return materials, boundaries, moving_cells

def load_model(fname, i_res, j_res, pdensity):
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

    print(boundaries)
    top_bound = boundaries['top_bound']
    bottom_bound = boundaries['bottom_bound']
    left_bound = boundaries['left_bound']
    right_bound = boundaries['right_bound']

    model_prop = {
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

    return model_prop
