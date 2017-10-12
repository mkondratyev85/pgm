import numpy as np
import matplotlib.pylab as plt
from scipy import sparse
import subprocess

from Stokeselvis import return_sparse_matrix_Stokes
from interpolate import interpolate, interpolate2m, interpolate_harmonic, fill_nans

from view import Matplot
from model import Model

average = lambda x: (x[:-1,:-1] + x[1:,:-1] +x[1:,1:] +x[:-1,1:])/4.0

def load_step(filename, Step):
	data = np.load("%s/%s.npz" % (filename, Step))
	mxx = data['arr_0']
	myy = data['arr_1']
	m_cat = data['arr_2']
	m_mu = data['arr_3']
	m_eta = data['arr_4']
	m_rho = data['arr_5']
	m_C = data['arr_6']
	m_sinphi = data['arr_7']
	m_s_xx = data['arr_8']
	m_s_xy = data['arr_9']
	m_e_xx = data['arr_10']
	m_e_xy = data['arr_11']
	m_P    = data['arr_12']

	width, height, j_res, i_res, gx_0, gy_0, right_bound, left_bound, top_bound, bottom_bound =\
	                                                    np.loadtxt("%s/params.txt" % filename, unpack=True)
	if right_bound == 0:
		right_bound == "freeslip"
	else:
		right_bound = "noslip"

	if left_bound == 0:
		left_bound == "freeslip"
	else:
		left_bound = "noslip"
	if top_bound == 0:
		top_bound == "freeslip"
	else:
		top_bound = "noslip"
	if bottom_bound == 0:
		bottom_bound == "freeslip"
	else:
		bottom_bound = "noslip"

	prop = {"mxx":mxx,"myy":myy,"m_cat":m_cat,"m_eta":m_eta,"m_rho":m_rho,"m_C":m_C,"m_sinphi":m_sinphi, "m_mu":m_mu,
			"m_s_xx":m_s_xx, "m_s_xy":m_s_xy, "m_e_xx":m_e_xx, "m_e_xy":m_e_xy, "m_P":m_P,
			"right_bound":right_bound, "left_bound":left_bound, "top_bound":top_bound, "bottom_bound":bottom_bound}

	return width, height, j_res, i_res, gx_0, gy_0, right_bound, left_bound, top_bound, bottom_bound, prop

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
            steps = self.model.make_step(maxT = MaxT)
            for step in steps:
                self.view.plot12(step)


    def run(self, maxT, step, filename, dt_min=1e+10):
            self.figname = filename
            mxx = self.mxx
            myy = self.myy
            m_cat = self.m_cat
            m_rho = self.m_rho
            m_eta = self.m_eta
            m_mu = self.m_mu
            m_C = self.m_C
            m_sinphi = self.m_sinphi
            m_s_xx = self.m_s_xx
            m_s_xy =  self.m_s_xy
            m_e_xx =  self.m_e_xx
            m_e_xy =  self.m_e_xy
            m_P    =  self.m_P

            m_s_ii_old = np.zeros(np.shape(mxx))
            m_e_ii_old = np.zeros(np.shape(mxx))
            
            i_res, j_res = self.i_res, self.j_res
            dx, dy = self.dx, self.dy
            gx_0, gy_0 = self.gx_0, self.gy_0
            kbond, kcont = self.kbond, self.kcont
            p0cell = self.p0cell

            eta_min = 1e+10
            eta_max = 1e+34

            T = self.T
            Step =  self.Step

            if self.right_bound == "freeslip": 
                    right_bound = 0
            else:
                    right_bound = 1
            if self.left_bound == "freeslip": 
                    left_bound = 0
            else:
                    left_bound = 1
            if self.top_bound == "freeslip": 
                    top_bound = 0
            else:
                    top_bound = 1
            if self.bottom_bound == "freeslip": 
                    bottom_bound = 0
            else:
                    bottom_bound = 1

            if not T : T = 0
            if not Step: Step = -1

            np.savetxt("%s/params.txt" % self.figname,
                                    np.vstack((self.width, self.height, j_res, i_res, gx_0, gy_0,
                                    right_bound, left_bound, top_bound,bottom_bound)).T)

            dt = 0
            plastic_iterations = 1
            while T < maxT:
                    dt = max(dt, dt_min)
                    for i in range(plastic_iterations):
                            plastic_yield = False

                            m_xelvis = m_eta/(m_mu*dt + m_eta)
                            m_s_xx_new = m_s_xx*m_xelvis + 2*m_eta*m_e_xx*(1-m_xelvis)
                            m_s_xy_new = m_s_xy*m_xelvis + 2*m_eta*m_e_xy*(1-m_xelvis)

                            m_s_ii_new = (m_s_xx_new**2 + m_s_xy_new**2)**.5
                            m_s_ii_yield = m_C + m_sinphi*m_P
                            mask = m_s_ii_yield <0
                            m_s_ii_yield[mask] = 0

                            ym = m_s_ii_yield > m_s_ii_new # yielding mask
                            check_for_plastic = False
                            #print("step")
                            if check_for_plastic and np.all(~ym):
                            #ym = m_s_ii_yield < m_s_ii_new # yielding mask
                            #if not np.all(ym):
                                    plastic_yield = True
                                    print("plastic yield!")

                                    m_e_ii_old[ym] = np.sqrt(m_e_xx[ym]**2 + m_e_xy[ym]**2)
                                    m_eta[ym]      = m_s_ii_yield[ym] /2/m_e_ii_old[ym]
                                    m_s_ii_old[ym] = np.sqrt(m_s_xx[ym]**2 + m_s_xy[ym]**2)
                                    m_s_xx[ym]     = m_s_xx[ym]*m_s_ii_yield[ym]/m_s_ii_old[ym]
                                    m_s_xy[ym]     = m_s_xy[ym]*m_s_ii_yield[ym]/m_s_ii_old[ym]

                            # eta_min < eta < eta_max
                            m_eta[m_eta<eta_min] = eta_min
                            m_eta[m_eta>eta_max] = eta_max


                            # we should interpolate eta_n separately but actually eta_n and eta_s are equal
                            eta_s, rho, so_xy = interpolate(mxx,myy,i_res,j_res, (m_eta, m_rho, m_s_xy))
                            eta_n, so_xx = interpolate(mxx+.5,myy+.5,i_res,j_res, (m_eta, m_s_xx))
                            mu_s = interpolate_harmonic(mxx,myy,i_res,j_res, m_mu )
                            mu_n = interpolate_harmonic(mxx+.5,myy+.5,i_res,j_res, m_mu )

                            #Check if we have nans
                            if np.isnan(eta_s).any(): fill_nans(eta_s)
                            if np.isnan(eta_n).any(): fill_nans(eta_n)
                            if np.isnan(rho).any(): fill_nans(rho)
                            if np.isnan(mu_s).any(): fill_nans(mu_s)
                            if np.isnan(mu_n).any(): fill_nans(mu_n)
                            if np.isnan(so_xx).any(): fill_nans(so_xx)
                            if np.isnan(so_xy).any(): fill_nans(so_xy)

                            # compute viscoelastic (numerical) viscosity and stress
                            eta_s0 = eta_s
                            eta_n0 = eta_n
                            so_xy0 = so_xy
                            so_xx0 = so_xx
                            xelvis_s = eta_s/(eta_s+dt*mu_s)
                            xelvis_n = eta_n/(eta_n+dt*mu_n)
                            eta_s = eta_s*(1-xelvis_s)
                            eta_n = eta_n*(1-xelvis_n)
                            so_xy = so_xy*xelvis_s
                            so_xx = so_xx*xelvis_n

                            Stokes_sparse, vector = return_sparse_matrix_Stokes(j_res, i_res, dx, dy, 
                                            eta_s, eta_n, rho, gx_0, gy_0, so_xx, so_xy, kbond, kcont, p0cell, 
                                            lower_boundary=self.bottom_bound, upper_boundary=self.top_bound,
                                            right_boundary=self.right_bound, left_boundary=self.left_bound)

                            Stokes_solve = sparse.linalg.spsolve(Stokes_sparse, vector)
                            P  = Stokes_solve[::3].reshape((i_res),(j_res))
                            Vx = Stokes_solve[1::3].reshape((i_res),(j_res))
                            Vy = Stokes_solve[2::3].reshape((i_res),(j_res))

                            P *= kcont

                            Vx_max = np.abs(Vx).max()
                            Vy_max = np.abs(Vy).max()
                            dtx = 0.1 * dx / Vx_max 
                            dty = 0.1 * dy / Vy_max 
                            dt = min(dtx,dty)

                            #dVx, dVy   = Vx[1:,  :-1] - Vx[:-1, :-1], Vy[ :-1, 1:] - Vy[:-1, :-1]
                            #dVx_, dVy_ = Vx[ :, 1:-1] - Vx[:,   :-2], Vy[1:-1,  :] - Vy[:-2, :]
                            dVx_dx = (Vx[:-1,1:] - Vx[:-1,:-1])/dx
                            dVy_dy = (Vy[1:,:-1] - Vy[:-1,:-1])/dy

                            #dVx_dy = (Vx[:-1,1:] - Vx[:-1,:-1])/dy
                            #dVy_dx = (Vy[1:,:-1] - Vy[:-1,:-1])/dx
                            
                            dVx_dy = (Vx[1:-1,:] - Vx[:-2,:])/dy
                            dVy_dx = (Vy[:,1:-1] - Vy[:,:-2])/dx

                            #concatenating array to add additional rows and columns. Just copy remaning columns
                            #        abc    ad    aadd
                            # abc -> abc    be -> bbee
                            # def    def    cf    ccff
                            #        def
                            dVx_dy = np.append(dVx_dy[0,:][None,:],dVx_dy,axis=0)
                            dVx_dy = np.append(dVx_dy,dVx_dy[-1,:][None,:],axis=0)
                            dVy_dx = np.append(dVy_dx[:,0][:,None],dVy_dx,axis=1)
                            dVy_dx = np.append(dVy_dx,dVy_dx[:,-1][:,None],axis=1)

                            e_xx = dVx_dx # strain rate
                            e_xy = .5 * (dVx_dy + dVy_dx)
                            s_xx = (1-xelvis_n[1:,1:])*2*eta_n0[1:,1:]*e_xx + xelvis_n[1:,1:]*so_xx0[1:,1:]
                            s_xy = (1-xelvis_s)*2*eta_s0*e_xy + xelvis_s*so_xy0
                            s_ii = (s_xx**2 + average(s_xy**2))**.5

                            d_sxx = s_xx - average(so_xx0)
                            d_sxy = s_xy - so_xy0

                            if not plastic_yield: break

                    m_Vx = interpolate2m(mxx   , myy-.5, Vx[:-1,:])
                    m_Vy = interpolate2m(mxx-.5, myy   , Vy[:,:-1])
                    m_P  = interpolate2m(mxx-.5, myy-.5, P[1:,1:]) # tecnichaly, there must be +.5,+.5 
                                                                   # but since we slice P, indexing goes one item lower

                    w = dVy_dx - dVx_dy

                    m_s_xx = interpolate2m(mxx-.5,myy-.5,s_xx)
                    m_s_xy = interpolate2m(mxx,myy,s_xy)
                    #m_ds_xx = interpolate2m(mxx-.5,myy-.5,d_sxx)
                    #m_ds_xy = interpolate2m(mxx,myy,d_sxy)
                    #m_s_xx = m_s_xx + m_ds_xx
                    #m_s_xy = m_s_xy + m_ds_xy

                    m_e_xx = interpolate2m(mxx-.5,myy-.5,e_xx)
                    m_e_xy = interpolate2m(mxx,myy,e_xy)

                    m_w    = interpolate2m(mxx-.5 ,myy-.5 , w)
                    
                    m_a = m_w * dt
                    m_s_xx_ = m_s_xx - m_s_xy * 2 * m_a
                    m_s_xy_ = m_s_xy + m_s_xy * 2 * m_a
                    m_s_xx, m_s_xy = m_s_xx_, m_s_xy_

                    mxx += m_Vx*dt/dx
                    myy += m_Vy*dt/dy

                    T += dt
                    Step +=1

                    #s_xx = 2 * eta[1:,1:] * e_xx

                    #d_s_xx = s_xx - average(sxx_0)

                    #s_xy = 2 * eta * e_xy

                    #d_s_xy = s_xy - sxy_0

                    #sii = (s_xx**2 + average(s_xy)**2)**.5
                    #eii = (e_xx**2 + average(e_xy)**2)**.5


                    if Step % step : continue

                    parameters = {'T' : T,
                                  'step' : Step,
                                  'eta_s' : eta_s,
                                  'eta_n' : eta_n,
                                  'mxx' : mxx,
                                  'myy' : myy,
                                  'm_cat' : m_cat,
                                  'sii' : s_ii,
                                  'P' : P,
                                  'Vx' : Vx,
                                  'Vy' : Vy,
                                  'e_xx' : e_xx,
                                  'e_xy' : e_xy,
                                  's_xx' : s_xx,
                                  's_xy' : s_xy,
                                  'xelvis_s' : xelvis_s,
                                  'mu_n' : mu_n,
                                  'mu_s' : mu_s,
                                  'w' : w
                                 }

                    self.view.plot12(parameters)
                    self.save(Step, mxx, myy, m_cat, m_mu, m_eta, m_rho, m_C, m_sinphi, m_s_xx, m_s_xy, m_e_xx, m_e_xy, m_P)


    def save(self, Step, mxx, myy, m_cat, m_mu, m_eta, m_rho, m_C, m_sinphi, m_s_xx, m_s_xy, m_e_xx, m_e_xy, m_P):
        Myr = lambda t: t/(365.25*24*3600*10**6) # Convert seconds to millions of year
        np.savez("%s/%s.npz" % (self.figname, Step), mxx, myy, m_cat, m_mu, m_eta, m_rho, m_C, m_sinphi, m_s_xx, m_s_xy, m_e_xx, m_e_xy, m_P)
