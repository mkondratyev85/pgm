import numpy as np
from scipy import sparse

from Stokeselvis import return_sparse_matrix_Stokes
from interpolate import interpolate, interpolate2m, interpolate_harmonic, fill_nans
from largetime import Time

import matplotlib.pyplot as plt

average = lambda x: (x[:-1,:-1] + x[1:,:-1] +x[1:,1:] +x[:-1,1:])/4.0

class Model(object):

    def __init__(self, parameters):
        defaults = {
                   'T' : Time(0),
                   'step' : 0,
                   'width' : None,
                   'height' : None,
                   'j_res' : None,
                   'i_res' : None,
                   'gx_0' : None,
                   'gy_0' : None,
                   'p0cell' : None,
                   'right_bound' : None,
                   'left_bound' : None,
                   'top_bound' : None,
                   'bottom_bound' : None,
                   #'stress_changes' : 'simple',
                   #'stress_changes' : '2nd order',
                   'stress_changes' : 'subgrid',
                   'subgrid_relaxation' : .5,
                   #'advect_scheme' : 'simple',
                   'advect_scheme' : 'Runge-Kutta 2nd order',

        }
        self.markers = {'mxx' : None,
                        'myy' : None,
                        'm_cat' : None,
                        'm_rho' : None,
                        'm_eta' : None,
                        'm_C' : None,
                        'm_mu' : None,
                        'm_sinphi' : None,
                        'm_s_xx' : None,
                        'm_s_xy' : None,
                        'm_e_xx' : None,
                        'm_e_xy' : None,
                        'm_P' : None,
                       }

        self.__dict__.update(defaults)
        self.__dict__.update(self.markers)
        self.__dict__.update(parameters)

        for key in self.__dict__:
            if self.__dict__[key] is None:
                raise ValueError (f'{key} parameter must be set')

        self.width = int(self.width)
        self.height = int(self.height)
        self.i_res = int(self.i_res)
        self.j_res = int(self.j_res)

        self.dx = self.width / (self.j_res-1)
        self.dy = self.height /( self.i_res-1)

        self.dx2, self.dy2 = self.dx**2, self.dy**2

        self.kbond   = 4*self.m_eta.min()/((self.dx+self.dy)**2)
        self.kcont   = 2*self.m_eta.min()/(self.dx+self.dy)

    def make_step(self, maxT = 1):
        dt_min=1e+10
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
        step =  self.step

        if not T : T = Time(0)
        if not step: step = -1

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
                        eta_n = interpolate_harmonic(mxx+.5,myy+.5,i_res,j_res, m_eta )
                        eta_s = interpolate_harmonic(mxx,myy,i_res,j_res, m_eta )

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

                        gx_0, gy_0 = self.gx_0, self.gy_0
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
                        dtx = 0.2 * dx / Vx_max
                        dty = 0.2 * dy / Vy_max
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

                        # ds_xx = s_xx - average(so_xx0)
                        ds_xx = s_xx - so_xx0[1:,1:]
                        ds_xy = s_xy - so_xy0

                        if not plastic_yield: break


                m_Vx = interpolate2m(mxx   , myy-.5, Vx[:-1,:])
                m_Vy = interpolate2m(mxx-.5, myy   , Vy[:,:-1])
                m_P  = interpolate2m(mxx-.5, myy-.5, P[1:,1:]) # tecnichaly, there must be +.5,+.5
                                                               # but since we slice P, indexing goes one item lower

                w = dVy_dx - dVx_dy

                m_s_xx, m_s_xy = self.interpolate_stress_changes(mxx, myy,
                                                                 m_s_xx, m_s_xy,
                                                                 s_xx, s_xy,
                                                                 ds_xx, ds_xy,
                                                                 dt)

                m_e_xx = interpolate2m(mxx-.5,myy-.5,e_xx)
                m_e_xy = interpolate2m(mxx,myy,e_xy)

                m_w    = interpolate2m(mxx-.5 ,myy-.5 , w)

                m_a = m_w * dt
                m_s_xx_ = m_s_xx - m_s_xy * 2 * m_a
                m_s_xy_ = m_s_xy + m_s_xy * 2 * m_a
                m_s_xx, m_s_xy = m_s_xx_, m_s_xy_

                self.advect(mxx, myy, m_Vx, m_Vy, Vx, Vy, dt)

                T += dt
                step +=1

                #s_xx = 2 * eta[1:,1:] * e_xx

                #d_s_xx = s_xx - average(sxx_0)

                #s_xy = 2 * eta * e_xy

                #d_s_xy = s_xy - sxy_0

                #sii = (s_xx**2 + average(s_xy)**2)**.5
                #eii = (e_xx**2 + average(e_xy)**2)**.5


                #if step % step : continue

                parameters = {'T' : T,
                              'step' : step,
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
                yield parameters
    def interpolate_stress_changes(self, mxx, myy, m_s_xx, m_s_xy, s_xx=None, s_xy=None, ds_xx=None, ds_xy=None, dt=None ):
        if self.stress_changes == 'simple':
            m_s_xx = interpolate2m(mxx-.5,myy-.5,s_xx)
            m_s_xy = interpolate2m(mxx,myy,s_xy)
        elif self.stress_changes == '2nd order':
            m_ds_xx = interpolate2m(mxx-.5,myy-.5,ds_xx)
            m_ds_xy = interpolate2m(mxx,myy,ds_xy)
            m_s_xx = m_s_xx + m_ds_xx
            m_s_xy = m_s_xy + m_ds_xy
        elif self.stress_changes == 'subgrid':
            d_ve = self.subgrid_relaxation
            m_eta = self.m_eta
            m_mu = self.m_mu
            i_res, j_res = self.i_res, self.j_res
            m_dt_maxwell = m_eta / m_mu
            d_ = (-d_ve*dt/m_dt_maxwell).astype(np.float)
            m_s_xx_nodes = interpolate2m(mxx-.5,myy-.5,s_xx)
            m_s_xy_nodes = interpolate2m(mxx,myy,s_xy)
            #m_ds_xx_subgrid = (m_s_xx_nodes - m_s_xx)*(1-np.exp(-d_ve*dt/m_dt_maxwell))
            #m_ds_xy_subgrid = (m_s_xy_nodes - m_s_xy)*(1-np.exp(-d_ve*dt/m_dt_maxwell))
            m_ds_xx_subgrid = (m_s_xx_nodes - m_s_xx)*(1-np.exp(d_))
            m_ds_xy_subgrid = (m_s_xy_nodes - m_s_xy)*(1-np.exp(d_))
            (ds_xx_subgrid,) = interpolate(mxx+.5,myy+.5,i_res,j_res, (m_ds_xx_subgrid, ))
            (ds_xy_subgrid,) = interpolate(mxx,myy,i_res,j_res, (m_ds_xy_subgrid, ))
            ds_xx_remaining = ds_xx - ds_xx_subgrid[1:,1:]
            ds_xy_remaining = ds_xy - ds_xy_subgrid
            m_ds_xx_remaining = interpolate2m(mxx-.5,myy-.5, ds_xx_remaining)
            m_ds_xy_remaining = interpolate2m(mxx,myy, ds_xy_remaining)
            m_s_xx += m_ds_xx_subgrid + m_ds_xx_remaining
            m_s_xy += m_ds_xy_subgrid + m_ds_xy_remaining
        return m_s_xx, m_s_xy

    def advect(self, mxx, myy, m_Vx, m_Vy, Vx, Vy, dt ):
        dx, dy = self.dx, self.dy
        if self.advect_scheme=='simple':
            mxx += m_Vx*dt/dx
            myy += m_Vy*dt/dy
        elif self.advect_scheme=='Runge-Kutta 2nd order':
            mxx_b = mxx + 0.5*m_Vx*dt/dx
            myy_b = myy + 0.5*m_Vy*dt/dy

            m_Vx_eff = interpolate2m(mxx_b   , myy_b-.5, Vx[:-1,:])
            m_Vy_eff = interpolate2m(mxx_b-.5, myy_b   , Vy[:,:-1])

            mxx += m_Vx_eff*dt/dx
            myy += m_Vy_eff*dt/dy
        return mxx, myy
