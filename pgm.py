import numpy as np
import matplotlib.pylab as plt
from scipy import sparse
from Stokes import return_sparse_matrix_Stokes
from interpolate import interpolate, interpolate2m

from test_mmodel import load_model_simple

class PGM:
	def __init__(self, (width, height), (j_res, i_res), (gx_0, gy_0), (mxx, myy, m_cat, m_rho, m_eta)):
		self.height, self.width  = width, height
		self.j_res, self.j_res = j_res, i_res
		self.dx = width / (j_res-1)
		self.dy = height/ (i_res-1)
		self.dx2, self.dy2 = dx**2, dy**2

		self.p0cell  =  0 # Pressure condition in one cell (i==2 && j==3)

		# define grid for messing with indexes
		self.j      = np.linspace(0,j_res-1,j_res).astype('int')
		self.i      = np.linspace(0,i_res-1,i_res).astype('int')
		self.jj,self.ii  = np.meshgrid(j,i)
		
		self.mxx = mxx
		self.myy = myy
		self.m_cat = m_cat
		self.m_rho = m_rho
		self.m_eta = m_eta

		self.kbond   = 4*m_eta.min()/((dx+dy)**2)
		self.kcont   = 2*m_eta.min()/(dx+dy)

	def __run__(self, maxT, step, filename):
		T = 0
		Step = -1

		while T < maxT:
			# we should interpolate eta_n separately but actually eta_n and eta_s are equal
			eta_s, eta_n, rho = interpolate(mxx,myy,i_res,j_res,(m_eta, m_eta, m_rho))

			Stokes_sparse, vector = return_sparse_matrix_Stokes(j_res, i_res, dx, dy, eta_s, eta_n, rho, gy_0, kbond, kcont, p0cell)

			Stokes_solve = sparse.linalg.spsolve(Stokes_sparse, vector)
			P  = Stokes_solve[::3].reshape((i_res),(j_res))
			Vx = Stokes_solve[1::3].reshape((i_res),(j_res))
			Vy = Stokes_solve[2::3].reshape((i_res),(j_res))

			P = P*kcont

			Vx_max = np.abs(Vx).max()
			Vy_max = np.abs(Vy).max()
			dtx = 0.5 * dx / Vx_max 
			dty = 0.5 * dy / Vy_max 
			dt = min(dtx,dty)

			m_Vx = interpolate2m(mxx   , myy-.5, Vx[:-1,:])
			m_Vy = interpolate2m(mxx-.5, myy   , Vy[:,:-1])
			m_P  = interpolate2m(mxx-.5, myy-.5, P[1:,1:]) # tecnichaly, there must be +.5,+.5 but since we slice P, indexing goes one item lower

			eta_s, eta_n, rho = interpolate(mxx,myy,i_res,j_res,(m_eta, m_eta, m_rho)) # we should interpolate eta_n separately but actually eta_n and eta_s are equal

			#Check if we have nans
			if np.isnan(rho).any(): 
				print("Nan detected! Filling it using interpolation")
				mask = np.isnan(rho)
				rho[mask] = np.interp(np.flatnonzero(mask),np.flatnonzero(~mask),rho[~mask])
				eta_n[mask] = np.interp(np.flatnonzero(mask),np.flatnonzero(~mask),eta_n[~mask])
				eta_s[mask] = np.interp(np.flatnonzero(mask),np.flatnonzero(~mask),eta_s[~mask]) 

			mxx += m_Vx*dt/dx
			myy += m_Vy*dt/dy

			T += dt
			Step +=1

			if Step % step : continue

			dVx, dVy   = Vx[1:,  :-1] - Vx[:-1, :-1], Vy[ :-1, 1:] - Vy[:-1, :-1]
			dVx_, dVy_ = Vx[ :, 1:-1] - Vx[:,   :-2], Vy[1:-1,  :] - Vy[:-2, :]

			e_xx = dVx/dx # strain rate
			s_xx = 2 * eta_n[1:,1:] * e_xx 

			e_xy_ = (dVx_[1:-1,:]/dy + dVy_[:,1:-1]/dx) / 2
			s_xy_ = 2 * eta_s[1:-1,1:-1] * e_xy_

			s_xx_ = (s_xx[:-1,:-1] + s_xx[:-1,1:] + s_xx[1:,1:] + s_xx[1:,:-1]) / 4

			sii = (s_xx_**2 + s_xy_**2)**.5

			Vx_average = 0.5*(Vx[1:-1,:-2]+Vx[:-2,:-2])
			Vy_average = 0.5*(Vy[ :-2,1:-1]+Vy[:-2,:-2])
			
			self.plot(T, eta_n, mxx, myy, sii, P, Vx_average, Vy_average)

	def plot(self):
		Myr = lambda t: t/(365.25*24*3600*10**6) # Convert seconds to millions of year

		plt.clf()
		fig = plt.figure(figsize=(15,10))

		plt.suptitle("Model size: %s km x %s km (%s x %s cells, dx=%s km, dy=%s km). Current Time: %07.3f Myr. Step %s" %
					(self.width/1000, self.height/1000, self.j_res, self.i_res, self.dx/1000, self.dy/1000, Myr(T), Step))
		plt.subplot(2,2,1)
		plt.title("Viscosity")
		plt.imshow(eta_n,interpolation='none',cmap='copper')
		plt.colorbar()
		
		plt.subplot(2,2,2)
		plt.scatter(mxx,myy,c=m_cat,s=1,edgecolors='face',cmap='copper')
		plt.colorbar()
		plt.ylim([i_res-1,0])
		plt.xlim([0,j_res-1])
		
		plt.subplot(2,2,3)
		plt.title("Sigma II")
		plt.imshow(sii[:-1,:-1],interpolation='none',norm=LogNorm())
		plt.colorbar()

		plt.subplot(2,2,4)
		plt.title("P")
		plt.imshow(P[1:,1:],interpolation='none')
		plt.colorbar()
		plt.streamplot(jj[:-2,:-2],ii[:-2,:-2],Vx_average,Vy_average,color='white')
		plt.ylim([i_res-2,0])
		plt.xlim([0,j_res-2])
		
		plt.savefig('%s/%08.4f.png' % (self.figname, Myr(T)))
		plt.close(fig)

