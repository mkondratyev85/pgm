import numpy as np
import matplotlib.pylab as plt
from scipy import sparse
from Stokes import return_sparse_matrix_Stokes
from interpolate import interpolate, interpolate2m
import subprocess

def load_step(filename, Step):
	data = np.load("%s/%s.npz" % (filename, Step))
	mxx = data['arr_0']
	myy = data['arr_1']
	m_cat = data['arr_2']
	m_eta = data['arr_3']
	m_rho = data['arr_4']

	width, height, j_res, i_res, gx_0, gy_0 = np.loadtxt("%s/params.txt" % filename, unpack=True)

	return width, height, j_res, i_res, gx_0, gy_0, mxx, myy, m_cat, m_rho, m_eta

class PGM:
	def __init__(self, width, height, j_res, i_res, gx_0, gy_0, mxx, myy, m_cat, m_rho, m_eta, T = None, Step = None, p0cell=0):
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
		
		self.mxx = mxx
		self.myy = myy
		self.m_cat = m_cat
		self.m_rho = m_rho
		self.m_eta = m_eta

		self.T = T
		self.Step = T

		self.kbond   = 4*self.m_eta.min()/((self.dx+self.dy)**2)
		self.kcont   = 2*self.m_eta.min()/(self.dx+self.dy)
		self.label = subprocess.check_output(["git", "describe", "--always"])[:-1].decode('utf-8')

	def run(self, maxT, step, filename):
		mxx = self.mxx 
		myy = self.myy 
		m_cat = self.m_cat
		m_rho = self.m_rho
		m_eta = self.m_eta
		
		i_res, j_res = self.i_res, self.j_res
		dx, dy = self.dx, self.dy
		gx_0, gy_0 = self.gx_0, self.gy_0
		kbond, kcont = self.kbond, self.kcont
		p0cell = self.p0cell

		self.figname = filename

		T = self.T
		Step =  self.Step

		if not T : T = 0
		if not Step: Step = -1

		np.savetxt("%s/params.txt" % self.figname, np.vstack((self.width, self.height, j_res, i_res, gx_0, gy_0)).T)

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
			e_xx_ = (e_xx[:-1,:-1] + e_xx[:-1,1:] + e_xx[1:,1:] + e_xx[1:,:-1]) / 4

			sii = (s_xx_**2 + s_xy_**2)**.5

			Vx_average = 0.5*(Vx[1:-1,:-2]+Vx[:-2,:-2])
			Vy_average = 0.5*(Vy[ :-2,1:-1]+Vy[:-2,:-2])
			
			self.plot(T, Step, eta_n, mxx, myy, m_cat, sii, P, Vx_average, Vy_average)
			self.save(Step, mxx, myy, m_cat, m_eta, m_rho)

	def plot(self,T, Step, eta_n, mxx, myy, m_cat, sii, P, Vx_average, Vy_average):
		Myr = lambda t: t/(365.25*24*3600*10**6) # Convert seconds to millions of year

		plt.clf()
		fig = plt.figure(figsize=(15,10))

		plt.suptitle("Model size: %s km x %s km (%s x %s cells, dx=%s km, dy=%s km). Current Time: %07.3f Myr. Step %s git verstion: %s" %
					(self.width/1000, self.height/1000, self.j_res, self.i_res, self.dx/1000, self.dy/1000, Myr(T), Step, self.label))
		plt.subplot(2,2,1)
		plt.title("Viscosity")
		plt.imshow(eta_n,interpolation='none',cmap='copper')
		plt.colorbar()
		
		plt.subplot(2,2,2)
		plt.scatter(mxx,myy,c=m_cat,s=1,edgecolors='face',cmap='copper')
		plt.colorbar()
		plt.ylim([self.i_res-1,0])
		plt.xlim([0,self.j_res-1])
		
		plt.subplot(2,2,3)
		plt.title("Sigma II")
		plt.imshow(sii[:-1,:-1],interpolation='none')
		plt.colorbar()

		plt.subplot(2,2,4)
		plt.title("P")
		plt.imshow(P[1:,1:],interpolation='none')
		plt.colorbar()
		plt.streamplot(self.jj[:-2,:-2],self.ii[:-2,:-2],Vx_average,Vy_average,color='white')
		plt.ylim([self.i_res-2,0])
		plt.xlim([0,self.j_res-2])

		plt.savefig('%s/%08.4f.png' % (self.figname, Myr(T)))
		plt.close(fig)

	
	def save(self, Step, mxx, myy, m_cat, m_eta, m_rho):
		Myr = lambda t: t/(365.25*24*3600*10**6) # Convert seconds to millions of year
		np.savez("%s/%s.npz" % (self.figname, Step), mxx, myy, m_cat, m_eta, m_rho)
