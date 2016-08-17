import numpy as np
import matplotlib.pylab as plt
from scipy import sparse
#from Stokes import return_sparse_matrix_Stokes
from Stokeselvis_new import return_sparse_matrix_Stokes
from interpolate import interpolate, interpolate2m
import subprocess

average = lambda x: (x[:-1,:-1] + x[1:,:-1] +x[1:,1:] +x[:-1,1:])/4.0

def load_step(filename, Step):
	data = np.load("%s/%s.npz" % (filename, Step))
	mxx = data['arr_0']
	myy = data['arr_1']
	m_cat = data['arr_2']
	m_eta = data['arr_3']
	m_rho = data['arr_4']
	m_C = data['arr_5']
	m_sinphi = data['arr_6']


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

	prop = {"mxx":mxx,"myy":myy,"m_cat":m_cat,"m_eta":m_eta,"m_rho":m_rho,"m_C":m_C,"m_sinphi":m_sinphi,
			"right_bound":right_bound, "left_bound":left_bound, "top_bound":top_bound, "bottom_bound":bottom_bound}

	return width, height, j_res, i_res, gx_0, gy_0, right_bound, left_bound, top_bound, bottom_bound, prop

class PGM:
	#def __init__(self, width, height, j_res, i_res, gx_0, gy_0, mxx, myy, m_cat, m_rho, m_eta, T = None, Step = None, p0cell=0):
	def __init__(self, width, height, j_res, i_res, gx_0, gy_0, model_prop, T = None, Step = None, p0cell=0):
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
		self.m_sinphi = model_prop["m_sinphi"]
		self.right_bound = model_prop["right_bound"]
		self.left_bound = model_prop["left_bound"]
		self.top_bound = model_prop["top_bound"]
		self.bottom_bound = model_prop["bottom_bound"]

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
		m_C = self.m_C
		m_sinphi = self.m_sinphi
		m_sxx = np.zeros(np.shape(mxx))
		m_sxy = np.zeros(np.shape(mxx))
		
		i_res, j_res = self.i_res, self.j_res
		dx, dy = self.dx, self.dy
		gx_0, gy_0 = self.gx_0, self.gy_0
		kbond, kcont = self.kbond, self.kcont
		p0cell = self.p0cell

		self.figname = filename

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

		print(self.left_bound)
		np.savetxt("%s/params.txt" % self.figname,
				        np.vstack((self.width, self.height, j_res, i_res, gx_0, gy_0,
				        right_bound, left_bound, top_bound,bottom_bound)).T)

		while T < maxT:
			# we should interpolate eta_n separately but actually eta_n and eta_s are equal
			eta_s, eta_n, rho, sxx_old, sxy_old = interpolate(mxx,myy,i_res,j_res,(m_eta, m_eta, m_rho, 
				                                              m_sxx, m_sxy))


			Stokes_sparse, vector = return_sparse_matrix_Stokes(j_res, i_res, dx, dy, eta_s, eta_n, rho, gx_0, gy_0, sxx_old, sxy_old, kbond, kcont, p0cell, lower_boundary=self.bottom_bound, upper_boundary=self.top_bound,right_boundary=self.right_bound, left_boundary=self.left_bound)
			#Stokes_sparse, vector = return_sparse_matrix_Stokes(j_res, i_res, dx, dy, eta_s, eta_n, rho, gx_0, gy_0, kbond, kcont, p0cell, lower_boundary=self.bottom_bound, upper_boundary=self.top_bound,right_boundary=self.right_bound, left_boundary=self.left_bound)

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

			eta, rho = interpolate(mxx,myy,i_res,j_res,(m_eta, m_rho)) # we should interpolate eta_n separately but actually eta_n and eta_s are equal

			#Check if we have nans
			if np.isnan(rho).any(): 
				print("Nan detected! Filling it using interpolation")
				mask = np.isnan(rho)
				rho[mask] = np.interp(np.flatnonzero(mask),np.flatnonzero(~mask),rho[~mask])
				eta[mask] = np.interp(np.flatnonzero(mask),np.flatnonzero(~mask),eta[~mask])

			#dVx, dVy   = Vx[1:,  :-1] - Vx[:-1, :-1], Vy[ :-1, 1:] - Vy[:-1, :-1]
			#dVx_, dVy_ = Vx[ :, 1:-1] - Vx[:,   :-2], Vy[1:-1,  :] - Vy[:-2, :]
			dVx_dx = (Vx[:-1,1:] - Vx[:-1,:-1])/dx
			dVy_dy = (Vy[1:,:-1] - Vy[:-1,:-1])/dy

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
			s_xx = 2 * eta[1:,1:] * e_xx

			d_s_xx = s_xx - average(sxx_0)

			e_xy = .5 * (dVx_dy + dVy_dx)
			s_xy = 2 * eta * e_xy

			d_s_xy = s_xy - sxy_0

			sii = (s_xx**2 + average(s_xy)**2)**.5
			eii = (e_xx**2 + average(e_xy)**2)**.5

			w = dVy_dx - dVx_dy

			#e_xx = dVx/dx # strain rate
			#s_xx = 2 * eta_n[1:,1:] * e_xx 

			#e_xy_ = (dVx_[1:-1,:]/dy + dVy_[:,1:-1]/dx) / 2
			#s_xy_ = 2 * eta_s[1:-1,1:-1] * e_xy_

			#s_xx_ = (s_xx[:-1,:-1] + s_xx[:-1,1:] + s_xx[1:,1:] + s_xx[1:,:-1]) / 4
			#e_xx_ = (e_xx[:-1,:-1] + e_xx[:-1,1:] + e_xx[1:,1:] + e_xx[1:,:-1]) / 4

			#sii = (s_xx_**2 + s_xy_**2)**.5

			print (np.shape(s_xx),s_xy_.shape)
			m_s_xx = interpolate2m(mxx-.5,myy-.5,s_xx)
			m_s_xy = interpolate2m(mxx,myy,s_xy_)
			
			m_w    = interpolate2m(mxx-.5 ,myy-.5 , w)
			
			m_a = m_w * dt
			m_s_xx_ = m_s_xx - m_s_xy * 2 * m_a
			m_s_xy_ = m_s_xy + m_s_xy * 2 * m_a
			m_s_xx, m_s_xy = m_s_xx_, m_s_xy_

			Vx_average = 0.5*(Vx[1:-1,:-2]+Vx[:-2,:-2])
			Vy_average = 0.5*(Vy[ :-2,1:-1]+Vy[:-2,:-2])
			
			mxx += m_Vx*dt/dx
			myy += m_Vy*dt/dy

			T += dt
			Step +=1

			if Step % step : continue

			self.plot(T, Step, eta_n, mxx, myy, m_cat, sii, P, Vx_average, Vy_average)
			self.save(Step, mxx, myy, m_cat, m_eta, m_rho, m_C, m_sinphi)

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

		plt.savefig('%s/%12.8f.png' % (self.figname, Myr(T)))
		plt.close(fig)

	
	def save(self, Step, mxx, myy, m_cat, m_eta, m_rho, m_C, m_sinphi):
		Myr = lambda t: t/(365.25*24*3600*10**6) # Convert seconds to millions of year
		np.savez("%s/%s.npz" % (self.figname, Step), mxx, myy, m_cat, m_eta, m_rho, m_C, m_sinphi)
