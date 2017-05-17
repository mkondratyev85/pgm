import numpy as np
import matplotlib.pylab as plt
from scipy import sparse
from Stokeselvis import return_sparse_matrix_Stokes
from interpolate import interpolate, interpolate2m, interpolate_harmonic
import subprocess
import pl

Myr = lambda t: t/(365.25*24*3600*10**6) # Convert seconds to millions of year

average = lambda x: (x[:-1,:-1] + x[1:,:-1] +x[1:,1:] +x[:-1,1:])/4.0

def fill_nans(m):
	mask = np.isnan(m)
	m[mask] = np.interp(np.flatnonzero(mask),np.flatnonzero(~mask),m[~mask])

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
		self.moving_points = model_prop["moving_points"]

		self.T = T
		self.Step = T

		#self.kbond   = 4*self.m_eta.min()/((self.dx+self.dy)**2)
		#self.kcont   = 2*self.m_eta.min()/(self.dx+self.dy)
		self.label = subprocess.check_output(["git", "describe", "--always"])[:-1].decode('utf-8')

	def run(self, maxT, step, cell_step, filename, dt_min=1e+10):
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
		#kbond, kcont = self.kbond, self.kcont
		p0cell = self.p0cell

		eta_min = 1e+2
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

		dt = 1e+10
		dt0 = dt
		plastic_iterations = 10
		plastic_mask = np.zeros(mxx.shape).astype("bool")
		while T < maxT:
			#dt = max(dt, dt_min)
			dt = 1e+10
			was_plastic = False
			for i in range(plastic_iterations):
				dt = 1e+10
				if Myr(T) > 10: gy_0 = 0
				print ("T:%s, dt:%s, dt0:%s" % (T,dt,dt0))

				plastic_yield = False

				m_xelvis = m_eta/(m_mu*dt0 + m_eta)
				m_s_xx_new = m_s_xx*m_xelvis + 2*m_eta*m_e_xx*(1-m_xelvis)
				m_s_xy_new = m_s_xy*m_xelvis + 2*m_eta*m_e_xy*(1-m_xelvis)

				m_s_ii_new = (m_s_xx_new**2 + m_s_xy_new**2)**.5
				m_s_ii_yield = m_C + m_sinphi*m_P
				mask = m_s_ii_yield <0
				m_s_ii_yield[mask] = 0

				ym = m_s_ii_yield < m_s_ii_new # yielding mask
				check_for_plastic = False
				#print("step")
				plastic_current_mask = np.zeros(mxx.shape).astype("bool")
				if check_for_plastic and np.any(ym):
				#ym = m_s_ii_yield < m_s_ii_new # yielding mask
				#if not np.all(ym):
					plastic_yield = True
					print("plastic yield!")

					m_e_ii_old[ym] = np.sqrt(m_e_xx[ym]**2 + m_e_xy[ym]**2)
					m_eta[ym]      = m_s_ii_yield[ym] /2/m_e_ii_old[ym]
					m_s_ii_old[ym] = np.sqrt(m_s_xx[ym]**2 + m_s_xy[ym]**2)
					m_s_xx[ym]     = m_s_xx[ym]*m_s_ii_yield[ym]/m_s_ii_old[ym]
					m_s_xy[ym]     = m_s_xy[ym]*m_s_ii_yield[ym]/m_s_ii_old[ym]

					#m_eta[m_eta<eta_min] = eta_min
					#m_eta[m_eta>eta_max] = eta_max
					plastic_current_mask = ym
					plastic_mask = np.logical_or(plastic_mask, ym)

				eta_s, rho, so_xy = interpolate(mxx,myy,i_res,j_res, (m_eta, m_rho, m_s_xy))
				eta_n, so_xx = interpolate(mxx+.5,myy+.5,i_res,j_res, (m_eta, m_s_xx))
				mu_s = interpolate_harmonic(mxx,myy,i_res,j_res, m_mu )
				mu_n = interpolate_harmonic(mxx+.5,myy+.5,i_res,j_res, m_mu )


				#plt.scatter(mxx,myy,c=m_eta)
				#plt.colorbar()
				#plt.show()
				
				if was_plastic:
					so_xx, so_xy = so0_xx.copy(), so0_xy.copy()

				#pl.plot_matrix(self.figname, T, Step*10, eta_s, eta_n, so_xx,so_xy)

				#Check if we have nans
				if np.isnan(eta_s).any(): fill_nans(eta_s)
				if np.isnan(eta_n).any(): fill_nans(eta_n)
				if np.isnan(rho).any(): fill_nans(rho)
				if np.isnan(mu_s).any(): fill_nans(mu_s)
				if np.isnan(mu_n).any(): fill_nans(mu_n)
				if np.isnan(so_xx).any(): fill_nans(so_xx)
				if np.isnan(so_xy).any(): fill_nans(so_xy)

				# compute viscoelastic (numerical) viscosity and stress
				eta_s0 = np.copy(eta_s)
				eta_n0 = np.copy(eta_n)
				so_xy0 = np.copy(so_xy)
				so_xx0 = np.copy(so_xx)
				xelvis_s = eta_s/(eta_s+dt*mu_s)
				xelvis_n = eta_n/(eta_n+dt*mu_n)
				eta_s = eta_s*(1-xelvis_s)
				eta_n = eta_n*(1-xelvis_n)
				so_xy = so_xy*xelvis_s
				so_xx = so_xx*xelvis_n
				#eo_xx = so_xx/eta_n
				#eo_xy = so_xy/eta_s

				kbond   = 4*eta_s.min()/((dx+dy)**2)
				kcont   = 2*eta_s.min()/(dx+dy)

				#print (dx,dy)

				# find current moving points
				Vx_, Vy_ = {},{}
				angular_speed = 10 ** - 30
				for point in self.moving_points:
					py, px = point[2],point[3]	
					pVx, pVy = point[0],point[1]
					px_, py_ = round(px), round(py)	
					#print (pVx, pVy)
					#if pVx == -10.0: 
					#	pVx = 0
					#	cx,cy = px,py
					#if pVy == 10.0: 
					#	pVy = 0
					#	pVx = 10**-10
					#	#r = ((cx-px_)**2 - (cy-py_)**2)**0.5
					#	#pVx = - r * angular_speed * np.sin (angular_speed * dt0)
					#	#pVy =   r * angular_speed * np.cos (angular_speed * dt0)
					#pVx *= 10**-20
					#pVy *= 10**-20
					
					Vx_["%i,%i" % (px_, py_)] = pVx * kcont**2 /100
					Vy_["%i,%i" % (px_, py_)] = pVy * kcont**2 /100

				#pl.plot_matrix(self.figname, T, Step, eta_s, eta_n, so_xx,so_xy)
				#print (eta_s.min(),eta_s.max())

				Stokes_sparse, vector = return_sparse_matrix_Stokes(j_res, i_res, dx, dy, 
						eta_s, eta_n, rho, gx_0, gy_0, so_xx, so_xy, kbond, kcont, p0cell, 
						lower_boundary=self.bottom_bound, upper_boundary=self.top_bound,
						right_boundary=self.right_bound, left_boundary=self.left_bound, 
						Vx_=Vx_, Vy_=Vy_)

				Stokes_solve = sparse.linalg.spsolve(Stokes_sparse, vector)
				P  = Stokes_solve[::3].reshape((i_res),(j_res))
				Vx = Stokes_solve[1::3].reshape((i_res),(j_res))
				Vy = Stokes_solve[2::3].reshape((i_res),(j_res))

				P *= kcont
				#Vx *= kcont# / 100
				#Vy *= kcont# / 100

				#for point in self.moving_points:
				#	py, px = point[2],point[3]	
				#	pVx, pVy = point[0],point[1]
				#	px_, py_ = round(px), round(py)	
				#	print ("%s, %s, %s, %s" % (pVx, pVy, Vx[px_,py_], Vy[px_,py_]))

				Vx_max = np.abs(Vx).max()
				Vy_max = np.abs(Vy).max()
				dtx = cell_step * dx / Vx_max 
				dty = cell_step * dy / Vy_max 
				dt = min(dtx,dty)
				dt0 = dt

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
				e_xy = .5 * (dVx_dy + dVy_dx)
				xelvis_n = eta_n0/(eta_n0+dt*mu_n)
				xelvis_s = eta_s0/(eta_s0+dt*mu_s)
				s_xx = (1-xelvis_n[1:,1:])*2*eta_n0[1:,1:]*e_xx + xelvis_n[1:,1:]*so_xx0[1:,1:]
				s_xy = (1-xelvis_s)*2*eta_s0*e_xy + xelvis_s*so_xy0
				s_ii = (s_xx**2 + average(s_xy**2))**.5
				so0_xx = s_xx.copy()
				so0_xy = s_xy.copy()
				print(so0_xx.min(),so0_xx.max())

				#if not plastic_yield: break
				#print("plastic iteration")


				
				###################################
				#dm_so_xx = m_s_xx.copy()
				#d_ve = .7
				#dt_m_maxwell = m_eta/m_mu
				#ds_xx = s_xx - so_xx[1:,1:]
				#ds_xy = s_xy - so_xy
				#m_so_xx, m_so_xy = m_s_xx.copy(), m_s_xy.copy()
				#m_so_xy_nodes = interpolate2m(mxx,myy,so_xy)
				#m_so_xx_nodes = interpolate2m(mxx-.5,myy-.5, so_xx[1:,1:])
				#multiplier = np.exp((-d_ve*dt/dt_m_maxwell).astype(float))
				#dm_s_xx_subgrid = (m_so_xx_nodes - m_so_xx) * (1-multiplier)
				#dm_s_xy_subgrid = (m_so_xy_nodes - m_so_xy) * (1-multiplier)
				#d_s_xy_subgrid, t_ = interpolate(mxx,myy,i_res,j_res, ( dm_s_xy_subgrid, m_s_xy))
				#d_s_xx_subgrid, t_ = interpolate(mxx+.5,myy+.5,i_res,j_res, ( dm_s_xx_subgrid, m_s_xx))
				#d_s_xx_remainig = ds_xx - d_s_xx_subgrid[1:,1:]
				#d_s_xy_remainig = ds_xy - d_s_xy_subgrid
				#dm_s_xy_remainig = interpolate2m(mxx,myy,d_s_xy_remainig)
				#dm_s_xx_remainig = interpolate2m(mxx-.5,myy-.5,d_s_xx_remainig)
				#m_s_xx_corrected =  m_so_xx + dm_s_xx_subgrid + dm_s_xx_remainig
				#m_s_xy_corrected =  m_so_xy + dm_s_xy_subgrid + dm_s_xy_remainig
				#m_s_xx = m_s_xx_corrected.copy()
				#m_s_xy = m_s_xy_corrected.copy()
				#self.plot2(T, Step, mxx, myy, m_so_xx_nodes, m_so_xx, dm_s_xx_subgrid, d_s_xx_subgrid, d_s_xx_remainig, dm_s_xx_remainig, m_s_xx )
				###################################

				m_s_xx = interpolate2m(mxx-.5,myy-.5,s_xx)
				m_s_xy = interpolate2m(mxx,myy,s_xy)
				#ds_xx = s_xx - so_xx[1:,1:]
				#ds_xy = s_xy - so_xy
				#m_s_xx = m_s_xx + interpolate2m(mxx-.5,myy-.5,ds_xx)
				#m_s_xy = m_s_xy + interpolate2m(mxx,myy,ds_xy)

				m_e_xx = interpolate2m(mxx-.5,myy-.5,e_xx)
				m_e_xy = interpolate2m(mxx,myy,e_xy)
				#de_xx = e_xx - eo_xx[1:,1:]
				#de_xy = e_xy - eo_xy
				#m_e_xx = m_e_xx + interpolate2m(mxx-.5,myy-.5,de_xx)
				#m_e_xy = m_e_xy + interpolate2m(mxx,myy,de_xy)



				if not plastic_yield: break
				print("plastic iteration")
				#was_plastic = True

			m_Vx = interpolate2m(mxx   , myy-.5, Vx[:-1,:])
			m_Vy = interpolate2m(mxx-.5, myy   , Vy[:,:-1])
			m_P  = interpolate2m(mxx-.5, myy-.5, P[1:,1:]) # tecnichaly, there must be +.5,+.5 
								       # but since we slice P, indexing goes one item lower
			w = dVy_dx - dVx_dy

			m_w    = interpolate2m(mxx-.5 ,myy-.5 , w)

			T += dt
			Step +=1

			print ("Step: %s, T: %s, dt: %s" % (Step, T, dt))

			mxxB = mxx + .5*m_Vx*dt/dx
			myyB = myy + .5*m_Vy*dt/dy
			m_VxB = interpolate2m(mxxB   , myyB-.5, Vx[:-1,:])
			m_VyB = interpolate2m(mxxB-.5, myyB   , Vy[:,:-1])
			#mxxC = mxx + .5*m_VxB*dt/dx
			#myyC = myy + .5*m_VyB*dt/dy
			#m_VxC = interpolate2m(mxxC   , myyC-.5, Vx[:-1,:])
			#m_VyC = interpolate2m(mxxC-.5, myyC   , Vy[:,:-1])
			#mxxD = mxx + m_VxC*dt/dx
			#myyD = myy + m_VyC*dt/dy
			#m_VxD = interpolate2m(mxxD   , myyD-.5, Vx[:-1,:])
			#m_VyD = interpolate2m(mxxD-.5, myyD   , Vy[:,:-1])
			#m_Vx_eff = 1/6*(m_Vx + 2*m_VxB + 2*m_VxC + m_VxD)
			#m_Vy_eff = 1/6*(m_Vy + 2*m_VyB + 2*m_VyC + m_VyD)

			mxx += m_VxB*dt/dx
			myy += m_VyB*dt/dy
			#mxx += m_Vx*dt/dx
			#myy += m_Vy*dt/dy
			#mxx += m_Vx_eff*dt/dx
			#myy += m_Vy_eff*dt/dy
#
#			mp_xx = np.array([point[3] for point in self.moving_points])
#			mp_yy = np.array([point[2] for point in self.moving_points])
#
#			#mp_Vx = interpolate2m(mp_xx   , mp_yy-.5, Vx[:-1,:])
#			#mp_Vy = interpolate2m(mp_xx-.5, mp_yy   , Vy[:,:-1])
#			m_Vx_eff, m_Vy_eff = self.move_gr(mp_xx, mp_yy, Vx, Vy,dt,dx,dy)
#
#			mp_xx += m_Vx_eff*dt/dx
#			mp_yy += m_Vy_eff*dt/dy
#
			for (i,point) in enumerate(self.moving_points):
				py, px = point[2],point[3]	
				px_, py_ = round(px), round(py)	
				#pVx, pVy = point[0],point[1]
				pVx, pVy = Vx[px_,py_], Vy[px_,py_]
				self.moving_points[i][3] = px+pVy*dt/dx
				self.moving_points[i][2] = py+pVx*dt/dy
				#print (pVx, m_Vx_eff[i])
				#print ("%s, %s, %s, %s" % (pVx, pVy, Vx[px_,py_], Vy[px_,py_]))

			m_a = m_w * dt
			m_s_xx_ = m_s_xx - m_s_xy * 2 * m_a
			m_s_xy_ = m_s_xy + m_s_xy * 2 * m_a
			m_s_xx, m_s_xy = m_s_xx_.copy(), m_s_xy_.copy()

			#m_e_xx_ = m_e_xx - m_e_xy * 2 * m_a
			#m_e_xy_ = m_e_xy + m_e_xy * 2 * m_a
			#m_e_xx, m_e_xy = m_e_xx_, m_e_xy_

			dt0 = dt

			if Step % step: continue

			m_s_ii = (m_s_xx**2 + m_s_xy**2)**.5
			#pl.plot(self.figname, T, Step, i_res, j_res, eta_s, mxx, myy, m_cat, s_ii, P, Vx, Vy, e_xx, e_xy, s_xx, s_xy, xelvis_s, mu_n, mu_s, w ,m_Vx, m_Vy, m_P, m_s_xx, m_s_xy, m_s_ii, m_e_xx, m_e_xy, m_eta, xelvis_s, eta_s0)
			pl.plot_small(self.figname, T, Step, self.i_res, self.j_res, self.ii, self.jj,self.moving_points,mxx, myy, m_cat, s_ii, P, Vx, Vy, e_xx, e_xy, plastic_mask, plastic_current_mask)

			#self.save(Step, mxx, myy, m_cat, m_mu, m_eta, m_rho, m_C, m_sinphi, m_s_xx, m_s_xy, m_e_xx, m_e_xy, m_P)

	
	def save(self, Step, mxx, myy, m_cat, m_mu, m_eta, m_rho, m_C, m_sinphi, m_s_xx, m_s_xy, m_e_xx, m_e_xy, m_P):
		Myr = lambda t: t/(365.25*24*3600*10**6) # Convert seconds to millions of year
		np.savez("%s/%s.npz" % (self.figname, Step), mxx, myy, m_cat, m_mu, m_eta, m_rho, m_C, m_sinphi, m_s_xx, m_s_xy, m_e_xx, m_e_xy, m_P)

	def move_gr(self,mxx,myy,Vx,Vy,dt,dx,dy):
		m_Vx = interpolate2m(mxx   , myy-.5, Vx[:-1,:])
		m_Vy = interpolate2m(mxx-.5, myy   , Vy[:,:-1])
		mxxB = mxx + .5*m_Vx*dt/dx
		myyB = myy + .5*m_Vy*dt/dy
		m_VxB = interpolate2m(mxxB   , myyB-.5, Vx[:-1,:])
		m_VyB = interpolate2m(mxxB-.5, myyB   , Vy[:,:-1])
		mxxC = mxx + .5*m_VxB*dt/dx
		myyC = myy + .5*m_VyB*dt/dy
		m_VxC = interpolate2m(mxxC   , myyC-.5, Vx[:-1,:])
		m_VyC = interpolate2m(mxxC-.5, myyC   , Vy[:,:-1])
		mxxD = mxx + m_VxC*dt/dx
		myyD = myy + m_VyC*dt/dy
		m_VxD = interpolate2m(mxxD   , myyD-.5, Vx[:-1,:])
		m_VyD = interpolate2m(mxxD-.5, myyD   , Vy[:,:-1])
		m_Vx_eff = 1/6*(m_Vx + 2*m_VxB + 2*m_VxC + m_VxD)
		m_Vy_eff = 1/6*(m_Vy + 2*m_VyB + 2*m_VyC + m_VyD)
		return m_Vx_eff, m_Vy_eff
