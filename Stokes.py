import numpy as np
from scipy.sparse import linalg
from scipy import sparse

def add_to_sparse(x,y,value,row,col,data):
		row.append(x)
		col.append(y)
		data.append(value)

def return_sparse_matrix_Stokes(j_res, i_res, dx, dy, eta_s, eta_n, rho, gy_0, kbond, kcont, p0cell,
		                        lower_boundary="slip", upper_boundary="slip",
								left_boundary="slip", right_boundary="slip"):
	# Constructing sparse matrix for solving: x-Stokes, y-Stokes and continuity equations
	# x-Stokes: ETA(d2vx/dx2+d2vx/dy2)-dP/dx=0
	# y-Stokes: ETA(d2vy/dx2+d2vy/dy2)-dP/dy=gy*RHO
	# continuity: dvx/dx+dvy/dy=0
	# Boundary conditions: free slip
	
	row = []
	col = []
	data = []

	# define grid for messing with indexes
	k      = np.linspace(0,(j_res*i_res-1),(j_res*i_res)).astype('int')
	k.shape = ((i_res,j_res))
	
	vector = np.ones((3*i_res,j_res))
	vector = vector.reshape((3*j_res*i_res,1))

	P = lambda k: 3*k
	Vx = lambda k: 3*k+1
	Vy = lambda k: 3*k+2
	
	dx2,dy2 = dx**2, dy**2

	for j in range(0,j_res):
		for i in range(0,i_res):
			# Continuity equation 
			# Ghost pressure unknowns (i=0, j=0): P(i,j)=0
			if i==0 or j==0:
				add_to_sparse(P(k[i][j]),P(k[i][j]),1*kbond,row,col,data)
				vector[P(k[i][j])] = 0
			# Upper and lower left corners dP/dx=0 => P(i,j)-P(i,j+1)=0
			elif (i==1 and j==1) or (i==i_res-1 and j==1):
				add_to_sparse(P(k[i][j]),P(k[i][j])  , 1*kbond     ,row,col,data)
				add_to_sparse(P(k[i][j]),P(k[i][j+1]),-1*kbond    ,row,col,data)
				vector[P(k[i][j])] = 0
			# Upper and lower right corners dP/dx=0 => P(i,j)-P(i,j-1)=0
			elif (i==1 and j==j_res-1) or (i==i_res-1 and j==j_res-1):
				add_to_sparse(P(k[i][j]),P(k[i][j])  , 1*kbond     ,row,col,data)
				add_to_sparse(P(k[i][j]),P(k[i][j-1]),-1*kbond    ,row,col,data)
				vector[P(k[i][j])] = 0
			# One cell
			elif i==1 and j==2:
				add_to_sparse(P(k[i][j]),P(k[i][j]),1*kbond    ,row,col,data) # Coefficient for P(i,j)
				vector[P(k[i][j])] = p0cell                                   # Right-hand-side part
			else:
				# Internal nodes: dvx/dx+dvy/dy=0
				# dvx/dx=(vx(i-1,j)-vx(i-1,j-1))/dx
				add_to_sparse(P(k[i][j]),Vx(k[i-1][j])   ,kcont/dx     ,row,col,data) # Coefficient for P(i-1,j)
				add_to_sparse(P(k[i][j]),Vx(k[i-1][j-1]) ,-kcont/dx    ,row,col,data) # Coefficient for P(i-1,j-1)
				# dvy/dy=(vy(i,j-1)-vy(i-1,j-1))/dy
				add_to_sparse(P(k[i][j]),Vy(k[i  ][j-1]) ,kcont/dy     ,row,col,data) # Coefficient for P(i,j-1)
				add_to_sparse(P(k[i][j]),Vy(k[i-1][j-1]) ,-kcont/dy    ,row,col,data) # Coefficient for P(i-1,j-1)
				vector[P(k[i][j])] = 0                                                # Right-hand-side part

			# x-Stokes equation
			# Ghost Vx unknowns (i=i_res) and boundary nodes (i=0, i=i_res-1, j=0, j=j_res-1)
			# Ghost Vx unknowns (i=i_res: Vx(i,j)=0
			if i == i_res-1:
				add_to_sparse(Vx(k[i][j]),Vx(k[i][j]) ,kbond    ,row,col,data) # Coefficient for Vx(i,j)
				vector[Vx(k[i][j])] = 0                                        # Right-hand-side part
			# Left and Right boundaries (j=0, j=j_res)
			elif (j==0 or j==j_res-1) and i<i_res-1:
				# Free slip, No slip: vx(i,j)=0
				add_to_sparse(Vx(k[i][j]),Vx(k[i][j]) , kbond    ,row,col,data) # Coefficient for Vx(i,j)
				vector[Vx(k[i][j])] = 0                                         # Right-hand-side part
			# Upper boundary, iner points (i=0, 0<j<j_res)
			elif i==0 and 0<j<j_res-1:
				# Free slip dVx/dy=0: Vx(i,j)-Vx(i+1,j)=0
				if upper_boundary=="sleep":
					add_to_sparse(Vx(k[i][j]),Vx(k[i][j]) ,   kbond    ,row,col,data) # Coefficient for Vx(i,j)
					add_to_sparse(Vx(k[i][j]),Vx(k[i+1][j]) ,-kbond    ,row,col,data) # Coefficient for Vx(i+1,j)
					vector[Vx(k[i][j])] = 0                                           # Right-hand-side part
				## No slip vx=0: vx(i,j)-1/3*vx(i+1,j)=0
				else:
					add_to_sparse(Vx(k[i][j]),Vx(k[i][j]) ,   kbond          ,row,col,data) # Coefficient for Vx(i,j)
					add_to_sparse(Vx(k[i][j]),Vx(k[i+1][j]) ,-(1.0/3)*kbond  ,row,col,data) #Coefficient for Vx(i+1,j)
					vector[Vx(k[i][j])] = 0                                                 # Right-hand-side part
			# Lower boundary, iner points (i=i_res-1, 0<j<j_res)
			elif i==i_res-2 and 0<j<j_res-1:
				# Free slip dvx/dy=0: vx(i,j)-vx(i-1,j)=0
				if lower_boundary=="sleep":
					add_to_sparse(Vx(k[i][j]),Vx(k[i][j]) ,   kbond    ,row,col,data) # Coefficient for Vx(i,j)
					add_to_sparse(Vx(k[i][j]),Vx(k[i-1][j]) ,-kbond    ,row,col,data) # Coefficient for Vx(i-1,j)
					vector[Vx(k[i][j])] = 0                                           # Right-hand-side part
				## No slip vx=0: vx(i,j)-1/3*vx(i-1,j)=0
				else:
					add_to_sparse(Vx(k[i][j]),Vx(k[i][j]) ,   kbond          ,row,col,data) # Coefficient for Vx(i,j)
					add_to_sparse(Vx(k[i][j]),Vx(k[i-1][j]) ,-(1.0/3)*kbond  ,row,col,data) #Coefficient for Vx(i-1,j)
					vector[Vx(k[i][j])] = 0                                                 # Right-hand-side part
			else:
				# Internal nodes: dSxx/dx+dSxy/dy-dP/dx=0    
				# dSxx/dx=2*etan(i+1,j+1)*(vx(i,j+1)-vx(i,j))/dx^2-2*etan(i+1,j)*(vx(i,j)-vx(i,j-1))/dx^2
				add_to_sparse(Vx(k[i,j]),Vx(k[i,j+1]), 2*eta_n[i+1,j+1]/dx2                    ,row,col,data) #Coefficient for Vx(i,j+1)
				add_to_sparse(Vx(k[i,j]),Vx(k[i,j-1]), 2*eta_n[i+1,j]/dx2                      ,row,col,data) #Coefficient for Vx(i,j-1)
				add_to_sparse(Vx(k[i,j]),Vx(k[i,j]),  -2*eta_n[i+1,j+1]/dx2-2*eta_n[i+1,j]/dx2 ,row,col,data) #Coefficient for Vx(i,j)
				#dSxy/dy=etas(i+1,j)*((vx(i+1,j)-vx(i,j))/dy^2+(vy(i+1,j)-vy(i+1,j-1))/dx/dy)-
				# -etas(i,j)*((vx(i,j)-vx(i-1,j))/dy^2+(vy(i,j)-vy(i,j-1))/dx/dy)-
				add_to_sparse(Vx(k[i,j]),Vx(k[i+1,j]),  eta_s[i+1,j]/dy2    ,row,col,data)                    #Coefficient for Vx(i+1,j)
				add_to_sparse(Vx(k[i,j]),Vx(k[i-1,j]),  eta_s[i,j]/dy2      ,row,col,data)                    #Coefficient for Vx(i-1,j)
				add_to_sparse(Vx(k[i,j]),Vx(k[i,j]),  -eta_s[i+1,j]/dy2-eta_s[i,j]/dy2 ,row,col,data)         #Coefficient for Vx(i,j)
				add_to_sparse(Vx(k[i,j]),Vy(k[i+1,j]),   eta_s[i+1,j]/dx/dy    ,row,col,data)               #Coefficient for Vy(i+1,j)
				add_to_sparse(Vx(k[i,j]),Vy(k[i+1,j-1]),-eta_s[i+1,j]/dx/dy    ,row,col,data)               #Coefficient for Vy(i+1,j-1)
				add_to_sparse(Vx(k[i,j]),Vy(k[i,j]),  -eta_s[i,j]/dx/dy        ,row,col,data)               #Coefficient for Vy(i,j)
				add_to_sparse(Vx(k[i,j]),Vy(k[i,j-1]), eta_s[i,j]/dx/dy        ,row,col,data)               #Coefficient for Vy(i,j-1)

				# -dP/dx=(P(i+1,j)-P(i+1,j+1))/dx
				add_to_sparse(Vx(k[i][j]),P(k[i+1][j]),   kcont/dx           ,row,col,data) # Coefficient for P(i+1,j)
				add_to_sparse(Vx(k[i][j]),P(k[i+1][j+1]),-kcont/dx           ,row,col,data) # Coefficient for P(i+1,j+1)
				# Right-hand-side part:0
				vector[Vx(k[i][j])] = 0                                           # Right-hand-side part

			# y-Stokes equation
			# Ghost vy unknowns (j=j_res) and boundary nodes (i=0, i=i_res-1, j=0, j=j_res-1)
			# Ghost vy unknowns (j=jres: vy(i,j)=0
			if j==j_res-1:
				add_to_sparse(Vy(k[i][j]),Vy(k[i][j]) ,kbond    ,row,col,data) # Coefficient for Vy(i,j)
				vector[Vy(k[i][j])] = 0                                        # Right-hand-side part
			# Upper and lower boundaries (i=0, i=i_res)
			elif (i==0 or i==i_res-1) and  j<j_res:
				# Free slip, No slip: vy(i,j)=0
				add_to_sparse(Vy(k[i][j]),Vy(k[i][j]) , kbond    ,row,col,data) # Coefficient for Vy(i,j)
				vector[Vy(k[i][j])] = 0                                         # Right-hand-side part
			# Left boundary, iner points (j=0, 0<i<i_res)
			elif j==0 and 0 < i < i_res-1:
				# Free slip dvy/dx=0: vy(i,j)-vy(i,j+1)=0
				if left_boundary == "sleep":
					add_to_sparse(Vy(k[i][j]),Vy(k[i][j]) ,   kbond    ,row,col,data) # Coefficient for Vy(i,j)
					add_to_sparse(Vy(k[i][j]),Vy(k[i][j+1]) ,-kbond    ,row,col,data) # Coefficient for Vy(i,j+1)
					vector[Vy(k[i][j])] = 0                                           # Right-hand-side part
				## No slip vy=0: vy(i,j)-1/3*vy(i,j+1)=0
				else:
					add_to_sparse(Vy(k[i][j]),Vy(k[i][j]) ,   kbond          ,row,col,data) # Coefficient for Vy(i,j)
					add_to_sparse(Vy(k[i][j]),Vy(k[i][j+1]) ,-(1.0/3)*kbond  ,row,col,data) # Coefficient for Vy(i,j+1)
					vector[Vy(k[i][j])] = 0                                                 # Right-hand-side part
			# Right boundary, iner points (j=j_res-1, 0<i<i_res)
			elif j==j_res-2 and 0 < i <i_res-1:
				# Free slip dvy/dx=0: vy(i,j)-vy(i,j-1)=0
				if right_boundary=="sleep":
					add_to_sparse(Vy(k[i][j]),Vy(k[i][j]) ,   kbond    ,row,col,data) # Coefficient for Vy(i,j)
					add_to_sparse(Vy(k[i][j]),Vy(k[i][j-1]) ,-kbond    ,row,col,data) # Coefficient for Vy(i,j-1)
					vector[Vy(k[i][j])] = 0                                           # Right-hand-side part
				## No slip vy=0: vy(i,j)-1/3*vy(i,j-1)=0
				else:
					add_to_sparse(Vy(k[i][j]),Vy(k[i][j]) ,   kbond          ,row,col,data) # Coefficient for Vy(i,j)
					add_to_sparse(Vy(k[i][j]),Vy(k[i][j-1]) ,-(1.0/3)*kbond  ,row,col,data) # Coefficient for Vy(i,j-1)
					vector[Vy(k[i][j])] = 0                                                 # Right-hand-side part
			else:
				# Internal nodes: dSyy/dy+dSxy/dx-dP/dy=-gy*RHO
				#dSyy/dy=2*etan(i+1,j+1)*(vy(i+1,j)-vy(i,j))/dy^2-2*etan(i,j+1)*(vy(i,j)-vy(i-1,j))/dy^2
				add_to_sparse(Vy(k[i,j]),Vy(k[i+1,j]), 2*eta_n[i+1,j+1]/dy2                    ,row,col,data) #Coefficient for Vy(i+1,j)
				add_to_sparse(Vy(k[i,j]),Vy(k[i-1,j]), 2*eta_n[i,j+1]/dy2                      ,row,col,data) #Coefficient for Vy(i-1,j)
				add_to_sparse(Vy(k[i,j]),Vy(k[i,j]),  -2*eta_n[i+1,j+1]/dy2-2*eta_n[i,j+1]/dy2 ,row,col,data) #Coefficient for Vy(i,j)
				#dSxy/dx=etas(i,j+1)*((vy(i,j+1)-vy(i,j))/dx^2+(vx(i,j+1)-vx(i-1,j+1))/dx/dy)-
				#         -etas(i,j)*((vy(i,j)-vy(i,j-1))/dx^2+(vx(i,j)-vx(i-1,j))/dx/dy)-
				add_to_sparse(Vy(k[i,j]),Vy(k[i,j+1]),  eta_s[i,j+1]/dx2    ,row,col,data)                    #Coefficient for Vy(i+1,j)
				add_to_sparse(Vy(k[i,j]),Vy(k[i,j-1]),  eta_s[i,j]/dx2      ,row,col,data)                    #Coefficient for Vy(i-1,j)
				add_to_sparse(Vy(k[i,j]),Vy(k[i,j]),  -eta_s[i,j+1]/dx2-eta_s[i,j]/dx2 ,row,col,data)         #Coefficient for Vy(i,j)
				add_to_sparse(Vy(k[i,j]),Vx(k[i,j+1]),   eta_s[i,j+1]/dx/dy    ,row,col,data)               #Coefficient for Vx(i+1,j)
				add_to_sparse(Vy(k[i,j]),Vx(k[i-1,j+1]),-eta_s[i,j+1]/dx/dy    ,row,col,data)               #Coefficient for Vx(i+1,j-1)
				add_to_sparse(Vy(k[i,j]),Vx(k[i,j]),  -eta_s[i,j]/dx/dy        ,row,col,data)               #Coefficient for Vx(i,j)
				add_to_sparse(Vy(k[i,j]),Vx(k[i-1,j]), eta_s[i,j]/dx/dy        ,row,col,data)               #Coefficient for Vx(i,j-1)
				# -dP/dy=(P(i,j+1)-P(i+1,j+1))/dx
				add_to_sparse(Vy(k[i][j]),P(k[i][j+1]),   kcont/dy           ,row,col,data) # Coefficient for P(i,j+1)
				add_to_sparse(Vy(k[i][j]),P(k[i+1][j+1]),-kcont/dy           ,row,col,data) # Coefficient for P(i+1,j+1)
				# Right part: -RHO*gy
				vector[Vy(k[i][j])] = -gy_0 * (rho[i,j] + rho[i,j+1])/2.0                   # Right-hand-side part


	mtx = sparse.coo_matrix((data, (row, col)), shape=(3*j_res*i_res, 3*j_res*i_res))
	mtx = mtx.tocsr()
	return mtx, vector
