import numpy as np

def interpolate2m(mxx,myy,B):
	# Interpolation from grid to markers p.117 eq.8.19
	i_res, j_res = B.shape

	mxx_int = mxx.astype(int)
	myy_int = myy.astype(int)

	mxx_res = mxx - mxx_int
	myy_res = myy - myy_int
	values = np.zeros(np.shape(mxx_int))
	for idx in range(len(mxx_int)):
		i = myy_int[idx]
		j = mxx_int[idx]
		x = mxx_res[idx]
		y = myy_res[idx]
		if (i > i_res-2) and (j > j_res-2):
			values[idx] = B[i,j]
		elif i > i_res-2:
			values[idx] = (B[i,j]*(1-x) + B[i,j+1]*x)*2
		elif j > j_res-2: 
			values[idx] = (B[i,j]*(1-y) + B[i+1,j]*y)*2
		else:
			values[idx] = B[i,j]*(1-x)*(1-y) + B[i,j+1]*x*(1-y) + B[i+1,j]*(1-x)*y + B[i+1,j+1]*x*y
	return values

def interpolate(mxx,myy,i_res,j_res, datas):
	# Bilineral interpolation (first-order accurate) p.116 eq.8.18
	mxx_round = np.round(mxx)
	myy_round = np.round(myy)
	mxx_res = np.abs(mxx - mxx_round)
	myy_res = np.abs(myy - myy_round)
	wm = (1 - mxx_res)*(1-myy_res)

	down = np.zeros((i_res,j_res))
	values = []
	for i in range(len(datas)):
		values.append(np.zeros((i_res,j_res)))
	for idx in range(len(mxx)):
		i = int(myy_round[idx])
		j = int(mxx_round[idx])
		down[i,j] += wm[idx]
		for (k,data_n) in enumerate(values):
			data_n[i,j] += datas[k][idx] * wm[idx][...] # [...] is a hack for proper multiplying of different types
	for i in range(len(values)):
		values[i] = values[i]/down
	return values

def interpolate_single(mxx,myy, i_res,j_res, data):
	# Bilineral interpolation (first-order accurate) p.116 eq.8.18
	mxx_round = np.round(mxx)
	myy_round = np.round(myy)
	mxx_res = np.abs(mxx - mxx_round)
	myy_res = np.abs(myy - myy_round)
	wm = (1 - mxx_res)*(1-myy_res)

	down = np.zeros((i_res,j_res))
	values = np.zeros((i_res,j_res))
	for idx in range(len(mxx)):
		i = myy_round[idx]
		j = mxx_round[idx]
		down[i,j] += wm[idx]
		values[i,j] += data[idx] * wm[idx]
	values = values/down
	return values
