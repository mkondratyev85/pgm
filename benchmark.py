import numpy as np
import matplotlib.pyplot as plt

def load_model(i_res, j_res, marker_density):
	image = np.load("/media/fatune/source/geodynamic/pgm/benchmark.npy")
	image_i, image_j = image.shape

	# markers
	mxx = []
	myy = []
	for x in range((j_res-1)*2):
		for y in range((i_res-1)*2):
			for _ in range(marker_density):
				mxx.append((x+np.random.uniform(0,1,1))/2.0)
				myy.append((y+np.random.uniform(0,1,1))/2.0)
	mxx = np.asarray(mxx)
	myy = np.asarray(myy)

	mj = (mxx*image_j/(j_res-1)).astype(int)
	mi = (myy*image_i/(i_res-1)).astype(int)

	values = np.zeros(np.shape(mxx))
	for idx in range(len(mxx)):
		j,i = mj[idx], mi[idx]
		values[idx] = image[i,j]
	
	values = values.astype(int)

	rho_key = np.array([1, 4000, 4000, 1])
	eta_key = np.array([1000000000000000000000000, 1000000000000000000000000000, 1000000000000000000000000000,                                1000000000000000000000000])
	mu_key = np.array([100000000000000000000, 10000000000, 10000000000, 100000000000000000000])
	C_key = np.array([10, 10, 10, 10])
	sinphi_key = np.array([0.58778525229247314, 0.58778525229247314, 0.58778525229247314, 0.58778525229247314])

	m_cat = np.copy(values)
	m_rho = rho_key[values]
	m_eta = eta_key[values]
	m_mu = mu_key[values]
	m_C = C_key[values]
	m_sinphi = sinphi_key[values]

	top_bound = "sleep"
	bottom_bound = "sleep"
	left_bound = "nosleep"
	right_bound = "sleep"

	model_prop = { "mxx":mxx, "myy":myy, "m_cat": m_cat, "m_rho":m_rho, "m_eta":m_eta, "m_mu":m_mu, 
	"m_C":m_C, "m_sinphi":m_sinphi, "top_bound":top_bound, "bottom_bound":bottom_bound, 
	"left_bound":left_bound, "right_bound":right_bound }

	return model_prop