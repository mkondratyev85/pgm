from pgm import PGM
from pgm import load_step
from test_mmodel import load_model

sec = lambda t: t * 365.25 * 24 * 3600 * 10**6 # Converts Myrs back to seconds

width  = 1500 * 1000 # width of a model in meters
height = 1000 * 1000 # height of a model in meters
j_res  =  41 # width resolution
i_res  =  41 # height resolution
gx_0 = 0 # horizontal gravity field in m/s2
gy_0 = 10 # vertical gravity field in m/s2

pdensity = 4

mxx, myy, m_cat, m_rho, m_eta, m_mu, m_C, m_sinphi = load_model(i_res, j_res, pdensity)

#width, height, j_res, i_res, gx_0, gy_0, mxx, myy, m_cat, m_rho, m_eta = load_step("/tmp/t",8)


model = PGM( width, height, j_res, i_res, gx_0, gy_0, mxx, myy, m_cat, m_rho, m_eta)
model.run(sec(10), 2, "/tmp/t")
