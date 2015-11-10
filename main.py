from pgm import PGM
from test_mmodel import load_model

sec = lammda t: t * 365.25 * 24 * 3600 * 10**6 # Converts Myrs back to seconds

width  = 1500 * 1000 # width of a model in meters
height = 1000 * 1000 # height of a model in meters
j_res  =  101 # width resolution
i_res  =  91 # height resolution

pdensity = 2

gx_0 = 0 # horizontal gravity field in m/s2
gy_0 = 10 # vertical gravity field in m/s2

mxx, myy, m_cat, m_rho, m_eta, m_mu, m_C, m_sinphi = load_model(i_res, j_res, pdensity)

PGM.init( (width, height), (j_res, i_res), (gx_0, gy_0), (mxx, myy, m_cat, m_rho, m_eta))
PGM.run(sec(10), 1, "/tmp/")
