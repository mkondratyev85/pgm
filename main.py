from control import PGM, load_step
#from test_mmodel import load_model
#from t import load_model
from benchmark2 import load_model


sec = lambda t: t * 365.25 * 24 * 3600 * 10**6 # Converts Myrs back to seconds

width  = 1000 * 1000 # width of a model in meters
height = 1000 * 1000 # height of a model in meters
j_res  =  21 # width resolution
i_res  =  23 # height resolution
gx_0 = 0 # horizontal gravity field in m/s2
gy_0 = 10 # vertical gravity field in m/s2

pdensity = 10

model_prop = load_model(i_res, j_res, pdensity)

#width, height, j_res, i_res, gx_0, gy_0, right_bound, left_bound, top_bound, bottom_bound, model_prop = load_step("/home/fatune/t6",35)
#gy_0 = 0

figname = '/tmp/t7/'
control = PGM( width, height, j_res, i_res, gx_0, gy_0, model_prop, figname = figname)

parameters = {'width' : width,
              'height' : height,
              'j_res' : j_res,
              'i_res' : i_res,
              'gx_0' : gx_0,
              'gy_0' : gy_0,
              'p0cell' : 0,
}
parameters.update(model_prop)

control.init_(parameters)
control.make_step_(MaxT=sec(10000000000))

#control.run(sec(1000000000000), 10, figname)
