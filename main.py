from control import PGM, load_model, load_settings

sec = lambda t: t * 365.25 * 24 * 3600 * 10**6 # Converts Myrs back to seconds

fname = 'benchmarks/push.py'
model_prop = load_model(fname)

figname = '/tmp/t7/'
control = PGM(model_prop, figname = figname)

control.init_(model_prop)
control.run_(MaxT=sec(10000000000))
