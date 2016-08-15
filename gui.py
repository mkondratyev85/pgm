import matplotlib, sys
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pylab as plt
from scipy import ndimage
import numpy as np


if sys.version_info[0] < 3:
	import Tkinter as Tk
	import ttk
else:
	import tkinter as Tk
	import tkinter.ttk as ttk

class observable(object):
	def __init__(self,var):
		self._var = var
		self._observers = []

	def __getitem__(self, index):
		return self._var[index]

	def __setitem__(self, index, value):
		self._var[index] = value
		for callback in self._observers:
			callback(self._var)
	
	def erase(self):
		self._var = []

	def get(self):
		return self._var

	def set(self, value):
		self._var = value
		for callback in self._observers:
			callback(self._var)

	def append(self, value):
		self._var.append(value)
		for callback in self._observers:
			callback(self._var)

	def bind(self, callback):
		self._observers.append(callback)

materials = {"default":{"mu":1,"rho":2,"eta":3, "C":10, "sinphi":11},
		"sticky air": {"mu":4,"rho":5,"eta":6,"C":40,"sinphi":41}}

selected_category = 0

def material_changed(*args, **kwargs):
	global listbox, model_materials
	listbox.delete(0, Tk.END)
	for (i,item) in enumerate(model_materials.get()):
		listbox.insert(Tk.END, "%s : %s" % (i+1, item))
	if selected_category: listbox.activate(selected_category)

def material_selected(*args):
	global muvar, etavar, rhovar, model_materials
	if not selected_category: return False
	selectedmaterial = materialvar.get()
	muvar.set("mu = %s" % materials[selectedmaterial]["mu"])
	rhovar.set("rho = %s" % materials[selectedmaterial]["rho"])
	etavar.set("eta = %s" % materials[selectedmaterial]["eta"])
	Cvar.set("C = %s" % materials[selectedmaterial]["C"])
	sinphivar.set("sinphi = %s" % materials[selectedmaterial]["sinphi"])
	model_materials[selected_category] = selectedmaterial

def listbox_get(event):
	l = event.widget
	sel = int(l.curselection()[0])
	image_to_show = image.copy()
	image_to_show[image_to_show == sel] = -1
	redraw_canvas(image_to_show)
	global selected_category
	selected_category = sel
	materialvar.set(model_materials[selected_category])

def redraw_canvas(im_to_show):
	im.set_data(im_to_show)
	canvas.draw()


root = Tk.Tk()
root.wm_title("Geodynamic Model Constructor")

image_fname = Tk.filedialog.askopenfilename()

image = plt.imread(image_fname)
image = image[:,:,0]*100+image[:,:,1]*10 + image[:,:,2]
image_i, image_j = image.shape
uniqe, vals = np.unique(image,return_inverse=True)
image = vals.reshape((image_i, image_j))


model_materials = observable([])
model_materials.bind(material_changed)
mm = []
for (i,item) in enumerate(uniqe):
	image[image==item] = i
	mm.append("default")
uniqe, vals = np.unique(image,return_inverse=True)
image = vals.reshape((image_i, image_j))

# a tk.DrawingArea

my_cmap = matplotlib.cm.get_cmap('copper')
my_cmap.set_under('r')
fig = plt.figure()
im = plt.imshow(image,cmap=my_cmap)
ax = plt.gca()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.LEFT, fill=Tk.BOTH)


group = Tk.Frame(root)
group.pack(side=Tk.RIGHT,fill=Tk.Y)

catgroup = Tk.LabelFrame(group, text="List of categories:")
catgroup.pack(fill=Tk.X)

listbox = Tk.Listbox(catgroup)
listbox.bind("<<ListboxSelect>>", listbox_get)
listbox.pack(fill=Tk.X)

model_materials.set(mm)

propgroup = Tk.LabelFrame(group, text="Properties of selected category:")
propgroup.pack()

muvar = Tk.StringVar()
etavar = Tk.StringVar()
rhovar = Tk.StringVar()
Cvar = Tk.StringVar()
sinphivar = Tk.StringVar()

materialvar = Tk.StringVar()
materialvar.trace("w", material_selected)
material = ttk.Combobox(propgroup, textvariable=materialvar)
material['values'] = (list(materials.keys()))
material.current(0)
material.pack()

mulabel = Tk.Label(propgroup,textvariable=muvar).pack()
etalabel = Tk.Label(propgroup,textvariable=etavar).pack()
rholabel = Tk.Label(propgroup,textvariable=rhovar).pack()
Clabel = Tk.Label(propgroup,textvariable=Cvar).pack()
sinphilabel = Tk.Label(propgroup,textvariable=sinphivar).pack()

boundgroup = Tk.LabelFrame(group, text="Boundaries condition:")
boundgroup.pack(fill=Tk.X)

leftboundlabel = Tk.Label(boundgroup,text="Left boundary")
leftboundlabel.pack()


def quit(*args):
	print ('quit button press...')
	root.quit()
	root.destroy()

def save(*args):
	fname = "test"
	fname = Tk.filedialog.asksaveasfilename()
	template = """import numpy as np
import matplotlib.pyplot as plt

def load_model(i_res, j_res, marker_density):
	image = np.load("{fname:s}.npz")
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

	rho_key = np.array({rho_list:s})
	eta_key = np.array({eta_list:s})
	mu_key = np.array({mu_list:s})
	C_key = np.array({C_list:s})
	sinphi_key = np.array({sinphi_list:s})

	m_cat = np.copy(values)
	m_rho = rho_key[values]
	m_eta = eta_key[values]
	m_mu = mu_key[values]
	m_C = C_key[values]
	m_sinphi = sinphi_key[values]
	
	return mxx, myy, m_cat, m_rho, m_eta, m_mu, m_C, m_sinphi """ 
	context={
			"fname":"%s" % fname,
			"rho_list":'%s' % [materials[i]['rho'] for i in model_materials],
			"eta_list":'%s' % [materials[i]['eta'] for i in model_materials],
			"mu_list":'%s' % [materials[i]['mu'] for i in model_materials],
			"C_list":'%s' % [materials[i]['C'] for i in model_materials],
			"sinphi_list":'%s' % [materials[i]['sinphi'] for i in model_materials],
			}
	print("Saving")
	print("%s"%[materials[i]["mu"] for i in model_materials])
	with  open('%s.py' % fname ,'w') as myfile:
		  myfile.write(template.format(**context))

button_save = Tk.Button(group, text = 'Save...', command = save).pack()

button_quit = Tk.Button(group, text = 'Quit', command = quit).pack(side=Tk.BOTTOM)


root.mainloop()
