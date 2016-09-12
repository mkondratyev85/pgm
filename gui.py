import matplotlib, sys
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pylab as plt
from scipy import ndimage
import numpy as np
import pickle


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

materials = {"default":{"mu":1,"rho":1,"eta":1, "C":1, "sinphi":1},
		"magma" :{"mu":8*10**10,"rho":2800,"eta":10**16, "C":10**7,"sinphi":45},
		"light magma" :{"mu":8*10**10,"rho":2600,"eta":10**13, "C":10**7,"sinphi":45},
		"heavy magma" :{"mu":8*10**10,"rho":3200,"eta":10**16, "C":10**7,"sinphi":45},
		"sand" :{"mu":10**6,"rho":1560,"eta":10**9, "C":10,"sinphi":np.sin(np.radians(36))},
		"viso-elastic slab" :{"mu":10**10,"rho":4000,"eta":10**27, "C":10,"sinphi":np.sin(np.radians(36))},
		"viso-elastic medium" :{"mu":10**20,"rho":1,"eta":10**24, "C":10,"sinphi":np.sin(np.radians(36))},
		"rigid body" :{"mu":10**20,"rho":3000,"eta":10**25, "C":0,"sinphi":np.sin(np.radians(36))},
		"sticky air": {"mu":10**6,"rho":1,"eta":10**2,"C":10,"sinphi":0}}

def velocity_changed(*args):
	global point_listbox, moving_points
	if selected_point: 
		try:
			moving_points[selected_point-1][0] = float( Vx_var.get())
		except:
			pass
		try:
			moving_points[selected_point-1][1] = float(Vy_var.get())
		except:
			pass
		point_listbox.delete(0, Tk.END)
		for i in range(len(moving_points.get())):
			point_listbox.insert(Tk.END, "%s : %s" % (i+1, moving_points.get()[i]))
		point_listbox.activate(selected_point)
	else:
		point_listbox.delete(0, Tk.END)
		for i in range(len(moving_points.get())):
			point_listbox.insert(Tk.END, "%s : %s" % (i+1, moving_points.get()[i]))

selected_category = None

moving_points = observable([])
moving_points.bind(velocity_changed)

selected_point = None


def material_changed(*args, **kwargs):
	global listbox, model_materials
	listbox.delete(0, Tk.END)
	for (i,item) in enumerate([i["name"] for i in model_materials.get()]):
		listbox.insert(Tk.END, "%s : %s" % (i+1, item))
	if selected_category: listbox.activate(selected_category)

def material_selected(*args):
	global muvar, etavar, rhovar, model_materials
	if selected_category == None: return False
	selectedmaterial = materialvar.get()
	model_materials[selected_category] = { "name":selectedmaterial,
		                                   "rho":materials[selectedmaterial]["rho"],
		                                   "eta":materials[selectedmaterial]["eta"],
		                                   "mu":materials[selectedmaterial]["mu"],
		                                   "C":materials[selectedmaterial]["C"],
		                                   "sinphi":materials[selectedmaterial]["sinphi"], }
	muvar.set("mu = %.2E" % model_materials.get()[selected_category]["mu"])
	rhovar.set("rho = %.2E" % model_materials.get()[selected_category]["rho"])
	etavar.set("eta = %.2E" % model_materials.get()[selected_category]["eta"])
	Cvar.set("C = %.2E" % model_materials.get()[selected_category]["C"])
	sinphivar.set("sinphi = %.2E" % model_materials.get()[selected_category]["sinphi"])

def point_listbox_get(event):
	l = event.widget
	sel = int(l.curselection()[0])
	#image_to_show = image.copy()
	#image_to_show[image_to_show == sel] = -1
	#redraw_canvas(image_to_show)
	global selected_point
	selected_point = sel+1
	Vx_var.set(moving_points.get()[sel][0])
	Vy_var.set(moving_points.get()[sel][1])
	redraw_canvas(image)

def listbox_get(event):
	l = event.widget
	sel = int(l.curselection()[0])
	image_to_show = image.copy()
	image_to_show[image_to_show == sel] = -1
	redraw_canvas(image_to_show)
	global selected_category
	selected_category = sel
	selectedmaterial = model_materials[selected_category]["name"]
	materialvar.set(selectedmaterial)

def redraw_canvas(im_to_show):
	ax.clear()
	im = plt.imshow(image,cmap=my_cmap,interpolation="nearest")
	im.set_data(im_to_show)
	for point in moving_points:
		x,y = point[2],point[3]
		Vx, Vy = point[0],point[1]
		plt.scatter(x,y,c="blue")
		if Vx == 0 and Vy ==0: continue
		ax.quiver(x,y,Vx,Vy,angles='xy',scale_units='xy',color="blue")
	if selected_point:
		point  = moving_points[selected_point-1]
		x,y = point[2],point[3]
		Vx, Vy = point[0],point[1]
		plt.scatter(x,y,c="red")
		if not( Vx == 0 and Vy ==0):
			ax.quiver(x,y,Vx,Vy,angles='xy',scale_units='xy',color="red")
	width = image.shape[0]
	height = image.shape[1]
	plt.ylim([height-.5,-.5])
	plt.xlim([-.5,width-.5])
	canvas.draw()

def on_click(event):
	global moving_points
	if selected_point:
		print (selected_point)
		moving_points[selected_point-1][2] = event.xdata
		moving_points[selected_point-1][3] = event.ydata
		redraw_canvas(image)
	print( moving_points.get())


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
	#mm.append("default")
	mm.append({"name":"default",
		       "rho":materials["default"]["rho"],
		       "eta":materials["default"]["eta"],
		       "mu":materials["default"]["mu"],
		       "C":materials["default"]["C"],
		       "sinphi":materials["default"]["sinphi"], })
uniqe, vals = np.unique(image,return_inverse=True)
image = vals.reshape((image_i, image_j))

# a tk.DrawingArea

my_cmap = matplotlib.cm.get_cmap('copper')
my_cmap.set_under('r')
fig = plt.figure()
im = plt.imshow(image,cmap=my_cmap,interpolation="nearest")
ax = plt.gca()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.LEFT, fill=Tk.BOTH)
canvas.callbacks.connect('button_press_event', on_click)


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

boundgroup = Tk.LabelFrame(group, text="Boundary conditions:")
boundgroup.pack(fill=Tk.X)

leftvar = Tk.StringVar()
rightvar = Tk.StringVar()
topvar = Tk.StringVar()
bottomvar = Tk.StringVar()

Tk.Radiobutton(boundgroup,text="Free slip", variable=topvar, value="sleep").pack(anchor=Tk.N)
Tk.Radiobutton(boundgroup,text="No free slip", variable=topvar, value="nosleep").pack(anchor=Tk.N)

Tk.Radiobutton(boundgroup,text="Free slip", variable=leftvar, value="sleep").pack(anchor=Tk.W)
Tk.Radiobutton(boundgroup,text="No free slip", variable=leftvar, value="nosleep").pack(anchor=Tk.W)

Tk.Radiobutton(boundgroup,text="Free slip", variable=rightvar, value="sleep").pack(anchor=Tk.E)
Tk.Radiobutton(boundgroup,text="No free slip", variable=rightvar, value="nosleep").pack(anchor=Tk.E)

Tk.Radiobutton(boundgroup,text="Free slip", variable=bottomvar, value="sleep").pack(anchor=Tk.S)
Tk.Radiobutton(boundgroup,text="No free slip", variable=bottomvar, value="nosleep").pack(anchor=Tk.S)

leftvar.set("sleep")
rightvar.set("sleep")
topvar.set("sleep")
bottomvar.set("sleep")

movinggroup = Tk.LabelFrame(group, text="Moving points:")
movinggroup.pack(fill=Tk.X)

point_listbox = Tk.Listbox(movinggroup)
point_listbox.bind("<<ListboxSelect>>", point_listbox_get)
point_listbox.pack(fill=Tk.X)

def add_point(*args):
	width = image.shape[0]
	height = image.shape[1]
	moving_points.append([0,0,width/2,height/2])
	redraw_canvas(image)

def delete_point(*args):
	pass

Vx_var = Tk.StringVar()
Vx_var.trace("w", velocity_changed)
Vxlabel = Tk.Label(movinggroup,text="Vx =").pack()
VxEntry = Tk.Entry(movinggroup,textvariable=Vx_var).pack()

Vy_var = Tk.StringVar()
Vy_var.trace("w", velocity_changed)
Vylabel = Tk.Label(movinggroup,text="Vy =").pack()
VyEntry = Tk.Entry(movinggroup,textvariable=Vy_var).pack()

button_add_move = Tk.Button(movinggroup, text = 'Add moving point...', command = add_point).pack()
button_delete_move = Tk.Button(movinggroup, text = 'Delete moving point...', command = delete_point).pack()


def quit(*args):
	print ('quit button press...')
	root.quit()
	root.destroy()

def save(*args):
	fname = "test"
	fname = Tk.filedialog.asksaveasfilename()
	template = """import numpy as np
import matplotlib.pyplot as plt
import pickle

def load_model(i_res, j_res, marker_density):
	image = np.load("{fname:s}.npy")
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
	m_s_xx = np.zeros(np.shape(mxx))
	m_s_xy = np.zeros(np.shape(mxx))
	m_e_xx = np.zeros(np.shape(mxx))
	m_e_xy = np.zeros(np.shape(mxx))
	m_P    = np.zeros(np.shape(mxx))

	top_bound = "{top:s}"
	bottom_bound = "{bottom:s}"
	left_bound = "{left:s}"
	right_bound = "{right:s}"


	with open('{fname:s}.points', 'rb') as f:
		moving_points = pickle.load(f)
	for (i,point) in enumerate(moving_points):
		px = (point[2]+.5) * (j_res-1)/ image_j
		py = (point[3]+.5) * (i_res-1)/ image_i
		moving_points[i][2] = px
		moving_points[i][3] = py

	model_prop = {p:s} "mxx":mxx, "myy":myy, "m_cat": m_cat, "m_rho":m_rho, "m_eta":m_eta, "m_mu":m_mu, 
	"m_C":m_C, "m_sinphi":m_sinphi, "top_bound":top_bound, "bottom_bound":bottom_bound, 
	"left_bound":left_bound, "right_bound":right_bound,
	"m_s_xx": m_s_xx, "m_s_xy": m_s_xy, "m_e_xx": m_e_xx, "m_e_xy": m_e_xy, "m_P": m_P,
	"moving_points" : moving_points {p2:s}

	return model_prop"""
	
#	return mxx, myy, m_cat, m_rho, m_eta, m_mu, m_C, m_sinphi, top_bound, bottom_bound, left_bound, right_bound """ 
	context={
			"fname":"%s" % fname,
			"rho_list":'%s' % [model_materials[i]['rho'] for i in range(len( model_materials.get()))],
			"eta_list":'%s' % [model_materials[i]['eta'] for i in  range(len(model_materials.get()))],
			"mu_list":'%s' % [model_materials[i]['mu'] for i in  range(len(model_materials.get()))],
			"C_list":'%s' % [model_materials[i]['C'] for i in  range(len(model_materials.get()))],
			"sinphi_list":'%s' % [model_materials[i]['sinphi'] for i in  range(len(model_materials.get()))],
			"top":topvar.get(),
			"bottom":bottomvar.get(),
			"left":leftvar.get(),
			"right":rightvar.get(),
			"p":"{",
			"p2":"}"
			}
	with  open('%s.py' % fname ,'w') as myfile:
		  myfile.write(template.format(**context))
	np.save("%s" % (fname), image)

	# Saving the moving points:
	with open("%s.points" % fname, 'wb') as f:  # Python 3: open(..., 'wb')
	    pickle.dump(moving_points.get(), f)

button_save = Tk.Button(group, text = 'Save...', command = save).pack()

button_quit = Tk.Button(group, text = 'Quit', command = quit).pack(side=Tk.BOTTOM)


root.mainloop()
