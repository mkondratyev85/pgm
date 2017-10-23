import matplotlib, sys
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pylab as plt
import numpy as np

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

class View(object):
    selected_category = None

    def __init__(self):
        self.root = Tk.Tk()
        self.root.wm_title("PGM Model Constructor")

        #self.model_materials = observable([])
        #self.model_materials.bind(self.material_changed)

        my_cmap = matplotlib.cm.get_cmap('copper')
        my_cmap.set_under('r')

        self._get_image()
        fig = plt.figure()
        self.im = plt.imshow(self.image,cmap = my_cmap)
        ax = plt.gca()

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=Tk.LEFT, fill=Tk.BOTH)

        group = Tk.Frame(self.root)
        group.pack(side=Tk.RIGHT,fill=Tk.Y)

        catgroup = Tk.LabelFrame(group, text="List of categories:")
        catgroup.pack(fill=Tk.X)

        self.listbox = Tk.Listbox(catgroup)
        self.listbox.bind("<<ListboxSelect>>", self.listbox_get2)
        self.listbox.pack(fill=Tk.X)

        self.model_materials = self.mm

        propgroup = Tk.LabelFrame(group, text="Properties of selected category:")
        propgroup.pack()

        self.muvar = Tk.StringVar()
        self.etavar = Tk.StringVar()
        self.rhovar = Tk.StringVar()
        self.Cvar = Tk.StringVar()
        self.sinphivar = Tk.StringVar()

        self.materialvar = Tk.StringVar()
        self.materialvar.trace("w", self.material_selected2)
        material = ttk.Combobox(propgroup, textvariable=self.materialvar)
        material['values'] = (list(materials.keys()))
        material.current(0)
        material.pack()

        mulabel = Tk.Label(propgroup,textvariable=self.muvar).pack()
        etalabel = Tk.Label(propgroup,textvariable=self.etavar).pack()
        rholabel = Tk.Label(propgroup,textvariable=self.rhovar).pack()
        Clabel = Tk.Label(propgroup,textvariable=self.Cvar).pack()
        sinphilabel = Tk.Label(propgroup,textvariable=self.sinphivar).pack()

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

        self.update_category_list()

    def _redraw_canvas(self, im_to_show):
    	self.im.set_data(im_to_show)
    	self.canvas.draw()

    def _get_image(self):
        image_fname = Tk.filedialog.askopenfilename()

        image = plt.imread(image_fname)
        image = image[:,:,0]*100+image[:,:,1]*10 + image[:,:,2]
        image_i, image_j = image.shape
        uniqe, vals = np.unique(image,return_inverse=True)
        self.image = vals.reshape((image_i, image_j))

        self.mm = []
        for (i,item) in enumerate(uniqe):
            self.image[self.image==item] = i
            self.mm.append({"name":"default",
                       "rho":materials["default"]["rho"],
                       "eta":materials["default"]["eta"],
                       "mu":materials["default"]["mu"],
                       "C":materials["default"]["C"],
                       "sinphi":materials["default"]["sinphi"], })

    def material_selected(self, *args):
        print('material_selected')
        model_materials = self.model_materials
        muvar = self.muvar
        etavar = self.etavar
        rhovar = self.rhovar
        if self.selected_category == None:
            return False
        print (self.selected_category)
        selectedmaterial = self.materialvar.get()
        model_materials[self.selected_category] = { "name":selectedmaterial,
                                               "rho":materials[selectedmaterial]["rho"],
                                               "eta":materials[selectedmaterial]["eta"],
                                               "mu":materials[selectedmaterial]["mu"],
                                               "C":materials[selectedmaterial]["C"],
                                               "sinphi":materials[selectedmaterial]["sinphi"], }
        self.model_materials.set(model_materials)
        self.muvar.set("mu = %s" % model_materials.get()[self.selected_category]["mu"])
        self.rhovar.set("rho = %s" % model_materials.get()[self.selected_category]["rho"])
        self.etavar.set("eta = %s" % model_materials.get()[self.selected_category]["eta"])
        self.Cvar.set("C = %s" % model_materials.get()[self.selected_category]["C"])
        self.sinphivar.set("sinphi = %s" % model_materials.get()[self.selected_category]["sinphi"])

    def update_category_list(self):
        listbox = self.listbox
        model_materials = self.model_materials
        listbox.delete(0, Tk.END)
        for (i,item) in enumerate([i["name"] for i in model_materials]):
            listbox.insert(Tk.END, "%s : %s" % (i+1, item))
        if self.selected_category: listbox.activate(self.selected_category)

    def material_changed(self, *args, **kwargs):
        print('material_changed')
        listbox = self.listbox
        model_materials = self.model_materials
        listbox.delete(0, Tk.END)
        for (i,item) in enumerate([i["name"] for i in model_materials]):
            listbox.insert(Tk.END, "%s : %s" % (i+1, item))
        if self.selected_category: listbox.activate(self.selected_category)

    def material_selected2(self, *args):
        print('material selected 2', self.selected_category)
        print(self.materialvar.get())
        model_materials = self.model_materials
        selectedmaterial = self.materialvar.get()
        model_materials[self.selected_category] = { "name":selectedmaterial,
                                               "rho":materials[selectedmaterial]["rho"],
                                               "eta":materials[selectedmaterial]["eta"],
                                               "mu":materials[selectedmaterial]["mu"],
                                               "C":materials[selectedmaterial]["C"],
                                               "sinphi":materials[selectedmaterial]["sinphi"], }
        self.model_materials = model_materials
        self.update_category_list()



    def listbox_get2(self, event):
        try:
            self.selected_category =  int(event.widget.curselection()[0])
        except IndexError:
            pass
        image_to_show = self.image.copy()
        image_to_show[image_to_show == self.selected_category] = -1
        self._redraw_canvas(image_to_show)
        #selectedmaterial = self.model_materials[self.selected_category]["name"]

    def listbox_get(self, event):
        print('listbox_get')
        try:
            sel = int(event.widget.curselection()[0])
            self.sel = sel
            self.selected_category = sel
        except IndexError:
            sel = self.sel
        print(sel)

        image_to_show = self.image.copy()
        image_to_show[image_to_show == sel] = -1

        self._redraw_canvas(image_to_show)
        selectedmaterial = self.model_materials[sel]["name"]
        self.materialvar.set(selectedmaterial)

    def main_loop(self):
        self.root.mainloop()

class Model(object):

    def __init__(self):
        pass

class Controller(object):

    def __init__(self, Model, View):
        self.Model = Model
        self.View = View


materials = {"default":{"mu":1,"rho":1,"eta":1, "C":1, "sinphi":1},
        "magma" :{"mu":8*10**10,"rho":2800,"eta":10**16, "C":10**7,"sinphi":45},
        "light magma" :{"mu":8*10**10,"rho":2600,"eta":10**13, "C":10**7,"sinphi":45},
        "heavy magma" :{"mu":8*10**10,"rho":3200,"eta":10**16, "C":10**7,"sinphi":45},
        "sand" :{"mu":10**6,"rho":1560,"eta":10**9, "C":10,"sinphi":np.sin(np.radians(36))},
        "viso-elastic slab" :{"mu":10**10,"rho":4000,"eta":10**27, "C":10,"sinphi":np.sin(np.radians(36))},
        "viso-elastic medium" :{"mu":10**20,"rho":1,"eta":10**24, "C":10,"sinphi":np.sin(np.radians(36))},
        "sticky air": {"mu":10**6,"rho":1,"eta":10**2,"C":10,"sinphi":0}}


if (__name__ == "__main__"):
    view = View()
    model = Model()
    controller = Controller(model, view)

    view.main_loop()
