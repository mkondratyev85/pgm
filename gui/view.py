import matplotlib
import sys
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pylab as plt
import numpy as np

import tkinter as Tk
import tkinter.ttk as ttk

from .materials import materials as materials_

class View(object):
    selected_category = None

    def __init__(self, fload, fsave, fadd,
                 array, materials, boundaries, moving_cells):
        # set common variables
        self.array = array
        self.materials = materials
        self.boundaries = boundaries
        self.moving_cells = moving_cells

        # build GUI
        self.root = Tk.Tk()
        self.root.wm_title("PGM Model Constructor")

        # create a toplevel menu
        menubar = Tk.Menu(self.root)

        # create a pulldown menu, and add it to the menu bar
        filemenu = Tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Add image...", command=lambda: fadd(
            Tk.filedialog.askopenfilename(title="Select image file",
                                          filetypes=(("png files", "*.png"),
                                                     ("numpy array files", "*.npy"),
                                                     ("all files", "*.*")))))
        filemenu.add_separator()
        filemenu.add_command(label="Load...", command=lambda: fload(
            Tk.filedialog.askopenfilename(title="Select model file",
                                          filetypes=(("python files", "*.py"),
                                                     ("all files", "*.*")))))
        filemenu.add_command(label="Save...", command=lambda: fsave(
            Tk.filedialog.asksaveasfilename(title="Select file to save a model",
                                          filetypes=(("python files", "*.py"),
                                                     ("all files", "*.*")))))
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        # display the menu
        self.root.config(menu=menubar)


        # create custom colormap for image
        self.my_cmap = matplotlib.cm.get_cmap('copper')
        self.my_cmap.set_under('r')

        # create canvas
        fig = plt.figure()
        self.im = plt.imshow(self.array, cmap=self.my_cmap)
        ax = plt.gca()
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=Tk.LEFT, fill=Tk.BOTH)

        # create right side panel
        group = Tk.Frame(self.root)
        group.pack(side=Tk.RIGHT,fill=Tk.Y)

        # create list of materials
        catgroup = Tk.LabelFrame(group, text="List of model materials:")
        catgroup.pack(fill=Tk.X)

        self.lb_materials = Tk.Listbox(catgroup)
        self.lb_materials.bind("<<ListboxSelect>>", self.lb_material_selected)
        self.lb_materials.pack(fill=Tk.X)

        # create chose of selected material
        propgroup = Tk.LabelFrame(group, text="Properties of selected material:")
        propgroup.pack()

        self.muvar = Tk.StringVar()
        self.etavar = Tk.StringVar()
        self.rhovar = Tk.StringVar()
        self.Cvar = Tk.StringVar()
        self.sinphivar = Tk.StringVar()

        self.materialvar = Tk.StringVar()
        self.materialvar.trace("w", self.material_selected)
        material = ttk.Combobox(propgroup, textvariable=self.materialvar)
        material['values'] = (list(materials_.keys()))
        material.current(0)
        material.pack()

        mulabel = Tk.Label(propgroup,textvariable=self.muvar).pack()
        etalabel = Tk.Label(propgroup,textvariable=self.etavar).pack()
        rholabel = Tk.Label(propgroup,textvariable=self.rhovar).pack()
        Clabel = Tk.Label(propgroup,textvariable=self.Cvar).pack()
        sinphilabel = Tk.Label(propgroup,textvariable=self.sinphivar).pack()

    def material_selected(self, *args):
        if self.selected_category == None:
            return False
        selectedmaterial = self.materialvar.get()
        self.materials[self.selected_category] = { "name":selectedmaterial,
                                               "rho":materials_[selectedmaterial]["rho"],
                                               "eta":materials_[selectedmaterial]["eta"],
                                               "mu":materials_[selectedmaterial]["mu"],
                                               "C":materials_[selectedmaterial]["C"],
                                               "sinphi":materials_[selectedmaterial]["sinphi"], }
        self.muvar.set("mu = %s" % self.materials[self.selected_category]["mu"])
        self.rhovar.set("rho = %s" % self.materials[self.selected_category]["rho"])
        self.etavar.set("eta = %s" % self.materials[self.selected_category]["eta"])
        self.Cvar.set("C = %s" % self.materials[self.selected_category]["C"])
        self.sinphivar.set("sinphi = %s" % self.materials[self.selected_category]["sinphi"])

    def lb_material_selected(self, event):
        try:
            self.selected_category =  int(event.widget.curselection()[0])
        except IndexError:
            pass
        # redraw
        image_to_show = self.array.get().copy()
        image_to_show[image_to_show == self.selected_category] = -1
        self.redraw_canvas(image_to_show)


    def update_lb_materials(self, *args):
        listbox = self.lb_materials
        listbox.delete(0, Tk.END)
        for (i,item) in enumerate([i["name"] for i in self.materials]):
            listbox.insert(Tk.END, "%s : %s" % (i+1, item))
        if self.selected_category: listbox.activate(self.selected_category)

    def main_loop(self):
        self.root.mainloop()

    def quit(self, *args):
        self.root.quit()
        self.root.destroy()

    def redraw_canvas(self, im_to_show=None):
        if not im_to_show is None:
            self.im.set_data(im_to_show)
        else:
            self.im = plt.imshow(self.array, cmap=self.my_cmap)
        self.canvas.draw()
