import matplotlib
import sys
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pylab as plt
import numpy as np

import tkinter as Tk
import tkinter.ttk as ttk

from .materials import materials as materials_

class MyDialog:

    def __init__(self, parent):

        top = self.top = Tk.Toplevel(parent)

        Tk.Label(top, text="Value").pack()

        self.e = Tk.Entry(top)
        self.e.pack(padx=5)

        b = Tk.Button(top, text="OK", command=self.ok)
        b.pack(pady=5)

    def ok(self):
        print ("value is", self.e.get())

        self.top.destroy()

class View(object):
    selected_cell = None
    selected_category = None
    moving_circles = []

    def __init__(self, fload, fsave, fadd,
                 array, materials, boundaries, moving_cells):
        # set common variables
        self.array = array
        self.materials = materials
        self.boundaries = boundaries
        self.moving_cells = moving_cells

        # set bindings
        self.moving_cells.bind(self.redraw_moving_cells)

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
        self.my_cmap.set_over('w')

        # create canvas
        fig = plt.figure()
        self.im = plt.imshow(self.array, cmap=self.my_cmap)
        self.ax = plt.gca()
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

        # create boundary conditions radiobuttons

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

        # create moving points list

        movingpointsgroup = Tk.LabelFrame(group, text="Moving points:")
        movingpointsgroup.pack(fill=Tk.X)

        self.movingpoints = []
        self.mp_listbox = Tk.Listbox(movingpointsgroup)
        self.mp_listbox.bind("<<ListboxSelect>>", self.mp_select)
        self.mp_listbox.pack(fill=Tk.X)
        add_mp = Tk.Button(movingpointsgroup, text = 'Add', command = self.add_mp).pack()
        del_mp = Tk.Button(movingpointsgroup, text = 'Set Vx Vy', command = self.set_mp).pack()
        del_mp = Tk.Button(movingpointsgroup, text = 'Del', command = self.del_mp).pack(side=Tk.BOTTOM)

        fig.canvas.callbacks.connect('button_press_event', self.canvas_click_callback)
        fig.canvas.mpl_connect('button_press_event', self.canvas_click_callback)

        self.fig = fig

    def add_mp(self, *args):
        if self.selected_cell is None:
            return

        # check if we already have this cell in list
        if self.selected_cell in [(x,y) for (x,y),(Vx,Vy) in self.moving_cells]:
            return

        Vx, Vy = 0,0
        self.moving_cells.append((self.selected_cell,(Vx,Vy)))
        self.selected_cell = None
        self.selected_circle.remove()
        self.canvas.draw()

    def set_mp(self, *args):
        listbox = self.mp_listbox
        try:
            selected_index =  int(listbox.curselection()[0])
        except IndexError:
            print('cant select moving cell')
            return
        d = MyDialog(self.root)
        self.root.wait_window(d.top)


    def update_moving_cells_list(self, *args):
        listbox = self.mp_listbox
        listbox.delete(0, Tk.END)
        for cell in self.moving_cells:
            (x,y),(Vx,Vy) = cell
            listbox.insert(Tk.END, f"{Vx}, {Vy}")

    def mp_select(self, event):
        listbox = self.mp_listbox
        try:
            selected_index =  int(event.widget.curselection()[0])
        except IndexError:
            print('cant select moving cell')
            return
        (x,y),_ = self.moving_cells[selected_index]
        self.canvas_click_callback(event, xy=(x,y))

    def del_mp(self, *args):
        pass

    def canvas_click_callback(self, event, xy = None):
        if xy is None:
            x, y = int(round(event.xdata)), int(round(event.ydata))
        else:
            x, y = xy

        # check if we clicked on existent point
        if (x, y) in [(xy) for (xy),(Vx,Vy) in self.moving_cells]:
            index = [(xy) for (xy),_ in self.moving_cells].index((x,y))
            self.mp_listbox.selection_clear(0,Tk.END)
            self.mp_listbox.selection_set(index)

        if self.selected_cell is None:
            self.selected_cell =  x, y
            self.selected_circle = plt.Circle((x, y), .1, color='orange')
            self.ax.add_artist(self.selected_circle)
        else:
            self.selected_cell =  x, y
            self.selected_circle.center = x, y
        self.canvas.draw()

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
        material = self.materials[self.selected_category]['name']
        self.materialvar.set(material)

    def update_lb_materials(self, *args):
        listbox = self.lb_materials
        listbox.delete(0, Tk.END)
        for (i,item) in enumerate([i["name"] for i in self.materials]):
            listbox.insert(Tk.END, "%s : %s" % (i+1, item))
        if self.selected_category:
            listbox.selection_clear(0,Tk.END)
            listbox.selection_set(self.selected_category)

    def main_loop(self):
        self.root.mainloop()

    def quit(self, *args):
        self.root.quit()
        self.root.destroy()


    def redraw_moving_cells(self, *args):
        #remove artists
        for circle in self.moving_circles:
            circle.remove()

        # add artists
        self.moving_circles = []
        for cell in self.moving_cells:
            (x,y), (Vx, Vy) = cell
            circle = plt.Circle((x,y), .1, color='white')
            self.ax.add_artist(circle)
            self.moving_circles.append(circle)
        self.canvas.draw()

    def redraw_canvas(self, im_to_show=None):
        if not im_to_show is None:
            self.im.set_data(im_to_show)
        else:
            self.im = plt.imshow(self.array, cmap=self.my_cmap)
        self.canvas.draw()
