
import matplotlib
import sys
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pylab as plt
import numpy as np

import tkinter as Tk
import tkinter.ttk as ttk


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
        catgroup = Tk.LabelFrame(group, text="List of categories:")
        catgroup.pack(fill=Tk.X)

        self.lb_materials = Tk.Listbox(catgroup)
        # self.lb_materials.bind("<<ListboxSelect>>", self.listbox_get2)
        self.lb_materials.pack(fill=Tk.X)

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
        if im_to_show == None:
            self.im = plt.imshow(self.array, cmap=self.my_cmap)
        else:
            self.im.set_data(im_to_show)
        self.canvas.draw()
