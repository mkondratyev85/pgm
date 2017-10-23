
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
        filemenu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        # display the menu
        self.root.config(menu=menubar)

    # def save(self):
    # 	fname = Tk.filedialog.asksaveasfilename()
    #     self.fsave(fname)

    def main_loop(self):
        self.root.mainloop()

    def quit(self, *args):
        print('quit button press...')
        self.root.quit()
        self.root.destroy()
