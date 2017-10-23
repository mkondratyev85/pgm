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

    def __init__(self):
        self.root = Tk.Tk()
        self.root.wm_title("PGM Model Constructor")

    def main_loop(self):
        self.root.mainloop()

class Model(object):

	def __init__(self):
		pass

class Controller(object):

	def __init__(self, Model, View):
		self.Model = Model
		self.View = View




if (__name__ == "__main__"):
    view = View()
    model = Model()
    controller = Controller(model, view)

    view.main_loop()
