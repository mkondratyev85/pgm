class Observable(object):

    def __init__(self,var):
        self._var = var
        self._observers = []

    def __iter__(self):
        return iter(self._var)


    def __getitem__(self, index):
        return self._var[index]

    def __setitem__(self, index, value):
        self._var[index] = value
        for callback in self._observers:
            callback(self._var)

    def __len__(self):
        return len(self._var)

    def __str__(self):
        return self._var.__str__()

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
