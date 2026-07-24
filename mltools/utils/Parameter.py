import numpy as np

class Parameter:
    def __init__(self, *args, n=None):
        if len(args)>0:
            self.__state = np.array(args)
        else:
            self.__state = np.zeros(n)

    def __len__(self):
        return len(self.__state)

    def state(self):
        return self.__state

    def set(self, state):
        if len(self) == state.size:
            self.__state = state.flatten()
        else:
            raise ValueError()

    def add(self, state):
        self.__state += state.flat