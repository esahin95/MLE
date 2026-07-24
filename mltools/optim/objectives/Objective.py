from .. import Parameter

class Objective(Parameter):
    def __init__(self, *args, n=None):
        super().__init__(*args, n=n)
        self.__nEval = 0

    def eval(self):
        raise NotImplementedError()

    def inc(self):
        self.__nEval += 1

    def nEval(self):
        return self.__nEval