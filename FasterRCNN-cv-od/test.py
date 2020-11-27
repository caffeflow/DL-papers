# %%


class T(object):
    def __init__(self,):
        import numpy 
        self.array = numpy.arange
        pass

    def __call__(self):
        return self.array(5)
T()()
# %%
