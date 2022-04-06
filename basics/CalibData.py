import numpy as np
class CalibData:
    def __init__(self, dataPath):
        self.dataPath = dataPath
        data = np.load(dataPath)

        self.numBins = data['bins']
        self.grad_r = data['grad_r']
        self.grad_g = data['grad_g']
        self.grad_b = data['grad_b']
