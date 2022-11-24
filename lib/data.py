import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class Chipset:
    
    def __init__(self, basedir):
        self.basedir = basedir
        _, _, self.files = list(os.walk(datadir_packed))[0]

    def __len__(self):
        return len(self.files)
        
    def __iter__(self):
        for file in self.files:
            yield Chip(f"{self.basedir}/{file}")        
        
    def random(self):
        file = np.random.choice(self.files)
        return PChip(f"{self.basedir}/{file}")
        
class Chip:
    def __init__(self, filename):
        self.filename = filename
        with open(filename, "rb") as f:
            self.data = pickle.load(f)        
            
        for k,v in self.data.items():
            exec(f"self.{k}=v")
