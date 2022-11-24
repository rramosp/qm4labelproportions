import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from rlxutils import subplots
import tensorflow as tf

lc = {'0': 'none', '1': 'water', '2': 'trees', '3': 'not used', '4': 'flooded vegetation', '5': 'crops'
,
      '6': 'not used', '7': 'built area', '8': 'bare ground', '9': 'snow/ice', '10': 'clouds', '11': 'rangeland'}


class Chipset:
    
    def __init__(self, basedir):
        self.basedir = basedir
        _, _, self.files = list(os.walk(basedir))[0]

    def __len__(self):
        return len(self.files)
        
    def __iter__(self):
        for file in self.files:
            yield Chip(f"{self.basedir}/{file}")        
        
    def random(self):
        file = np.random.choice(self.files)
        return Chip(f"{self.basedir}/{file}")
        
class Chip:
    def __init__(self, filename):
        self.filename = filename
        with open(filename, "rb") as f:
            self.data = pickle.load(f)        
            
        for k,v in self.data.items():
            exec(f"self.{k}=v")

    def remove(self):
        """
        physically remove the file of this chip
        """
        os.remove(self.filename)

    def compute_label_proportions_on_chip_label(self):
        l = pd.Series(self.label.flatten()).value_counts() / 100**2
        return {i: (l[i] if i in l.index else 0) for i in range(12)}


    def plot(self):

        for ax,i in subplots(4, usizex=5, usizey=4):
            if i==0: 
                plt.imshow(self.chipmean)
                plt.title("/".join(self.filename.split("/")[-1:]))
            if i==1:
                cmap=matplotlib.colors.ListedColormap([plt.cm.tab20(i) for i in range(12)])

                plt.imshow(self.label, vmin=0, vmax=11, cmap=cmap, interpolation='none')
                plt.title("label")
                cbar = plt.colorbar(ax=ax, ticks=range(12))
                cbar.ax.set_yticklabels([f"{k} {v}" for k,v in lc.items()])  # vertically oriented colorbar

            if i==2:
                k = pd.DataFrame(self.label_proportions).T
                k = k[[str(i) for i in range(12) if str(i) in k.columns]]
                k.T.plot(kind='bar', ax=ax, cmap=plt.cm.viridis)
                plt.title("label proportions at\ndifferent partition sizes")
                plt.ylim(0,1); plt.grid();
            if i==3:
                l = pd.Series(self.label.flatten()).value_counts() / 100**2
                p = self.compute_label_proportions_on_chip_label()
                plt.bar(range(12), [p[i] for i in range(12)])
                plt.ylim(0,1); plt.grid()
                plt.title("label proportions on chip")   
        return self



class S2LandcoverDataGenerator(tf.keras.utils.Sequence):
    """
    from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    @classmethod
    def split(cls, basedir, train_size=0.7, test_size=0.3, val_size=0.0, cache_size=1000, **kwargs):
        assert np.abs(train_size + test_size + val_size - 1) < 1e-7
        assert not 'cache_size' in kwargs.keys()
        assert not 'chips_basedirs' in kwargs.keys()

        np.random.seed(10)
        chips_basedirs = np.r_[[f for f,subdirs,files in os.walk(basedir) if 'metadata.pkl' in files]]
        cs = Chipset(basedir)
        chips_basedirs = np.r_[cs.files]
        permutation = np.random.permutation(len(chips_basedirs))
        i1 = int(len(permutation)*train_size)
        i2 = int(len(permutation)*(train_size+test_size))

        # split the chips
        tr =  chips_basedirs[permutation[:i1]]
        ts =  chips_basedirs[permutation[i1:i2]]
        val = chips_basedirs[permutation[i2:]]

        # split also the cache sizes
        tr_cache_size = int(cache_size * train_size)
        ts_cache_size = int(cache_size * test_size)
        val_cache_size = cache_size - tr_cache_size - ts_cache_size

        tr = cls(basedir=basedir, chips_basedirs = tr, cache_size=tr_cache_size, **kwargs)
        ts = cls(basedir=basedir, chips_basedirs = ts, cache_size=ts_cache_size, **kwargs)
        val = cls(basedir=basedir, chips_basedirs = val, cache_size=val_cache_size, **kwargs)

        return tr, ts, val
            
    def __init__(self, basedir, 
                 partitions_id='5k', 
                 batch_size=32, 
                 shuffle=True,
                 cache_size=100,
                 chips_basedirs=None
                 ):
        self.basedir = basedir
        if chips_basedirs is None:
            cs = Chipset(basedir)
            self.chips_basedirs = cs.files
        else:
            self.chips_basedirs = chips_basedirs    
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.partitions_id = partitions_id
        self.on_epoch_end()
        self.cache = {}
        self.cache_size = cache_size
        print (f"got {len(self.chips_basedirs):6d} chips on {len(self)} batches. cache size is {self.cache_size}")

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.chips_basedirs) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.chips_basedirs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        chips_basedirs_temp = [self.chips_basedirs[k] for k in indexes]

        # Generate data
        x = self.__data_generation(chips_basedirs_temp)

        return x

    def load_chip(self, chip_filename):
        if chip_filename in self.cache.keys():
            return self.cache[chip_filename]
        else:
            chip = Chip(f"{self.basedir}/{chip_filename}")
            p = chip.label_proportions[f'partitions{self.partitions_id}']
            p = np.r_[[p[str(i)] if str(i) in p.keys() else 0 for i in range(12)]]    

            r = [chip.chipmean, chip.label, p]
            if len(self.cache) < self.cache_size:
                self.cache[chip_filename] = r
            return r

    def __data_generation(self, chips_basedirs_temp):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 100, 100, 3), dtype=np.float32)
        labels = np.empty((self.batch_size, 100, 100), dtype=np.int16)
        partition_proportions = np.empty((self.batch_size, 12), dtype=np.float32)

        # Generate data
        for i, chip_id in enumerate(chips_basedirs_temp):
            # Store sample
            chip_mean, label, p = self.load_chip(chip_id)
            X[i,] = chip_mean/256
            labels[i,] = label
            partition_proportions[i,] = p

        return X, partition_proportions, labels
