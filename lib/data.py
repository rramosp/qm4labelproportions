import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from rlxutils import subplots
import tensorflow as tf
import shapely as sh
import rasterio as rio
from pyproj.crs import CRS
import pathlib
import geopandas as gpd

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

    def get_polygon(self):
        """
        returns a shapely polygon in degrees as lon/lat (epsg 4326)
        """
        g = self.data['metadata']['corners']
        nw_lat, nw_lon = g['nw']
        se_lat, se_lon = g['se']
        p = sh.geometry.Polygon([[nw_lon, nw_lat], [se_lon, nw_lat], [se_lon, se_lat], [nw_lon, se_lat], [nw_lon, nw_lat]])
        return p

    def to_geotiff(self, filename=None):
        if filename is not None:
            geotiff_filename = filename
        else:
            fpath = pathlib.Path(self.filename)
            geotiff_filename  = self.filename[:-len(fpath.suffix)] + '.tif'
            
        pixels = self.data['chipmean']

        maxy, minx = self.metadata['corners']['nw']
        miny, maxx = self.metadata['corners']['se']        

        transform = rio.transform.from_origin(minx, maxy, (maxx-minx)/pixels.shape[1], (maxy-miny)/pixels.shape[0])

        new_dataset = rio.open(geotiff_filename, 'w', driver='GTiff',
                                    height = pixels.shape[0], width = pixels.shape[1],
                                    count=3, dtype=str(pixels.dtype),
                                    crs=CRS.from_epsg(4326),
                                    transform=transform)

        for i in range(3):
            new_dataset.write(pixels[:,:,i], i+1)
        new_dataset.close()

    def plot(self):

        for ax,i in subplots(3, usizex=5, usizey=3.5):
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
                k.T.plot(kind='bar', ax=ax, cmap=plt.cm.brg)
                plt.title("label proportions at\ndifferent partition sizes")
                plt.ylim(0,1); plt.grid();
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

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

    @classmethod
    def split_per_partition(cls, split_file, partitions_id=None, cache_size=1000, **kwargs):
        """
        creates three data loaders according to splits as defined in split_file
        """
        assert partitions_id is not None, "must set 'partitions_id'"

        # read split file, and data files
        splits = pd.read_csv(split_file)
        datadir = os.path.dirname(split_file)+"/data"
        files = os.listdir(datadir)

        # select only files which exist for each split
        split_files = {}
        for split in ['train', 'test', 'val']:
            split_identifiers = [i+".pkl" for i in splits[splits[f"split_{partitions_id}"]==split].identifier.values]
            split_files[split] = list(set(split_identifiers).intersection(files))

        # split also the cache sizes
        train_size = len(split_files['train'])
        test_size = len(split_files['test'])
        val_size = len(split_files['train'])
        total_size = train_size + test_size + val_size    

        tr_cache_size = int(cache_size * train_size / total_size)
        ts_cache_size = int(cache_size * test_size / total_size)
        val_cache_size = cache_size - tr_cache_size - ts_cache_size

        # create dataloader for each split
        tr = cls(basedir=datadir, chips_basedirs=split_files['train'], cache_size = tr_cache_size, partitions_id = partitions_id, **kwargs)
        ts = cls(basedir=datadir, chips_basedirs=split_files['test'], cache_size = ts_cache_size, partitions_id = partitions_id, **kwargs)
        val = cls(basedir=datadir, chips_basedirs=split_files['val'], cache_size = val_cache_size, partitions_id = partitions_id, **kwargs)
        
        return tr, ts, val        
            
    def __init__(self, basedir, 
                 partitions_id='5k', 
                 batch_size=32, 
                 shuffle=True,
                 cache_size=100,
                 chips_basedirs=None,
                 number_of_classes=12
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
        self.number_of_classes = number_of_classes
        print (f"got {len(self.chips_basedirs):6d} chips on {len(self)} batches. cache size is {self.cache_size}")

    def empty_cache(self):
        self.cache = {}

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
            if f'partitions{self.partitions_id}' in chip.label_proportions.keys():
                p = chip.label_proportions[f'partitions{self.partitions_id}']
            elif f'partitions_{self.partitions_id}' in chip.label_proportions.keys():
                k = f'partitions_{self.partitions_id}'
                if 'proportions' in chip.label_proportions[k].keys():
                    p = chip.label_proportions[k]['proportions']
                else:
                    p = chip.label_proportions[k]

            else:
                raise ValueError(f"chip has no label partitions for {self.partitions_id}")
            p = np.r_[[p[str(i)] if str(i) in p.keys() else 0 for i in range(self.number_of_classes)]]    

            # ensure 100x100 chips
            r = [chip.chipmean[:100,:100,:], chip.label[:100,:100], p]
            if len(self.cache) < self.cache_size:
                self.cache[chip_filename] = r
            return r

    def __data_generation(self, chips_basedirs_temp):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 100, 100, 3), dtype=np.float32)
        labels = np.empty((self.batch_size, 100, 100), dtype=np.int16)
        partition_proportions = np.empty((self.batch_size, self.number_of_classes), dtype=np.float32)

        # Generate data
        for i, chip_id in enumerate(chips_basedirs_temp):
            # Store sample
            chip_mean, label, p = self.load_chip(chip_id)
            X[i,] = chip_mean/256
            labels[i,] = label
            partition_proportions[i,] = p

        return X, (partition_proportions, labels)


class S2_ESAWorldCover_DataGenerator(S2LandcoverDataGenerator):

    def __init__(self, basedir, **kwargs):
        if 'number_of_classes' in kwargs.keys() and kwargs['number_of_classes']!=12:
            raise ValueError("cannot use 'number_of_classes'!=12, since it is fixed in ESAWordCover")
        kwargs['number_of_classes'] = 12
        super().__init__(basedir, **kwargs)

    @classmethod
    def split(cls, basedir, train_size=0.7, test_size=0.3, val_size=0.0, cache_size=1000, **kwargs):
        if 'number_of_classes' in kwargs.keys():
            raise ValueError("cannot use 'num_classes', since it is fixed in ESAWordCover")

        kwargs['number_of_classes'] = 12
        return super().split(basedir=basedir, train_size=train_size, test_size=test_size, val_size=val_size, cache_size=cache_size, **kwargs)

    @classmethod
    def split_per_partition(cls, split_file, partitions_id=None, cache_size=1000, **kwargs):
        print (kwargs)
        if 'number_of_classes' in kwargs.keys():
            raise ValueError("cannot use 'num_classes', since it is fixed in ESAWordCover")
        kwargs['number_of_classes'] = 12
        return super().split_per_partition(split_file=split_file, partitions_id=partitions_id, cache_size=cache_size, **kwargs)

class S2_EUCrop_DataGenerator(S2LandcoverDataGenerator):

    def __init__(self, basedir, **kwargs):
        if 'number_of_classes' in kwargs.keys() and kwargs['number_of_classes']!=23:
            raise ValueError("cannot use 'number_of_classes'!=12, since it is fixed in ESAWordCover")
        kwargs['number_of_classes'] = 23
        super().__init__(basedir, **kwargs)

    @classmethod
    def split(cls, basedir, train_size=0.7, test_size=0.3, val_size=0.0, cache_size=1000, **kwargs):
        if 'number_of_classes' in kwargs.keys():
            raise ValueError("cannot use 'num_classes', since it is fixed in ESAWordCover")

        kwargs['number_of_classes'] = 23
        return super().split(basedir=basedir, train_size=train_size, test_size=test_size, val_size=val_size, cache_size=cache_size, **kwargs)

    @classmethod
    def split_per_partition(cls, split_file, partitions_id=None, cache_size=1000, **kwargs):
        print (kwargs)
        if 'number_of_classes' in kwargs.keys():
            raise ValueError("cannot use 'num_classes', since it is fixed in ESAWordCover")
        kwargs['number_of_classes'] = 23
        return super().split_per_partition(split_file=split_file, partitions_id=partitions_id, cache_size=cache_size, **kwargs)
