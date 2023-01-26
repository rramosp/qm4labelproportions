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
from progressbar import progressbar as pbar

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
    def split(cls, basedir=None, 
                   train_size=0.7, 
                   test_size=0.2, 
                   val_size=0.1, 
                   cache_size=1000, 
                   max_chips=None, 
                   selected_classids = None,
                   partitions_id = 'aschip',
                   batch_size = 32,
                   shuffle = True,
                   ):
        assert basedir is not None, "must set 'basedir'"
        assert np.abs(train_size + test_size + val_size - 1) < 1e-7

        np.random.seed(10)
        cs = Chipset(basedir)
        chips_basedirs = np.r_[cs.files]
        
        if max_chips is not None:
            chips_basedirs = np.random.permutation(chips_basedirs)[:max_chips]
        
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

        # create dataloader for each split
        tr = cls(basedir=basedir, 
                 chips_basedirs=tr, 
                 cache_size = tr_cache_size, 
                 partitions_id = partitions_id, 
                 selected_classids = selected_classids,
                 batch_size = batch_size, 
                 shuffle = shuffle,
                 max_chips = max_chips
                 )
        ts = cls(basedir=basedir, 
                 chips_basedirs=ts, 
                 cache_size = ts_cache_size, 
                 partitions_id = partitions_id, 
                 selected_classids = selected_classids,
                 batch_size = batch_size, 
                 shuffle = shuffle,
                 max_chips = max_chips
                 )
        val = cls(basedir=basedir, 
                  chips_basedirs=val, 
                  cache_size = val_cache_size, 
                  partitions_id = partitions_id, 
                  selected_classids = selected_classids,
                  batch_size = batch_size, 
                  shuffle = shuffle,
                  max_chips = max_chips
                  )

        return tr, ts, val

    @classmethod
    def split_per_partition(cls, 
                            split_file = None, 
                            partitions_id = None, 
                            cache_size = 1000, 
                            max_chips = None,
                            selected_classids = None,
                            batch_size = 32,
                            shuffle = True,
                            ):
        """
        creates three data loaders according to splits as defined in split_file
        """
        assert split_file is not None, "must set 'split_file'"
        assert partitions_id is not None, "must set 'partitions_id'"

        
        # in case use specified a folder containing the splitfile
        if os.path.isdir(split_file):
            basedir = split_file
            split_file_found = None
            # look into that folder and the parent folder
            for basedir in [split_file, f"{split_file}/.."]:
                csv_files = [i for i in os.listdir(basedir) if i.endswith('.csv')]
                if len(csv_files)==1:
                    split_file_found = f"{basedir}/{csv_files[0]}"

            if split_file_found is None:
                raise ValueError("could not find any split file")

            split_file = split_file_found        
        
        # read split file, and data files
        splits = pd.read_csv(split_file)
        datadir = os.path.dirname(split_file)+"/data"
        files = os.listdir(datadir)

        print (len(files))
        if max_chips is not None:
            files = np.random.permutation(files)[:max_chips]

        # select only files which exist for each split
        split_files = {}
        for split in ['train', 'test', 'val']:
            split_identifiers = [i+".pkl" for i in splits[splits[f"split_{partitions_id}"]==split].identifier.values]
            split_files[split] = list(set(split_identifiers).intersection(files))

        # split also the cache sizes
        train_size = len(split_files['train'])
        test_size = len(split_files['test'])
        val_size = len(split_files['val'])
        total_size = train_size + test_size + val_size    

        tr_cache_size = int(cache_size * train_size / total_size)
        ts_cache_size = int(cache_size * test_size / total_size)
        val_cache_size = cache_size - tr_cache_size - ts_cache_size

        # create dataloader for each split
        tr = cls(basedir=datadir, 
                 chips_basedirs=split_files['train'], 
                 cache_size = tr_cache_size, 
                 partitions_id = partitions_id, 
                 selected_classids = selected_classids,
                 batch_size = batch_size, 
                 shuffle = shuffle,
                 max_chips = max_chips
                 )
        ts = cls(basedir=datadir, 
                 chips_basedirs=split_files['test'], 
                 cache_size = ts_cache_size, 
                 partitions_id = partitions_id, 
                 selected_classids = selected_classids,
                 batch_size = batch_size, 
                 shuffle = shuffle,
                 max_chips = max_chips
                 )
        val = cls(basedir=datadir, 
                  chips_basedirs=split_files['val'], 
                  cache_size = val_cache_size, 
                  partitions_id = partitions_id, 
                  selected_classids = selected_classids,
                  batch_size = batch_size, 
                  shuffle = shuffle,
                  max_chips = max_chips
                  )
        
        return tr, ts, val        
            
    def __init__(self, basedir, 
                 partitions_id='5k', 
                 batch_size=32, 
                 shuffle=True,
                 cache_size=100,
                 chips_basedirs=None,
                 max_chips = None,
                 selected_classids = None
                 ):
        self.basedir = basedir
        if chips_basedirs is None:
            cs = Chipset(basedir)
            self.chips_basedirs = cs.files
        else:
            self.chips_basedirs = chips_basedirs    
            
        if shuffle:
            self.chips_basedirs = np.random.permutation(self.chips_basedirs)

        if max_chips is not None:
            self.chips_basedirs = self.chips_basedirs[:max_chips]
            
        self.number_of_original_classes = self.get_number_of_classes()
        self.selected_classids = selected_classids
        if self.selected_classids is not None:
            self.selected_classids = sorted([i for i in self.selected_classids if i!=0])
            self.number_of_output_classes = len(self.selected_classids)+1
            for classid in self.selected_classids:
                if classid>=self.number_of_original_classes:
                    raise ValueError(f"class {classid} not valid. dataset contains {self.number_of_original_classes} classes")
        else:
            self.number_of_output_classes = self.number_of_original_classes

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.partitions_id = partitions_id
        self.on_epoch_end()
        self.cache = {}
        self.cache_size = cache_size
 
    
 
        print (f"got {len(self.chips_basedirs):6d} chips on {len(self)} batches. cache size is {self.cache_size}")

    def get_number_of_classes(self):
        return 12

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
            p = np.r_[[p[str(i)] if str(i) in p.keys() else 0 for i in range(self.number_of_original_classes)]]    

            # ensure 100x100 chips
            chipmean = chip.chipmean[:100,:100,:] 
            labels   = chip.label[:100,:100]

            # build output and cache it if there is space
            r = [chipmean, labels, p]
            if len(self.cache) < self.cache_size:
                self.cache[chip_filename] = r
            return r

    def __data_generation(self, chips_basedirs_temp):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 100, 100, 3), dtype=np.float32)
        labels = np.zeros((self.batch_size, 100, 100), dtype=np.int16)
        partition_proportions = np.empty((self.batch_size, self.number_of_output_classes), dtype=np.float32)

        # Generate data
        for i, chip_id in enumerate(chips_basedirs_temp):
            # Store sample
            chip_mean, label, p = self.load_chip(chip_id)
            X[i,] = chip_mean/256

            if self.selected_classids is None:
                labels[i,] = label
                partition_proportions[i,] = p
            else:
                # remap classes to selected ones, all non selected classes are mapped to 0 (background).

                # first, remap label
                labels[i,] = np.zeros_like(label)
                for idx, classid in enumerate(self.selected_classids):
                    labels[i,][label==classid] = idx+1
                # remap proportions (aggregating non selected classes into background)
                p_selected = p[self.selected_classids]
                p_background = p[[i for i in range(self.number_of_original_classes) if not i in self.selected_classids]].sum()
                partition_proportions[i,] = np.hstack([p_background, p_selected])                

        return X, (partition_proportions, labels)

    def plot_onchip_class_distributions(self):
        props = []
        for x,(p,l) in pbar(self):
            props.append(p.sum(axis=0))

        props = np.r_[props].sum(axis=0)
        props = props / sum(props)
        pd.Series(props).plot(kind='bar')
        plt.title("class distribution")
        plt.xlabel("class number")
        plt.grid();

class S2_ESAWorldCover_DataGenerator(S2LandcoverDataGenerator):

    def get_number_of_classes(self):
        return 12

class S2_EUCrop_DataGenerator(S2LandcoverDataGenerator):

    def get_number_of_classes(self):
        return 23
