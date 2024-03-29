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
import hashlib
from itertools import islice

from .chipset import *

def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch

def gethash(s: str):
    k = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**15
    k = str(hex(k))[2:].zfill(13)
    return k

class GeoDataLoader(tf.keras.utils.Sequence):
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
                   class_groups = None,
                   partitions_id = 'aschip',
                   batch_size = 32,
                   shuffle = True,
                   ):
        assert basedir is not None, "must set 'basedir'"
        assert np.abs(train_size + test_size + val_size - 1) < 1e-7

        np.random.seed(10)

        if 'data' in os.listdir(basedir):
            datadir = f"{basedir}/data"
        else:
            datadir = basedir

        cs = Chipset(datadir)
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
                 class_groups = class_groups,
                 batch_size = batch_size, 
                 shuffle = shuffle,
                 max_chips = max_chips
                 )
        ts = cls(basedir=basedir, 
                 chips_basedirs=ts, 
                 cache_size = ts_cache_size, 
                 partitions_id = partitions_id, 
                 class_groups = class_groups,
                 batch_size = batch_size, 
                 shuffle = shuffle,
                 max_chips = max_chips
                 )
        val = cls(basedir=basedir, 
                  chips_basedirs=val, 
                  cache_size = val_cache_size, 
                  partitions_id = partitions_id, 
                  class_groups = class_groups,
                  batch_size = batch_size, 
                  shuffle = shuffle,
                  max_chips = max_chips
                  )

        return tr, ts, val

    @classmethod
    def split_per_partition(cls, 
                            basedir = None, 
                            partitions_id = None, 
                            cache_size = 1000, 
                            max_chips = None,
                            class_groups = None,
                            batch_size = 32,
                            shuffle = True,
                            ):
        """
        creates three data loaders according to splits as defined in split_file
        """
        assert basedir is not None, "must set 'basedir'"
        assert partitions_id is not None, "must set 'partitions_id'"
        
        if basedir.endswith("/data"):
            basedir = basedir[:-5]

        split_file = f"{basedir}/splits.csv"

        # read split file, and data files
        splits = pd.read_csv(split_file)
        datadir = os.path.dirname(split_file)+"/data"
        files = os.listdir(datadir)

        print ("got", len(files), "chips in total")
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
                 class_groups = class_groups,
                 batch_size = batch_size, 
                 shuffle = shuffle,
                 max_chips = max_chips,
                 save_partitions_map = True
                 )
        ts = cls(basedir=datadir, 
                 chips_basedirs=split_files['test'], 
                 cache_size = ts_cache_size, 
                 partitions_id = partitions_id, 
                 class_groups = class_groups,
                 batch_size = batch_size, 
                 shuffle = shuffle,
                 max_chips = max_chips,
                 save_partitions_map = True
                 )
        val = cls(basedir=datadir, 
                  chips_basedirs=split_files['val'], 
                  cache_size = val_cache_size, 
                  partitions_id = partitions_id, 
                  class_groups = class_groups,
                  batch_size = batch_size, 
                  shuffle = shuffle,
                  max_chips = max_chips,
                  save_partitions_map = True
                  )
        
        return tr, ts, val        
            
    def __init__(self, basedir, 
                 partitions_id='aschip', 
                 batch_size=32, 
                 shuffle=True,
                 cache_size=100,
                 chips_basedirs=None,
                 max_chips = None,
                 class_groups = None,
                 save_partitions_map = False
                 ):

        """
        class_groups: see Chip.group_classes
        batch_size: can use 'per_partition' instead of int to make batches with chips
                    falling in the same partition. in this case, you can use 
                    'per_partition:max_batch_size=16' to limit the size of batches.
                    larger batches will be split into smaller ones with max_batch_size.

        """

        if not isinstance(batch_size, int) and not batch_size.startswith('per_partition'):
            raise ValueError("batch_size must be int or the string 'per_partition'")

        if not isinstance(batch_size, int) and batch_size.startswith('per_partition') and partitions_id=='aschip':
            raise ValueError("cannot have batch_size 'per_partition' and partitions_id 'aschip'")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.partitions_id = partitions_id
        self.cache = {}
        self.cache_size = cache_size

        if 'data' in os.listdir(basedir):
            basedir = f"{basedir}/data"

        if class_groups is not None:
            # check for zeros
            for g in class_groups:
                if not isinstance(g, int) and not isinstance(g, tuple):
                    raise ValueError("groups must be integer (single classes) or tuples")

                if g==0 or (isinstance(g, tuple) and 0 in g):
                    raise ValueError("cannot include class 0 in any group, since it is the default class")

        self.basedir = basedir
        if chips_basedirs is None:
            cs = Chipset(self.basedir)
            self.chips_basedirs = cs.files
        else:
            self.chips_basedirs = chips_basedirs    

        self.save_partitions_map = save_partitions_map 
        self.hashcode = gethash("".join(np.sort(self.chips_basedirs)))

        if shuffle:
            self.chips_basedirs = np.random.permutation(self.chips_basedirs)

        if max_chips is not None:
            self.chips_basedirs = self.chips_basedirs[:max_chips]
            
        self.number_of_input_classes = self.get_number_of_input_classes()
        self.class_groups = class_groups
        if self.class_groups is None:
            self.number_of_output_classes = self.number_of_input_classes    
        else:
            self.number_of_output_classes = len(self.class_groups)+1

        self.max_batch_size = None
        if not isinstance(self.batch_size, int) and self.batch_size.startswith('per_partition'):

            if self.batch_size!='per_partition' and not self.batch_size.startswith('per_partition:max_batch_size='):
                raise ValueError(f"batch size '{batch_size}' invalid")
            
            if self.batch_size.startswith('per_partition:max_batch_size='):
                self.max_batch_size = int(batch_size.split("=")[-1])

            self.load_partitions_map()
            info = ". loaded partitions map"
        else:
            info = ""

        print (f"got {len(self.chips_basedirs):6d} chips on {len(self)} batches. cache size is {self.cache_size}{info}")

        self.on_epoch_end()

        

    def load_partitions_map(self):
        partitions_map_file = f'{self.basedir}/../partitions_map_{self.hashcode}.csv'    
        if os.path.isfile(partitions_map_file) and self.save_partitions_map:
            r = pd.read_csv(partitions_map_file)
        else:
            r = []
            for chip_basedir in self.chips_basedirs:
                c = Chip(f"{self.basedir}/{chip_basedir}")
                cr = {k:v['partition_id'] for k,v in c.data['label_proportions'].items() if 'partition_id' in v.keys()}
                cr['chip_id'] = c.data['chip_id']
                r.append(cr)

            r = pd.DataFrame(r)   
            if self.save_partitions_map:    
                r.to_csv(partitions_map_file, index=False)
            
        colname = f'partitions_{self.partitions_id}'
        self.partitions_map = r
        self.partitions_batches = self.partitions_map.groupby(colname)[['chip_id']].agg(lambda x: list(x))
        self.partitions_batches[colname] = self.partitions_batches.index
        self.partitions_batches.index = range(len(self.partitions_batches))

        # split batches if max_batch_size was specified
        if self.max_batch_size is not None:
            indexes_to_delete = []
            rows_to_add = []

            for idx,t in self.partitions_batches.iterrows():
                if len(t.chip_id)>self.max_batch_size:
                    for s in batched(t.chip_id, self.max_batch_size):
                        r = t.copy()
                        r['chip_id'] = list(s)
                        rows_to_add.append(r)
                    indexes_to_delete.append(idx)

            r = pd.concat([self.partitions_batches.drop(indexes_to_delete),
                        pd.DataFrame(rows_to_add)])
            r.index = range(len(r))
            self.partitions_batches = r


    def get_number_of_input_classes(self):
        return 12

    def empty_cache(self):
        self.cache = {}

    def __len__(self):
        'Denotes the number of batches per epoch'
        if not isinstance(self.batch_size, int) and self.batch_size.startswith('per_partition'):
            return len(self.partitions_batches)
        else:
            return int(np.floor(len(self.chips_basedirs) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if not isinstance(self.batch_size, int) and self.batch_size.startswith('per_partition'):
            self.indexes = np.arange(len(self.partitions_batches))
        else:
            self.indexes = np.arange(len(self.chips_basedirs))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        if not isinstance(self.batch_size, int) and self.batch_size.startswith('per_partition'):
            batch_chips_basedirs = self.partitions_batches.iloc[index].chip_id

        else:
            # Generate indexes of the batch
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

            # Find list of IDs
            batch_chips_basedirs = [self.chips_basedirs[k] for k in indexes]

        # Generate data
        x = self.__data_generation(batch_chips_basedirs)

        return x

    def load_chip(self, chip_filename):
        if chip_filename in self.cache.keys():
            return self.cache[chip_filename]
        else:
            chip = Chip(f"{self.basedir}/{chip_filename}", number_of_classes=self.number_of_input_classes)

            # map classes if required
            if self.class_groups is not None:
                chip = chip.group_classes(self.class_groups)

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

            p = np.r_[[p[str(i)] if str(i) in p.keys() else 0 for i in range(self.number_of_output_classes)]]    

            # ensure 100x100 chips
            chipmean = chip.chipmean[:100,:100,:] 
            labels   = chip.label[:100,:100]

            # build output and cache it if there is space
            r = [chipmean, labels, p]
            if len(self.cache) < self.cache_size:
                self.cache[chip_filename] = r
            return r

    def __data_generation(self, batch_chips_basedirs):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((len(batch_chips_basedirs), 100, 100, 3), dtype=np.float32)
        labels = np.zeros((len(batch_chips_basedirs), 100, 100), dtype=np.int16)
        partition_proportions = np.empty((len(batch_chips_basedirs), self.number_of_output_classes), dtype=np.float32)

        # Assemble data in a batch
        for i, chip_id in enumerate(batch_chips_basedirs):
            chip_mean, label, p = self.load_chip(chip_id)
            X[i,] = chip_mean/256
            labels[i,] = label
            partition_proportions[i,] = p

        return X, (partition_proportions, labels)

    def get_partition_ids(self):
        return Chip(f"{self.basedir}/{self.chips_basedirs[0]}").get_partition_ids()

    def get_class_distribution(self):
        props = []
        for x,(p,l) in pbar(self):
            props.append(p.sum(axis=0))

        props = np.r_[props].sum(axis=0)
        props = props / sum(props)
        return props

    def plot_class_distribution(self):
        props = self.get_class_distribution()
        pd.Series(props).plot(kind='bar')
        plt.title("class distribution")
        plt.xlabel("class number")
        plt.grid()

    def plot_chips(self, chip_numbers=None):
        """
        chip_numbers: a list with chip numbers (from 0 to len(self))
                      or an int with a number of random chips to be selected
                      or None, to select 3 random chips
        """
        if chip_numbers is None:
            chip_ids = np.random.permutation(self.chips_basedirs)[:3]
        elif isinstance(chip_numbers, int):
            chip_ids = np.random.permutation(self.chips_basedirs)[:chip_numbers]
        else:
            chip_ids = [self.chips_basedirs[i] for i in chip_numbers]
        
        print (chip_ids)
        for chip_id in chip_ids:
            Chip(f"{self.basedir}/{chip_id}").plot()


    def plot_sample_batch(self, batch_idx=None, return_fig = False):
        if batch_idx is None:
            batch_idx = np.random.randint(len(self))
            
        x, (p, l)  = self[batch_idx]
        n = len(x)
        nc = self.number_of_output_classes
        partitions_ids = self.get_partition_ids()
        npts = len(partitions_ids)
        
        cmap=matplotlib.colors.ListedColormap([plt.cm.gist_ncar(i/self.number_of_output_classes) \
                                                for i in range(self.number_of_output_classes)])
        n_rows = 3
        
        for ax, ti in subplots(range(n*n_rows), n_cols=n, usizex=3.5):
            i = ti % n
            row = ti // n
            
            if row==0:
                plt.imshow(x[i])        
                if i==0: 
                    plt.ylabel("input rgb")

            if row==1:
                plt.imshow(l[i], vmin=0, vmax=self.number_of_output_classes, cmap=cmap, interpolation='none')
                if i==0: 
                    plt.ylabel("labels")
                    
                cbar = plt.colorbar(ax=ax, ticks=range(self.number_of_output_classes))
                cbar.ax.set_yticklabels([f"{i}" for i in range(self.number_of_output_classes)])  # vertically oriented colorbar
                                    
            if row==2:
                partitions_id_backup = self.partitions_id
                for k, partitions_id in enumerate(partitions_ids):
                    self.empty_cache()
                    self.partitions_id = partitions_id.split("_")[-1]
                    x, (p, l)  = self[batch_idx]                                        
                    plt.bar(np.arange(nc)+k*0.5/npts - 0.5/npts/2, p[i], 0.5/npts, label=partitions_id, alpha=.5)
                self.partitions_id = partitions_id_backup
                if i in [0, n//2, n-1]:
                    plt.legend()
                plt.grid();
                plt.xticks(np.arange(nc), np.arange(nc), rotation='vertical' if nc>15 else None);
                plt.xlabel("class number")
                plt.ylim(0,1.05)
                plt.ylabel("proportions")
                
        if return_fig:
            fig = plt.gcf()
            plt.close(fig)
            return fig


class S2_ESAWorldCover_DataLoader(GeoDataLoader):

    @classmethod
    def split_per_partition(cls, **kwargs):
        """
        added to that instrospection on this method refers to this class (and not to the parent)
        """
        return super().split_per_partition(**kwargs)

    def get_number_of_input_classes(self):
        return 12

    def get_class_names(self):
        return {'0': 'none', '1': 'water', '2': 'trees', '3': 'not used', '4': 'flooded vegetation', '5': 'crops',
                '6': 'not used', '7': 'built area', '8': 'bare ground', '9': 'snow/ice', '10': 'clouds', '11': 'rangeland'}

class S2_EUCrop_DataLoader(GeoDataLoader):

    @classmethod
    def split_per_partition(cls, **kwargs):
        """
        added to that instrospection on this method refers to this class (and not to the parent)
        """
        return super().split_per_partition(**kwargs)

    def get_number_of_input_classes(self):
        return 23
