from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import pickle
from rlxutils import subplots
from skimage import exposure
import glob
import rasterio as rio
from progressbar import progressbar as pbar
from shapely.geometry import Polygon, GeometryCollection
import geopandas as gpd
from rasterio.windows import Window
import os
from pyproj.crs import CRS
import matplotlib
import pandas as pd
import tensorflow as tf

lc = {'0': 'none', '1': 'water', '2': 'trees', '3': 'not used', '4': 'flooded vegetation', '5': 'crops',
      '6': 'not used', '7': 'built area', '8': 'bare ground', '9': 'snow/ice', '10': 'clouds', '11': 'rangeland'}

class Chipset:
    def __init__(self, basedir, rio_dataset_list, idfile='metadata.pkl'):

        assert sum([i.crs==rio_dataset_list[0].crs for i in rio_dataset_list[1:]]) == len(rio_dataset_list)-1, \
               "all rio datasets must be in the same crs"

        files = list(glob.glob(f'{basedir}/*/{idfile}')) +\
                list(glob.glob(f'{basedir}/*/*/{idfile}')) + \
                list(glob.glob(f'{basedir}/*/*/*/{idfile}')) +\
                list(glob.glob(f'{basedir}/*/*/*/*/{idfile}')) 
        self.basedir = basedir
        self.files = [i[:-(len(idfile)+1)] for i in files]
        self.crs = rio_dataset_list[0].crs
        self.rio_dataset_list = rio_dataset_list

    def random(self):
        return Chip(np.random.choice(self.files), self.crs)
        
    def __iter__(self):
        for f in self.files:
            yield Chip(f, self.crs)
        
    def get_geometry(self, crs=None):
        if crs is None:
            crs = self.crs
        return GeometryCollection([Chip(basedir=f, crs=crs).get_polygon() for f in pbar(self.files)])
        
    def restrict_partitions(self, partitions):
        """
        selects the geometries in 'partition' that intersect the
        convex hull of this chipset

        partition: a geopandas dataframe
        returns: a geopandas dataframe, a subset of 'partition'
        """
        p = partitions
        g = self.get_geometry(p.crs)
        gh = g.convex_hull
        return p[[i.intersects(gh) for i in p.geometry]]
    
    def load_partitions(self, partitions_file):
        self.partitions_file = partitions_file
        self.partitions_prefix = partitions_file.split("_")[0].split("/")[-1]
        self.partitions = gpd.read_file(self.partitions_file)
        self.partitions = self.restrict_partitions(self.partitions)

    def build_allchips_partitions_geojson(self, prefix):
        distribs = [c.load_label_distribution(prefix=prefix) for c in self]
        distribs = [i for i in distribs if i is not None]
        if len(distribs)>0:
            g = gpd.GeoDataFrame(distribs)
            assert g.partition_epsg.values.std()==0
            g.crs = CRS.from_epsg(g.partition_epsg.values[0])
            g.to_file(f"{self.basedir}/{prefix}_chips.geojson")
            return g    
        
    def generate_all(self):
        for c in pbar(self, max_value=len(self.files)):
            c.generate_all(rio_dataset_list=self.rio_dataset_list, partitions=self.partitions, prefix=self.partitions_prefix)

    def generate_all_label_proportions(self):
        for c in pbar(self, max_value=len(self.files)):
            c.generate_label_proportions(partitions=self.partitions, prefix=self.partitions_prefix)
            
            
    def generation_summary(self):
        with_chip = sum([c.has_chip() for c in self])
        with_chipmean = sum([c.has_chipmean() for c in self])
        print (f"total chips:       {len(self.files)}")
        print (f"with chip.npz:     {with_chip}")
        print (f"with chipmean.npz: {with_chipmean}")
        
    def remove_chipnpz(self):
        for c in pbar(self, max_value=len(self.files)):
            c.remove_chipnpz()

    def load_all_partitions(self):
        self.all_partitions = {i: gpd.read_file(f"{self.basedir}/partitions{i}_with_label_proportions.geojson") for i in ['5k', '10k', '20k', '50k']}
        self.all_chips_partitions = {i: gpd.read_file(f"{self.basedir}/chips_partitions{i}.geojson") for i in ['5k', '10k', '20k', '50k']}


class Chip:
    
    def __init__(self, basedir, crs=None, threshold=1500):        
        self.crs = crs
        if self.crs is None:
            self.crs = CRS.from_epsg(4326)
        self.basedir = basedir
        with open(f"{basedir}/metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
        self.timestaps = np.load(f"{basedir}/timestamps.npz")['arr_0']
            
        self.chipfile = f"{self.basedir}/chip.npz"
        self.meanfile = f"{self.basedir}/chip_adjusted_mean.npz"
        self.label_proportions = {}
        self.chip_mean = None
        self.chip = None
        self.labelfile = f"{self.basedir}/label.npz"
        self.threshold = threshold

    def getfiles(self):
        return [i.decode() if type(i)==bytes else i for i in os.listdir(self.basedir)]        

    def has_chip(self):
        return os.path.isfile(self.chipfile)

    def has_chipmean(self):
        return os.path.isfile(self.meanfile)

    def has_label(self):
        return os.path.isfile(self.labelfile)

    def remove_chipnpz(self):
        if self.has_chip():
            os.remove(self.chipfile)        

    def load_chip(self):
        c = np.load(self.chipfile)['arr_0']
        c = np.transpose(c, [0,2,3,1])  

        # keep timesteps with at most 10% saturated pixels
        c = c[[(c[i,:,:,0:3]>1500).mean()<0.1 for i in range(c.shape[0])]][:,:,:,:3]
        # clip and normalize to 0,1
        c[c>self.threshold] = self.threshold
        c = c/self.threshold
        # still remove timesteps with some pixel saturated
        if c.shape[0]>0:
            c = c[[np.mean(c[i,:,:]>0.99)<1e-3 for i in range(c.shape[0])]]
        self.chip=c
        
    def load_chipmean(self):
        if not 'chip_mean' in dir(self) or self.chip_mean is None:
            self.chip_mean = np.load(self.meanfile)['arr_0']

    def load_label(self):
        if not 'label' in dir(self) or self.label is None:
            self.label = np.load(self.labelfile)['arr_0']
        return self.label

    def load_label_proportions(self, partitions_id=None):
        if partitions_id is None:
            files = [i for i in self.getfiles() if i.startswith('label_proportions_')]
        else:
            files = [f"label_proportions_partitions{partitions_id}.pkl"]
        r = {}
        for file in files:
            with open(f'{self.basedir}/{file}', 'rb') as f:
                k = file.split(".")[-2][18:]
                r[k] = pickle.load(f)
        self.label_proportions = r
        return r        

    def generate_adjusted_mean(self):
        if self.has_chip():
            self.load_chip()
            if self.chip.shape[0]>0:
                self.chip_mean = self.chip.mean(axis=0)
                self.chip_mean = exposure.adjust_gamma(self.chip_mean, 1.5)
                self.save_adjusted_mean_to_npz()
                self.save_adjusted_mean_to_geotif()
            else:
                self.chip_mean = None
        else:
            self.chip_mean = None
        
    def generate_label(self, rio_dataset_list):

        for lc in rio_dataset_list:    
            p = self.get_polygon()
            minx, miny, maxx, maxy = p.bounds
            pminx, pminy = lc.index(minx, miny)
            pmaxx, pmaxy = lc.index(maxx, maxy)
            w = lc.read(1, window=Window(col_off=pminy, row_off=pmaxx, width=(pmaxy-pminy), height=np.abs(pmaxx-pminx)))
            self.label = w.astype(np.int16)
            if w.shape[0]!=0 and w.shape[1]!=0:
                self.save_label_to_npz()
                self.save_label_to_geotif()
                return
            
    def generate_label_proportions(self, partitions, prefix):
        """
        given a partition with label distributions (normally coarser than the chip grid),
        assigns the label distributions of the partition corresponding to this chip.
        if the chip falls across several partitions, we labels are weighted proportionally
        to the intersection area with each partition.
        """
        p = partitions
        pcols = [c for c in p.columns if np.char.isnumeric(str(c))]

        cpol = self.get_polygon()
        pi = p[[cpol.intersects(i) for i in p.geometry]].copy()
        pi['intersection'] = [cpol.intersection(gi).area for gi in pi.geometry]
        dicts = [{col:pii[col] for col in pcols} for _,pii in pi.iterrows()]
        weights = pi['intersection'].values
        label_distrib = weighted_dicts(dicts, weights)    
        label_distrib['geometry'] = cpol
        label_distrib['partition_epsg'] = p.crs.to_epsg()
        self.label_proportions[prefix] = label_distrib    
        with open(f"{self.basedir}/label_proportions_{prefix}.pkl", "bw") as f:
            pickle.dump(label_distrib,f)   
            
    def compute_label_proportions_on_chip_label(self):
        self.load_label()
        l = pd.Series(self.label.flatten()).value_counts() / 100**2
        return {i: (l[i] if i in l.index else 0) for i in range(12)}

    def generate_all(self, rio_dataset_list=None, partitions=None, prefix=None):
        if not 'chip' in dir(self):
            self.load_chip()
        self.generate_adjusted_mean()
        if rio_dataset_list is not None:
            self.generate_label(rio_dataset_list)
        if partitions is not None and prefix is not None:
            self.generate_label_proportions(partitions, prefix)
            
    def load_and_plot(self):

        self.load_chipmean()

        self.load_label()

        self.load_label_proportions()

        for ax,i in subplots(4, usizex=5, usizey=4):
            if i==0: 
                plt.imshow(self.chip_mean)
                plt.title("/".join(self.basedir.split("/")[-2:]))
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

    def rgb(self):
        if self.has_chip():
            self.load_chip()
            self.generate_adjusted_mean()
        elif self.has_chipmean():
            self.load_chipmean()

        if self.chip_mean is None:
            print ("no cloudless time steps in this chip")
            return
        
        if self.has_chip():
            self.load_chip()
            for ax,i in subplots(self.chip.shape[0]+2,n_cols=8, usizex=3, usizey=3):
                if i<self.chip.shape[0]:
                    plt.imshow(self.chip[i,:,:])
                    plt.title(f"saturated {np.mean(self.chip[i,:,:]>0.99):.5f}")
                elif i==self.chip.shape[0] and self.has_chipmean():
                    self.load_chipmean()
                    plt.imshow(self.chip_mean)
                    plt.title("mean adjusted")
                elif i==self.chip.shape[0]+1 and self.has_label():
                    self.load_label()
                    plt.imshow(self.label, vmin=0, vmax=20, cmap=plt.cm.tab20)
                    plt.colorbar()
                    plt.title("label")

        else:
            for ax,i in subplots(2,n_cols=8, usizex=3, usizey=3):
                if i==0 and self.has_chipmean():
                    self.load_chipmean()
                    plt.imshow(self.chip_mean)
                    plt.title("mean adjusted")
                elif i==1 and self.has_label():
                    self.load_label()
                    plt.imshow(self.label, vmin=0, vmax=20, cmap=plt.cm.tab20)
                    plt.colorbar()
                    plt.title("label")
             
    def save_adjusted_mean_to_npz(self):
        if self.chip_mean is not None and np.isnan(self.chip_mean).sum()==0:
            np.savez(self.meanfile, (self.chip_mean*255).astype(np.int16))
        
    def save_adjusted_mean_to_geotif(self):
        if self.chip_mean is None or np.isnan(self.chip_mean).sum()!=0:
            return
        
        p = self.get_polygon()
        minx, miny, maxx, maxy = p.bounds
        
        arr = (self.chip_mean*255).astype(np.int16)
        transform = rio.transform.from_origin(minx, maxy, (maxx-minx)/arr.shape[1], (maxy-miny)/arr.shape[0])

        new_dataset = rio.open(f'{self.basedir}/chip_adjusted_mean.tif', 'w', driver='GTiff',
                                    height = arr.shape[0], width = arr.shape[1],
                                    count=3, dtype=str(arr.dtype),
                                    crs=self.crs,
                                    transform=transform)

        for i in range(3):
            new_dataset.write(arr[:,:,i], i+1)
        new_dataset.close() 
        
    def save_label_to_npz(self):        
        np.savez(self.labelfile, self.label)
        
    def save_label_to_geotif(self):
        p = self.get_polygon()
        minx, miny, maxx, maxy = p.bounds
        w = self.label
        transform = rio.transform.from_origin(minx, maxy, (maxx-minx)/w.shape[1], (maxy-miny)/w.shape[0])

        new_dataset = rio.open(f'{self.basedir}/label.tif', 'w', driver='GTiff',
                                    height = w.shape[0], width = w.shape[1],
                                    count=1, dtype=str(w.dtype),
                                    crs=self.crs,
                                    transform=transform)

        new_dataset.write(w[:,:], 1)
        new_dataset.close()         
            
    def get_polygon(self):
        n,w = self.metadata['corners']['nw']
        s,e = self.metadata['corners']['se']
        p = Polygon([[w,n],[w,s],[e,s],[e,n],[w,n]])
        #p = Polygon([[n,w],[s,w],[s,e],[n,e],[n,w]])
        d = gpd.GeoDataFrame([[p]], columns=['geometry'], crs='4326')
        d = d.to_crs(self.crs)
        return d.geometry.values[0]
    
    def load_label_distribution(self, prefix=""):
        fname = f"{self.basedir}/label_proportions_{prefix}.pkl"
        if os.path.isfile(fname):
            with open(fname, "br") as f:
                return pickle.load(f)
        else:
            return None
        
    
def weighted_dicts(dicts, weights):
    all_keys = np.unique([i for j in [list(d.keys()) for d in dicts] for i in j])
    for d in dicts:
        d.update({k:0. for k in all_keys if not k in d.keys()})
    return {k: sum([w*d[k] for d,w in zip(dicts, weights)])/sum(weights) for k in all_keys}

def katana(geometry, threshold, count=0, random_variance=0.1):
    """
    splits a polygon recursively into rectangles
    geometry: the geometry to split
    threshold: approximate size of rectangles
    random_variance: 0  - try to make all rectangles of the same size
                     >0 - the greater the number, the more different the rectangle sizes
                     values between 0 and 1 seem more useful
                     
    returns: a list of Polygon or MultyPolygon objects
    """
    
    
    """Split a Polygon into two parts across it's shortest dimension"""
    assert random_variance>=0
    bounds = geometry.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    
    random_factor = 2*(1+(np.random.random()-0.5)*random_variance*2)
    
    if max(width, height) <= threshold or count == 250:
        # either the polygon is smaller than the threshold, or the maximum
        # number of recursions has been reached
        return [geometry]
    if height >= width:
        # split left to right
        a = box(bounds[0], bounds[1], bounds[2], bounds[1]+height/random_factor)
        b = box(bounds[0], bounds[1]+height/random_factor, bounds[2], bounds[3])
    else:
        # split top to bottom
        a = box(bounds[0], bounds[1], bounds[0]+width/random_factor, bounds[3])
        b = box(bounds[0]+width/random_factor, bounds[1], bounds[2], bounds[3])
    result = []
    for d in (a, b,):
        c = geometry.intersection(d)
        if not isinstance(c, GeometryCollection):
            c = [c]
        for e in c:
            if isinstance(e, (Polygon)):
                result.extend(katana(e, threshold, count+1, random_variance))
            if isinstance(e, (MultiPolygon)):
                for p in e.geoms:
                    result.extend(katana(p, threshold, count+1, random_variance))
    if count > 0:
        return result
    # convert multipart into singlepart
    final_result = []
    for g in result:
        if isinstance(g, MultiPolygon):
            final_result.extend(g)
        else:
            final_result.append(g)
    return final_result


class S2LandcoverDataGenerator(tf.keras.utils.Sequence):
    """
    from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def __init__(self, basedir, partitions_id='5k', batch_size=32, shuffle=True):
        self.basedir = basedir
        self.chips_basedirs = [f for f,subdirs,files in os.walk(basedir) if 'metadata.pkl' in files]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.partitions_id = partitions_id
        self.on_epoch_end()
        print (f"got {len(self.chips_basedirs)} chips")

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

    def __data_generation(self, chips_basedirs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 100, 100, 3), dtype=np.float32)
        label = np.empty((self.batch_size, 100, 100), dtype=np.int16)
        partition_proportions = np.empty((self.batch_size, 12), dtype=np.float32)

        # Generate data
        for i, ID in enumerate(chips_basedirs_temp):
            # Store sample
            chip = Chip(ID)
            chip.load_chipmean()
            chip.load_label()
            chip.load_label_proportions(self.partitions_id)
            p = chip.label_proportions[f'partitions{self.partitions_id}']
            p = np.r_[[p[str(i)] if str(i) in p.keys() else 0 for i in range(12)]]    
            
            X[i,] = chip.chip_mean/256
            label[i,] = chip.label
            partition_proportions[i,] = p

        return X, partition_proportions, label