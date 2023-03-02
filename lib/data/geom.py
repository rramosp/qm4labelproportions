import ee
import geemap
import shapely as sh
from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection

from . import data
from pyproj import CRS
import numpy as np
from progressbar import progressbar as pbar
import geopandas as gpd
import pandas as pd

from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from retry import retry
import hashlib
import requests
import rasterio
import rasterio.mask
import shutil
import os
import multiprocessing
from skimage import exposure
from joblib import Parallel, delayed
from rlxutils import mParallel
from skimage.io import imread
from joblib import Parallel, delayed
from rlxutils import mParallel
import re

def get_region_hash(region):
    """
    region: a shapely geometry
    returns a hash string for region using its coordinates
    """
    s = str(np.r_[region.envelope.boundary.coords].round(5))
    k = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**15
    k = str(hex(k))[2:].zfill(13)
    return k

def get_regionlist_hash(regionlist):
    """
    returns a hash string for a list of shapely geometries
    """
    s = [get_region_hash(i) for i in regionlist]
    s = " ".join(s)
    k = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**15
    k = str(hex(k))[2:].zfill(13)
    return k


def get_utm_crs(lon, lat):
    """
    returns a UTM CRS in meters with the zone corresponding to lon, lat
    """
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=lon,
            south_lat_degree=lat,
            east_lon_degree=lon,
            north_lat_degree=lat,
        ),
    )
    if len(utm_crs_list)==0:
        raise ValueError(f"could not get utm for lon/lat: {lon}, {lat}")
        
    utm_crs = CRS.from_epsg(utm_crs_list[0].code)
    return utm_crs

_gee_get_tile_progress_period = 100
def _get_tile(i,gee_tile):
    # helper function to download gee tiles
    try:
        gee_tile.get_tile()
    except Exception as e:
        print (f"\n----error----\ntile {gee_tile.identifier}\n------")
        print (e)

    if i%_gee_get_tile_progress_period==0:
        print (f"{i} ", end="", flush=True)


def align_to_lonlat(geometry):
    """
    aligns a rectangle so that sides have constant lat or lon. this is
    required by GEE, since unaligned geometries produce null pixels on the borders.
    geometry: a rectangle (shapely polygon with 5 coords) in epsg4326 
    """
    epsg4326 = CRS.from_epsg(4326)

    if not 'boundary' in dir(geometry):
        raise ValueError("can only align Polygons")

    coords = np.r_[geometry.boundary.coords]
    if len(coords)!=5:
        raise ValueError("can only align rectangles (with 5 coords)")

    clon, clat = list(geometry.centroid.coords)[0]
    utm = get_utm_crs(clon, clat)

    geometrym = gpd.GeoSeries([geometry], crs=epsg4326).to_crs(utm).values[0]
    coordsm = np.r_[geometrym.boundary.coords]

    # assume the size lengths to the the observed maximum
    sizex, sizey = np.abs((coordsm[1:]-coordsm[:-1])).max(axis=0)

    # measure how many meters per degree in lon and lat
    lon0,lat0 = list(gpd.GeoSeries([sh.geometry.Point([clon, clat])], crs=epsg4326).to_crs(utm).values[0].coords)[0]
    lon1,lat1 = list(gpd.GeoSeries([sh.geometry.Point([clon+0.001, clat])], crs=epsg4326).to_crs(utm).values[0].coords)[0]
    lon2,lat2 = list(gpd.GeoSeries([sh.geometry.Point([clon, clat+0.001])], crs=epsg4326).to_crs(utm).values[0].coords)[0]

    meters_per_degree_lon = (lon1-lon0) * 1000
    meters_per_degree_lat = (lat2-lat0) * 1000
    delta_degrees_lon =  sizex/2 / meters_per_degree_lon
    delta_degrees_lat =  sizey/2 / meters_per_degree_lat

    aligned =  sh.geometry.Polygon([[clon-delta_degrees_lon, clat-delta_degrees_lat], 
                                    [clon-delta_degrees_lon, clat+delta_degrees_lat],
                                    [clon+delta_degrees_lon, clat+delta_degrees_lat],
                                    [clon+delta_degrees_lon, clat-delta_degrees_lat],
                                    [clon-delta_degrees_lon, clat-delta_degrees_lat]])

    return aligned

class PartitionSet:
    
    def __init__(self, name, region=None, data=None):
        """
        name: a name for this partition
        region: a shapely shape in epsg 4326 lon/lat coords
        data: a geopandas dataframe with the partition list
        """
        
        assert region is not None or data is not None, "must specify either region or data"
        assert (region is not None) + (data is not None) == 1, "cannot specify both region and data"
        assert not "_" in name, "'name' cannot contain '_'"
        self.name   = name
        self.region = region
        self.data   = data
        
        if self.data is not None:
            if self.data.crs == CRS.from_epsg(4326):
                # convert to meters crs to measure areas
                lon,lat = np.r_[sh.geometry.GeometryCollection(self.data.geometry.values).envelope.boundary.coords].mean(axis=0)
                datam = self.data.to_crs(get_utm_crs(lon, lat))
                self.data['area_km2'] = [i.area/1e6 for i in datam.geometry]
            else: 
                # if we are not in epsg 4326 assume we have a crs using meters           
                self.data['area_km2'] = [i.area/1e6 for i in self.data.geometry]

            self.data = self.data.to_crs(CRS.from_epsg(4326))
            self.data['identifier'] = [get_region_hash(i) for i in self.data.geometry]

        if region is None:
            self.region = sh.ops.unary_union(self.data.geometry)
                        
        # corresponding UTM CRS in meters to this location
        lon, lat = np.r_[self.region.envelope.boundary.coords].mean(axis=0)

        self.utm_crs = get_utm_crs(lon, lat)
        self.epsg4326 = CRS.from_epsg(4326)

        # the region in UTM CRS meters
        self.region_utm = gpd.GeoDataFrame({'geometry': [self.region]}, crs = self.epsg4326).to_crs(self.utm_crs).geometry[0]
            
        self.loaded_from_file = False

    def reset_data(self):
        self.data = None
        self.loaded_from_file = False
        return self
        
    def make_random_partitions(self, max_rectangle_size, random_variance=0.1, n_jobs=5):
        """
        makes random rectangular tiles with max_rectangle_size as maximum side length expressed in meters.
        stores result as a geopandas dataframe in self.data
        """
        assert self.data is None, "cannot make partitions over existing data"
        
        # cut off region, assuming region_utm is expressed in meters
        parts = katana(self.region_utm, threshold=max_rectangle_size, random_variance=random_variance)

        # reproject to epsg 4326
        self.data = gpd.GeoDataFrame({
                                      'geometry': parts, 
                                      'area_km2': [i.area/1e6 for i in parts]
                                      },
                                    crs = self.utm_crs).to_crs(self.epsg4326)

        # align geometries to lonlat
        def f(part):
            try:
                aligned_part = align_to_lonlat(part)
            except Exception as e:
                aligned_part = part
            return aligned_part

        parts = mParallel(n_jobs=n_jobs, verbose=30)(delayed(f)(part) for part in self.data.geometry.values)
        self.data.geometry = parts

        self.data['identifier'] =  [get_region_hash(i) for i in self.data.geometry]
        return self
        
    def make_grid(self, rectangle_size):

        """
        makes a grid of squares for self.region (which must be in epsg 4326 lon/lat)
        rectangle_size: side length in meters of the resulting squares
        stores result as a geopandas dataframe in self.data

        """
        assert self.data is None, "cannot make partitions over existing data"

        coords = np.r_[self.region_utm.envelope.boundary.coords]        
        m = rectangle_size

        minlon, minlat = coords.min(axis=0)
        maxlon, maxlat = coords.max(axis=0)
        parts = []
        for slon in pbar(np.arange(minlon, maxlon, m)):
            for slat in np.arange(minlat, maxlat, m):
                p = sh.geometry.Polygon([[slon, slat], 
                                         [slon, slat+m],
                                         [slon+m, slat+m],
                                         [slon+m, slat],
                                         [slon, slat]])

                if p.intersects(self.region_utm):
                    parts.append(p.intersection(self.region_utm))
                    
        self.data = gpd.GeoDataFrame({
                                      'geometry': parts, 
                                      'area_km2': [i.area/1e6 for i in parts]
                                      },
                                    crs = self.utm_crs).to_crs(self.epsg4326)
        self.data['identifier'] =  [get_region_hash(i) for i in self.data.geometry]

        return self

    def get_gee_tiles(self, 
                      image_collection, 
                      dest_dir=".", 
                      file_prefix="geetile_", 
                      pixels_lonlat=None, 
                      meters_per_pixel=None,
                      remove_saturated_or_null = False,
                      enhance_images = None,
                      dtype = None,
                      skip_if_exists = False):
        r = []
        for g,i in zip(self.data.geometry.values, self.data.identifier.values):
            r.append(GEETile(image_collection=image_collection,
                            region = g,
                            identifier = i,
                            dest_dir = dest_dir, 
                            file_prefix = file_prefix,
                            pixels_lonlat = pixels_lonlat,
                            meters_per_pixel = meters_per_pixel,
                            remove_saturated_or_null = remove_saturated_or_null,
                            enhance_images = enhance_images,
                            dtype = dtype,
                            skip_if_exists = skip_if_exists)
                    )
        return r        
        
    def download_gee_tiles(self, 
                           image_collection, 
                           image_collection_name, 
                           n_processes=10, 
                           pixels_lonlat=None, 
                           meters_per_pixel=None,
                           remove_saturated_or_null = False,
                           enhance_images = None,
                           max_downloads=None, 
                           shuffle=True,
                           skip_if_exists = False,
                           dtype = None):

        """
        downloads in parallel tiles from GEE. See GEETile below for parameters info.
        it will use the same folder in which the partitions where saved to or loaded from as geojson.
        collection_name: a name to create a subfolder for this partitions.
        """        
        if not 'origin_file' in dir(self):
            raise ValueError("must first save this partitions with 'save_as', or load an existing partitions file with 'from_file'")

        global _gee_get_tile_progress_period
        
        dest_dir = os.path.splitext(self.origin_file)[0]+ "/" + image_collection_name

        if not skip_if_exists and os.path.exists(dest_dir):
            raise ValueError(f"destination folder {dest_dir} already exists")
        
        os.makedirs(dest_dir, exist_ok=True)

        gtiles = self.get_gee_tiles(image_collection,
                                    dest_dir = dest_dir,
                                    file_prefix = "",
                                    pixels_lonlat = pixels_lonlat, 
                                    meters_per_pixel = meters_per_pixel,
                                    remove_saturated_or_null = remove_saturated_or_null,
                                    enhance_images = enhance_images,
                                    dtype = dtype, 
                                    skip_if_exists = skip_if_exists)

        if shuffle:
            gtiles = np.random.permutation(gtiles)
        if max_downloads is not None:
            gtiles = gtiles[:max_downloads]

        print (f"downloading {len(gtiles)} tiles", flush=True)                                    
        _gee_get_tile_progress_period = np.max([len(gtiles)//100,1])
        pool = multiprocessing.Pool(n_processes)
        pool.starmap(_get_tile, enumerate(gtiles))
        pool.close()

    def get_partitions(self):
        """
        returns a list the partition objects of this partitionset 
        """
        r = [Partition(partitionset = self, 
                                        identifier = i.identifier, 
                                        geometry = i.geometry, 
                                        crs = self.data.crs) \
            for i in self.data.itertuples()]
        return r

    def save_as(self, dest_dir, partitions_name):
        """
        method used to save partitions that were just created by make_random or make_grid
        """
        if self.data is None:
            raise ValueError("there are no partitions. You must call make_random_partitions or make_grid first")

        if self.loaded_from_file:
            raise ValueError("cannot save partitions previously loaded. You must call reset_data first and create new different partitions")

        if "_" in partitions_name or "partitions" in partitions_name:
            raise ValueError("cannot have '_' or 'partitions' in partitions_name")

        h = get_regionlist_hash(self.data.geometry)
        filename = f"{dest_dir}/{self.name}_partitions_{partitions_name}_{h}.geojson"
        self.data.to_file(filename, driver='GeoJSON')
        self.origin_file = filename
        print (f"saved to {filename}")
        self.partitions_name = partitions_name
        return self      

    def save(self):
        """
        method used to save partitions previously loaded (after adding some column or metadata)
        """
        computed_hash = get_regionlist_hash(self.data.geometry)
        filename_hash = os.path.splitext(os.path.basename(self.origin_file))[0].split("_")[-1]
        if computed_hash != filename_hash:
            raise ValueError("cannot save since geometries changed, use save_as to create a new partition set")
        self.data.to_file(self.origin_file, driver='GeoJSON')
        print (f"saved to {self.origin_file}")


    def add_proportions(self, image_collection_name, n_jobs=5, transform_label_fn=lambda x: x):
        """
        adds proportions from an image collection with the same geometry (such when this partitionset
        is an rgb image collection and image_collection_name contains segmentation masks)
        """
        def f(identifier, geometry):
            proportions = Partition(partitionset = self, 
                                    identifier = identifier, 
                                    geometry = geometry, 
                                    crs = self.data.crs).compute_proportions_from_raster(
                                                                image_collection_name,
                                                                transform_label_fn = transform_label_fn,
                                                            )
            return proportions

        r = mParallel(n_jobs=n_jobs, verbose=30)(delayed(f)(i.identifier, i.geometry) for i in self.data.itertuples())
        self.data[f"{image_collection_name}_proportions"] = r

        self.save()

    def add_cross_proportions(self, image_collection_name, other_partitionset):
        """
        add class proportions of the geometries of this partitionset when embedded in a coarser partitionset.
        see Partition.compute_cross_proportions below
        """
        parts = self.get_partitions()
        proportions = []
        for part in pbar(parts):
            cross_proportions, identifier = part.compute_cross_proportions(image_collection_name, other_partitionset)
            proportions.append({'partition_id': identifier, 
                                'proportions': cross_proportions})
        self.data[f"{image_collection_name}_proportions_at_{other_partitionset.partitions_name}"] = proportions
        self.save()

    def split(self, nbands, angle, train_pct, test_pct, val_pct, split_col_name='split'):
        
        """
        splits the geometries in train, test, val by creating spatial bands
        and assigning each band to train, test or val according to the pcts specified.
        
        nbands: the number of bands
        angle: the angle with which bands are created (in [-pi/2, pi/2])
        train_pct, test_pct, val_pct: the pcts of bands for each kind,
                bands of the same kind are put together and alternating
                as much as possible.
        """
        if angle<-np.pi/2 or angle>np.pi/2:
            raise ValueError("angle must be between 0 and pi/2")
            
        p = self
        coords = np.r_[[np.r_[i.envelope.boundary.coords].mean(axis=0) for i in p.data.geometry]]

        cmin = coords.min(axis=0)
        cmax = coords.max(axis=0)
        crng = cmax - cmin

        if not np.allclose(train_pct + test_pct + val_pct, 1, atol=1e-3):
            raise ValueError("percentages must add up to one")

        min_pct = np.min([i for i in [train_pct, test_pct, val_pct] if i!=0])
        bands_train = int(np.round(train_pct/min_pct,0))
        bands_test  = int(np.round(test_pct/min_pct,0))
        bands_val   = int(np.round(val_pct/min_pct,0))

        if bands_train + bands_test + bands_val > nbands:
            raise ValueError(f"not enough bands for specified percentages. increase nbands to at least {bands_train + bands_test + bands_val}")
        
        if np.abs(angle)<np.pi/4:
            plon, plat = np.abs(angle)/(np.pi/4), 1
        else:
            plon, plat = np.sign(angle), (np.pi/2-np.abs(angle))/(np.pi/4)
        
        ncoords = (coords - cmin)/crng

        if angle<0:
            ncoords = 1-ncoords
        
        # find the factor that matches the desired number of bands
        for k in np.linspace(0.1,50,10000):
            band_id = ((plon*ncoords[:,0] + plat*ncoords[:,1])/(k/nbands)).astype(int)
            band_id = band_id - np.min(band_id)
            if len(np.unique(band_id))==nbands:
                break

        bands_ids = np.sort(np.unique(band_id))

        splits = ['train']*bands_train + ['test']*bands_test + ['val']*bands_val
        splits = (splits * (len(bands_ids)//len(splits) + 1))[:len(bands_ids)]

        band_split_map = {band_id: split for band_id, split in zip(bands_ids, splits)}

        split = [band_split_map[i] for i in band_id]

        self.data[split_col_name] = split

        
    def split_per_partitions(self, nbands, angle, train_pct, test_pct, val_pct, image_collection_name, other_partitions_id):
        """
        splits the geometries (as in 'split'), but modifies the result keeping together 
        in the same split all geometries within the same partition.
        
        must have previously run 'add_cross_proportions'.
        """
        self.split(nbands=nbands, angle=angle, 
                train_pct=train_pct, 
                test_pct=test_pct, 
                val_pct=val_pct, split_col_name='tmp_split')
        
        # gather the id of the other partitions
        self.data['tmp_partition_id'] =  [i['partition_id'] \
                                        for i in self.data[f"{image_collection_name}_proportions_at_{other_partitions_id}"].values]

        # group the splits and get the most frequent one
        self.data[f'split_{other_partitions_id}'] = self.data.groupby('tmp_partition_id')[['tmp_split']]\
                                                    .transform(lambda x: pd.Series(x).value_counts().index[0])
        
        self.data.drop('tmp_partition_id', axis=1, inplace=True)
        self.data.drop('tmp_split', axis=1, inplace=True)

    def save_splits(self):
        # save the split into a separate file for fast access
        fname = os.path.splitext(self.origin_file)[0] + "_splits.csv"
        splits_df = self.data[[c for c in self.data.columns if ('split' in c and c!='split_nb') or c=='identifier']]
        splits_df.to_csv(fname, index=False)
        print (f"all splits saved to {fname}")

        self.save()

    @classmethod
    def from_file(cls, filename):
        data = gpd.read_file(filename)
        r = cls("fromfile", data=data)
        r.origin_file = filename
        pname = re.search('_partitions_(.+?)_', filename)
        if pname is None:
            r.partitions_name = None
        else:
            r.partitions_name = pname.group(1)

        return r          


class Partition:
    
    def __init__(self, partitionset, identifier, geometry, crs):
        self.identifier = identifier
        self.geometry = geometry
        self.crs = crs
        self.partitionset = partitionset
        self.partitionset_dir = os.path.splitext(self.partitionset.origin_file)[0]
        
    def compute_proportions_from_raster(self, image_collection_name, transform_label_fn=lambda x: x):
        basedir = self.partitionset_dir + "/" + image_collection_name
        filename = f"{basedir}/{self.identifier}.tif"
        img = imread(filename)
        r = {transform_label_fn(k):v for k,v in zip(*np.unique(img, return_counts=True))}        
        total = sum(r.values())
        r = {k:v/total for k,v in r.items()}
        return r
    
    def compute_cross_proportions(self, image_collection_name, other_partitionset):
        """
        compute class proportions of this geometry when embedded in a coarser partitionset.
        class proportions are computed by (1) obtaining the intersecting partitions on the
        other partitionset, (2) combining the proportions in the intersecting partitions by
        weighting them according to the intersection area with this geometry
        
        returns: a list of proportions, the id of the geometry in "other_partitionset" with greater contribution
        """
        t = other_partitionset
        relevant = t.data[[i.intersects(self.geometry) for i in t.data.geometry.values]]

        # weight each higher grained geometry by % of intersection with this geometry
        w = np.r_[[self.geometry.intersection(i).area for i in relevant.geometry]]
        w = w / w.sum()
        cross_proportions = dict ((pd.DataFrame(list(relevant[f"{image_collection_name}_proportions"].values)) * w.reshape(-1,1) ).sum(axis=0))

        if len(w)>0:
            largest_other_partition_id = relevant.identifier.values[np.argmax(w)]
        else:
            largest_other_partition_id = -1

        return cross_proportions, largest_other_partition_id

    def compute_proportions_by_interesection(self, other_partitions):
        pass        

class GEETile:
    
    def __init__(self, 
                 region, 
                 image_collection, 
                 dest_dir=".", 
                 file_prefix="geetile_", 
                 meters_per_pixel=None, 
                 pixels_lonlat=None, 
                 identifier=None,
                 remove_saturated_or_null = False,
                 enhance_images = None,
                 dtype = None,
                 skip_if_exists = True):
        """
        region: shapely geometry in epsg 4326 lon/lat
        dest_dir: folder to store downloaded tile from GEE
        file_prefix: to name the tif file with the downloaded tile
        meters_per_pixel: an int, if set, the tile pixel size will be computed to match the requested meters per pixel
        pixels_lonlat: a tuple, if set, the tile will have this exact size in pixels, regardless the physical size.
        image_collection: an instance of ee.ImageCollection
        remove_saturated_or_null: if true, will remove image if saturated or null > 1%
        enhance_images: operation to enhance images
        """
        if not enhance_images in [None, 'none', 'gamma']:
            raise ValueError(f"'enhace_images' value '{enhance_images}' not allowed")

        if sum([(meters_per_pixel is None), (pixels_lonlat is None)])!=1: 
            raise ValueError("must specify exactly one of meters_per_pixel or pixels_lonlat")
            
        self.region = region
        self.meters_per_pixel = meters_per_pixel
        self.image_collection = image_collection
        self.dest_dir = dest_dir
        self.file_prefix = file_prefix
        self.remove_saturated_or_null = remove_saturated_or_null
        self.enhance_images = enhance_images
        self.dtype = dtype
        self.skip_if_exists = skip_if_exists


        if identifier is None:
            self.identifier = get_region_hash(self.region)
        else:
            self.identifier = identifier
                    
        if pixels_lonlat is not None:
            self.pixels_lon, self.pixels_lat = pixels_lonlat



    @retry(tries=10, delay=1, backoff=2)
    def get_tile(self):

        # check if should skip
        ext = 'tif'
        outdir = os.path.abspath(self.dest_dir)
        filename    = f"{outdir}/{self.file_prefix}{self.identifier}.{ext}"

        if self.skip_if_exists and os.path.exists(filename):
            return


        # get appropriate utm crs for this region to measure stuff in meters 
        lon, lat = list(self.region.envelope.boundary.coords)[0]
        utm_crs = get_utm_crs(lon, lat)
        self.region_utm = gpd.GeoDataFrame({'geometry': [self.region]}, crs = CRS.from_epsg(4326)).to_crs(utm_crs).geometry[0]

        # compute image pixel size if meters per pixels where specified
        if self.meters_per_pixel is not None:
            coords = np.r_[self.region_utm.envelope.boundary.coords]
            self.pixels_lon, self.pixels_lat = np.ceil(np.abs(coords[1:] - coords[:-1]).max(axis=0) / self.meters_per_pixel).astype(int)

        # build image request
        dims = f"{self.pixels_lon}x{self.pixels_lat}"

        try:
            rectangle = ee.Geometry.Polygon(list(self.region.boundary.coords)) 
        except:
            # in case multipolygon, or mutipart or other shapely geometries without a boundary
            rectangle = ee.Geometry.Polygon(list(self.region.envelope.boundary.coords)) 

        url = self.image_collection.getDownloadURL(
            {
                'region': rectangle,
                'dimensions': dims,
                'format': 'GEO_TIFF',
            }
        )

        # download and save to tiff
        r = requests.get(url, stream=True)

        if r.status_code != 200:
            r.raise_for_status()

        with open(filename, 'wb') as outfile:
            shutil.copyfileobj(r.raw, outfile)

        # reopen tiff to mask out region and set image type
        with rasterio.open(filename) as src:
            out_image, out_transform = rasterio.mask.mask(src, [self.region], crop=True)
            out_meta = src.meta

        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        if self.dtype is not None:
            out_image = out_image.astype(self.dtype)
            out_meta['dtype'] = self.dtype

        with rasterio.open(filename, "w", **out_meta) as dest:
            dest.write(out_image)  

        # open raster again to adjust and check saturation and invalid pixels
        with rasterio.open(filename) as src:
            x = src.read()
            x = np.transpose(x, [1,2,0])
            m = src.read_masks()
            profile = src.profile.copy()


        # enhance image
        if self.enhance_images=='gamma':
            x = exposure.adjust_gamma(x, gamma=.8, gain=1.2)

        # if more than 1% pixels saturated or invalid then remove this file
        if self.remove_saturated_or_null and \
            (np.mean(x>250)>0.01 or np.mean(m==0)>0.01):
            os.remove(filename)
        else:
            # write enhanced image
            with rasterio.open(filename, 'w', **profile) as dest:
                for i in range(src.count):
                    dest.write(x[:,:,i], i+1)      

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

# ---- old stuff ----

def flatten_geom(geom):
    
    """
    recursively converts a MultiPolygon into a list of shapely shapes
    geom: a shapely geometry
    returns: a list of geometries 
            (if 'geom' is not a multipart geometry it returns a list containing only geom)
    """
    
    if isinstance(geom, list):
        geoms = geom
    elif 'geoms' in dir(geom):
        geoms = geom.geoms
    else:
        return [geom]
        
    r = []
    for g in geoms:
        r.append(flatten_geom(g))
        
    r = [i for j in r for i in j]

    return r

def change_crs(shapes, to_crs, from_crs=CRS.from_epsg(4326)):
    """
    shapes: a shapely shape or a list of shapely shapes
    from_crs: pyproj CRS object representing the CRS in which geometries are expressed
    to_crs: pyproj CRS object with the target crs 

    returns: a GeometryCollection if 'shapes' is a shapely multi geometry 
             a list of shapes if 'shapes' is a list of shapely shapes
             a shapely shape if 'shapes' is a shapely shape
    """

    if 'geoms' in dir(shapes):
        r = gpd.GeoDataFrame({'geometry': list(shapes.geoms)}, crs=from_crs).to_crs(to_crs)        
    elif isinstance (shapes, list):
        r = gpd.GeoDataFrame({'geometry': shapes}, crs=from_crs).to_crs(to_crs)        
    else:
        r = gpd.GeoDataFrame([shapes], columns=['geometry'], crs=from_crs).to_crs(to_crs)        
        
    r = list(r.to_crs(to_crs).geometry.values)
    
    if 'geoms' in dir(shapes):
        r = sh.geometry.GeometryCollection(r)
    elif isinstance (shapes, list):
        pass
    else:
        r = r[0]

    return r

def makegrid(shape, size_meters, crs=CRS.from_epsg(4326)):
    
    """
    makes a grid of squares for a given geometry
    
    geom: a shapely geometry
    size_meters: side length in meters of the resulting squares
    crs: the crs in which geom is expressed
    
    returns: a list of square shapely polygons expressed in crs, covering geom
    
    """

    # get appropriate meters utm crs to measure stuff in meters
    lon, lat = list(shape.centroid.coords)[0]
    utm_crs = get_utm_crs(lon, lat)

    # reproject to utm crs
    mgeom = change_crs(shape, from_crs=crs, to_crs=utm_crs)
    mgeom = flatten_geom(mgeom)
    coords = np.vstack([list(i.exterior.coords) for i in mgeom])
    m = size_meters
    
    mgeom = sh.geometry.GeometryCollection(mgeom)
    minlon, minlat = coords.min(axis=0)
    maxlon, maxlat = coords.max(axis=0)
    mgrid = []
    for slon in pbar(np.arange(minlon, maxlon, m)):
        for slat in np.arange(minlat, maxlat, m):
            p = sh.geometry.Polygon([[slon, slat], 
                                     [slon, slat+m],
                                     [slon+m, slat+m],
                                     [slon+m, slat],
                                     [slon, slat]])
            
            if p.intersects(mgeom):
                mgrid.append(p)

    # reproject back to original crs
    r = change_crs(mgrid, from_crs=utm_crs, to_crs=crs)
    return r

