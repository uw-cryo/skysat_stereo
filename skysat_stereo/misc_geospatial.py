#! /usr/bin/env python
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import geopandas as gpd
from imview import pltlib
import matplotlib.pyplot as plt
from pygeotools.lib import iolib,geolib,warplib

def shp_merger(shplist):
    """
    merge multiple geopandas shapefiles into 1 multi-row shapefile
    Parameters
    ------------
    shplist: list
        list of geopandas shapefiles
    Returns
    ------------
    gpd_merged: geopandas geodataframe
        merged multirow shapefile
    """
    #Taken from here: "https://stackoverflow.com/questions/48874113/concat-multiple-shapefiles-via-geopandas"
    gpd_merged = pd.concat([shp for shp in shplist]).pipe(gpd.GeoDataFrame)
    return gpd_merged

def plot_composite_fig(ortho,dem,count,nmad,outfn,product='triplet'):
    """
    Plot the gallery figure for final DEM products
    Parameters
    ------------
    ortho: str
        path to orthoimage
    dem: str
        path to dem
    count: str
        path to count map
    nmad: str
        path to NMAD
    outfn: str
        path to save output figure
    ortho: str
        product to plot (triplet/video)
    """
    if product == 'triplet':
        figsize=(10,8)
    else:
        figsize=(10,3)
    f,ax = plt.subplots(1,4,figsize=figsize)
    ds_list = warplib.memwarp_multi_fn([ortho,dem,count,nmad],res='max')
    ortho,dem,count,nmad = [iolib.ds_getma(x) for x in ds_list]
    pltlib.iv(ortho,ax=ax[0],cmap='gray',scalebar=True,cbar=False,ds=ds_list[0],skinny=False)
    pltlib.iv(dem,ax=ax[1],hillshade=True,scalebar=False,ds=ds_list[1],label='Elevation (m WGS84)',skinny=False)
    pltlib.iv(count,ax=ax[2],cmap='YlOrRd',label='DEM count',skinny=False)
    pltlib.iv(nmad,ax=ax[3],cmap='inferno',clim=(0,10),label='Elevation NMAD (m)',skinny=False)
    plt.tight_layout()
    f.savefig(outfn,dpi=300,bbox_inches='tight',pad_inches=0.1)

def clip_raster_by_shp_disk(r_fn,shp_fn,invert=False):
    """
    # this is a lightweight version of directly being used from https://github.com/dshean/pygeotools/blob/master/pygeotools/clip_raster_by_shp.py
    # meant to limit subprocess calls
    """
    if not os.path.exists(r_fn):
        sys.exit("Unable to find r_fn: %s" % r_fn)
    if not os.path.exists(shp_fn):
        sys.exit("Unable to find shp_fn: %s" % shp_fn)
     #Do the clipping
    r, r_ds = geolib.raster_shpclip(r_fn, shp_fn, invert)
    out_fn = os.path.splitext(r_fn)[0]+'_shpclip.tif'
    iolib.writeGTiff(r, out_fn, r_ds)


