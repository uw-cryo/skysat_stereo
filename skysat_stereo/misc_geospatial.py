#! /usr/bin/env python
import matplotlib
matplotlib.use('Agg')

import os,sys,glob
import numpy as np
import pandas as pd
import geopandas as gpd
from imview import pltlib
import matplotlib.pyplot as plt
from pygeotools.lib import iolib,geolib,warplib,malib
from demcoreg import dem_mask

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

def clip_raster_by_shp_disk(r_fn,shp_fn,extent='raster',invert=False,out_fn=None):
    """
    # this is a lightweight version of directly being used from https://github.com/dshean/pygeotools/blob/master/pygeotools/clip_raster_by_shp.py
    # meant to limit subprocess calls
    """
    if not os.path.exists(r_fn):
        sys.exit("Unable to find r_fn: %s" % r_fn)
    if not os.path.exists(shp_fn):
        sys.exit("Unable to find shp_fn: %s" % shp_fn)
     #Do the clipping
    r, r_ds = geolib.raster_shpclip(r_fn, shp_fn,extent=extent,invert=invert)
    if not out_fn:
        out_fn = os.path.splitext(r_fn)[0]+'_shpclip.tif'
    iolib.writeGTiff(r, out_fn, r_ds)

def ndvtrim_function(src_fn):
    """ 
    # this is a direct port from https://github.com/dshean/pygeotools/blob/master/pygeotools/trim_ndv.py
    # intended to make it a function
    """
    if not iolib.fn_check(src_fn):
        sys.exit("Unable to find src_fn: %s" % src_fn)
    
    #This is a wrapper around gdal.Open()
    src_ds = iolib.fn_getds(src_fn)
    src_gt = src_ds.GetGeoTransform()

    print("Loading input raster into masked array")
    bma = iolib.ds_getma(src_ds)

    print("Computing min/max indices for mask")
    edge_env = malib.edgefind2(bma, intround=True)

    print("Updating output geotransform")
    out_gt = list(src_gt)
    #This should be OK, as edge_env values are integer multiples, and the initial gt values are upper left pixel corner
    #Update UL_X
    out_gt[0] = src_gt[0] + src_gt[1]*edge_env[2]
    #Update UL_Y, note src_gt[5] is negative
    out_gt[3] = src_gt[3] + src_gt[5]*edge_env[0]
    out_gt = tuple(out_gt)


    out_fn = os.path.splitext(src_fn)[0]+'_trim.tif'
    print("Writing out: %s" % out_fn)
    #Extract valid subsection from input array
    #indices+1 are necessary to include valid row/col on right and bottom edges
    iolib.writeGTiff(bma[edge_env[0]:edge_env[1]+1, edge_env[2]:edge_env[3]+1], out_fn, src_ds, gt=out_gt)
    bma = None

def dem_mask_disk(mask_list,dem_fn):
    """
    This is lightweight version ported from here for convinence: https://github.com/dshean/demcoreg/blob/master/demcoreg/dem_mask.py
    """
    dem_ds = iolib.fn_getds(dem_fn)
    print(dem_fn)
    #Get DEM masked array
    dem = iolib.ds_getma(dem_ds)
    print("%i valid pixels in original input tif" % dem.count())
    newmask = dem_mask.get_mask(dem_ds,mask_list,dem_fn=dem_fn)
    #Apply mask to original DEM - use these surfaces for co-registration
    newdem = np.ma.array(dem, mask=newmask)
    #Check that we have enough pixels, good distribution
    min_validpx_count = 100
    min_validpx_std = 10
    validpx_count = newdem.count()
    validpx_std = newdem.std()
    print("%i valid pixels in masked output tif to be used as ref" % validpx_count)
    print("%0.2f std in masked output tif to be used as ref" % validpx_std)
    #if (validpx_count > min_validpx_count) and (validpx_std > min_validpx_std):
    if (validpx_count > min_validpx_count):
        out_fn = os.path.splitext(dem_fn)[0]+"_ref.tif"
        iolib.writeGTiff(newdem, out_fn, src_ds=dem_ds)
    else:
        print("Not enough valid pixels!")
    
