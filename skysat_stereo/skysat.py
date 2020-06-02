#! /usr/bin/env python

from pygeotools.lib import geolib,iolib
from shapely.geometry import Polygon, point
import geopandas as gpd


def skysat_footprint(img_fn,incrs=None):
    if os.path.islink(img_fn):
        img_ds = iolib.fn_getds(os.readlink(img_fn))
    else:
        img_ds = iolib.fn_getds(img_fn)
    nx = img_ds.RasterXSize
    ny = img_ds.RasterYSize
    #img_coord (0,0), (nx,0), (nx,ny), (0,ny) correspond to ul,ur,lr,ll
    z = np.float(img_ds.GetMetadata('RPC')['HEIGHT_OFF'])
    #z = np.float(ht.split(' ',1)[1].splitlines()[0])
    img_x = [0,nx,nx,0]
    img_y = [0,0,ny,ny]   
    img_z = [z,z,z,z] #should ideally accept a common height above datum, read from rpc #done
    mx,my = rpc2map(img_fn,img_x,img_y,img_z)
    coord_list = list(zip(mx,my))
    footprint_poly = Polygon(coord_list)
    footprint_shp = gpd.GeoDataFrame(index=[0],geometry=[footprint_poly],crs=geo_crs)
    if incrs:
        footprint_shp = footprint_shp.to_crs(incrs)
    return footprint_shp

