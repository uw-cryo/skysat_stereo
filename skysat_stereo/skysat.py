#! /usr/bin/env python

from pygeotools.lib import geolib,iolib
from shapely.geometry import Polygon, point
import geopandas as gpd
from skysat_stereo.lib import asp_utils


def skysat_footprint(img_fn,incrs=None):
    """
    Define ground corner footprint from RPC model
    Parameters
    ----------
    img_fn: str
        path to image with embedded RPC info in tiff tag
    incrs: dict
        crs to convert the final footprint into, by default the footprint is returned in geographic coordinates (EPSG:4326)
    Returns
    ----------
    footprint_shp: geopandas geodataframe
        geodataframe containg ground footprints in specified incrs
    """
    
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
    mx,my = asp_utils.rpc2map(img_fn,img_x,img_y,img_z)
    coord_list = list(zip(mx,my))
    footprint_poly = Polygon(coord_list)
    footprint_shp = gpd.GeoDataFrame(index=[0],geometry=[footprint_poly],crs=geo_crs)
    if incrs:
        footprint_shp = footprint_shp.to_crs(incrs)
    return footprint_shp

def parse_frame_index(frame_index,df_only=False):
    df = pd.read_csv(frame_index)
    geo_crs = {'init':'epsg:4326'}
    df_rec = df.copy()
    df_rec['geom'] = df_rec['geom'].apply(fix_polygon_wkt)
    gdf = gpd.GeoDataFrame(df_rec,geometry='geom',crs=geo_crs)
    if df_only:
        out = df
    else:
        out = gdf
    return out

