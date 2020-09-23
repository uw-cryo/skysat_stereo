#! /usr/bin/env python

import argparse
import glob
from skysat_stereo import asp_utils as asp
from skysat_stereo import skysat
from skysat_stereo import misc_geospatial as geo
import time
import os,sys,glob
from p_tqdm import p_umap, p_map
from multiprocessing import cpu_count
from shapely.geometry import Polygon
from itertools import combinations,compress
import numpy as np
import geopandas as gpd
import pandas as pd
def getparser():
    parser = argparse.ArgumentParser(description='Script to make overlapping pairs based on user defined minimum overlap percentage')
    parser.add_argument('-img_folder', help='Folder containing images with RPC information', required=True)
    parser.add_argument('-percentage', '--percentage', help='percentage_overlap between 0 to 1', required=True)
    parser.add_argument('-outfn','--out_fn',help='Text file containing the overlapping pairs')
    return parser

# Global var
geo_crs = {'init':'EPSG:4326'}

def main():
    #The following block of code is useful for getting a shapefile encompassing the entire subset (Use for clipping DEMs etc)
    #Also, I define the local ortho coordinates using the center of the big bounding box
    init_time = time.time()
    parser = getparser()
    args = parser.parse_args()
    img_folder = args.img_folder
    try:
        img_list = sorted(glob.glob(os.path.join(img_folder,'*.tif')))
        print("Number of images {}".format(len(img_list)))
    except:
        print ("No images found in the directory. Make sure they end with a .tif extension")
        sys.exit()
    out_fn = args.out_fn
    perc_overlap = np.float(args.percentage)
    out_shp = os.path.splitext(out_fn)[0]+'_bound.gpkg'
    n_proc = cpu_count()
    shp_list = p_umap(skysat.skysat_footprint,img_list,num_cpus=2*n_proc)
    merged_shape = geo.shp_merger(shp_list)
    bbox = merged_shape.total_bounds
    print (f'Bounding box lon_lat is:{bbox}')
    bound_poly = Polygon([[bbox[0],bbox[3]],[bbox[2],bbox[3]],[bbox[2],bbox[1]],[bbox[0],bbox[1]]])
    bound_shp = gpd.GeoDataFrame(index=[0],geometry=[bound_poly],crs=geo_crs)
    bound_centroid = bound_shp.centroid
    cx = bound_centroid.x.values[0]
    cy = bound_centroid.y.values[0]
    pad = np.ptp([bbox[3],bbox[1]])/6.0
    lat_1 = bbox[1]+pad
    lat_2 = bbox[3]-pad
    #local_ortho = '+proj=ortho +lat_0={} +lon_0={}'.format(cy,cx)
    local_aea = "+proj=aea +lat_1={} +lat_2={} +lat_0={} +lon_0={} +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs".format(lat_1,lat_2,cy,cx)
    print ('Local Equal Area coordinate system is : {} \n'.format(local_aea))
    print('Saving bound shapefile at {} \n'.format(out_shp))
    bound_shp.to_file(out_shp,driver='GPKG')
    img_combinations = list(combinations(img_list,2))
    n_comb = len(img_combinations)
    perc_overlap = np.ones(n_comb,dtype=float)*perc_overlap
    proj = local_aea
    tv = p_map(skysat.frame_intsec, img_combinations, [proj]*n_comb, perc_overlap,num_cpus=4*n_proc)
    # result to this contains truth value (0 or 1, overlap percentage)
    truth_value = [tvs[0] for tvs in tv]
    overlap = [tvs[1] for tvs in tv]
    valid_list = list(compress(img_combinations,truth_value))
    overlap_perc_list = list(compress(overlap,truth_value))
    print('Number of valid combinations are {}, out of total {}  input images making total combinations {}\n'.format(len(valid_list),len(img_list),n_comb))
    with open(out_fn, 'w') as f:
        img1_list = [x[0] for x in valid_list]
        img2_list = [x[1] for x in valid_list]
        for idx,i in enumerate(valid_list):
            #f.write("%s %s\n" % i) 
            f.write(f"{os.path.abspath(img1_list[idx])} {os.path.abspath(img2_list[idx])}\n")
    out_fn_overlap = os.path.splitext(out_fn)[0]+'_with_overlap_perc.pkl'
    img1_list = [x[0] for x in valid_list]
    img2_list = [x[1] for x in valid_list]
    out_df = pd.DataFrame({'img1':img1_list,'img2':img2_list,'overlap_perc':overlap_perc_list})
    out_df.to_pickle(out_fn_overlap)
    out_fn_stereo = os.path.splitext(out_fn_overlap)[0]+'_stereo_only.pkl'
    stereo_only_df = skysat.prep_trip_df(out_fn_overlap)
    stereo_only_df.to_pickle(out_fn_stereo)
    print("Saving overlap with stereo pairs only")
    print('Script completed in time {} s!'.format(time.time()-init_time))

if __name__=="__main__":
    main()
