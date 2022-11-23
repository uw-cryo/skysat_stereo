#! /usr/bin/env python

import argparse
import glob
from skysat_stereo import asp_utils as asp
from skysat_stereo import skysat
from skysat_stereo import skysat_stereo_workflow as workflow
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
    parser.add_argument('-percentage', '--percentage', help='percentage_overlap between 0 to 1', type=float, required=True)
    parser.add_argument('-outfn','--out_fn',help='Text file containing the overlapping pairs', type=str, required=True)
    parser.add_argument('-cross_track',action='store_true',help='Also make cross-track pairs')
    parser.add_argument('-aoi_bbox',help='Return interesecting footprint within this aoi only', default=None)
    return parser

# Global var
geo_crs = 'EPSG:4326'

def main():
    #The following block of code is useful for getting a shapefile encompassing the entire subset (Use for clipping DEMs etc)
    #Also, I define the local ortho coordinates using the center of the big bounding box
    init_time = time.time()
    parser = getparser()
    args = parser.parse_args()
    img_folder = args.img_folder
    workflow.prepare_stereopair_list(img_folder,args.percentage,args.out_fn,args.aoi_bbox,cross_track=args.cross_track)
    print(f'Script completed in time {time.time()-init_time}')

if __name__=="__main__":
    main()
