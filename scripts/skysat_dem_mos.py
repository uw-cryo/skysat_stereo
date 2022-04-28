#! /usr/bin/env python

import os,sys,glob,shutil
import numpy as np
import argparse
from skysat_stereo import asp_utils as asp
from skysat_stereo import skysat
from skysat_stereo import skysat_stereo_workflow as workflow 
from tqdm import tqdm
from p_tqdm import p_map

from pygeotools.lib import iolib,warplib

def getparser():
    parser = argparse.ArgumentParser(description='Script to compute DEM mosaics from triplet output directory')
    parser.add_argument('-DEM_folder', help='Folder containing subdirectories of DEM', required=True)
    parser.add_argument('-out_folder', help='Where composite DEMs are to be saved, if none, creates a composite DEM directory in the input main directory', required=False,default=None)
    parser.add_argument('-identifier',help='if we want to mosaic individually aligned DEM which have been produced by skysat_pc_cam.py, place the identifiers here',required=False,default=None)
    mode_ch = ['video','triplet']
    parser.add_argument('-mode',default='triplet',choices=mode_ch,help="select if mosaicing video or triplet stereo output DEMs (default: %(default)s)")
    parser.add_argument('-tile_size',default=None,help='Tile size for tiled processing, helpful on nodes with less memory or if num_dems are large')
    binary_ch = [1,0]
    parser.add_argument('-filter_dem',choices=binary_ch,default=1,type=int,
        help="filter video DEM composites using max NMAD and min count combination (default: %(default)s)")
    parser.add_argument('-min_video_count',type=float,default=2,
        help='minimum DEM count to use in filtering (default: %(default)s)')
    parser.add_argument('-max_video_nmad',type=float,default=5,
        help='maximum DEM NMAD variability to filter, if DEM count is also <= min_count (default: %(default)s)') 
    return parser

def main():
    parser = getparser()
    args = parser.parse_args()
    dir = os.path.abspath(args.DEM_folder)
    workflow.dem_mosaic_wrapper(dir,mode=args.mode,out_folder=args.out_folder,identifier=args.identifier,
                       tile_size=args.tile_size,filter_dem=args.filter_dem,
                       min_video_count=args.min_video_count,max_video_nmad=args.max_video_nmad)
    print("Script complete")
    
if __name__=="__main__":
    main()