#! /usr/bin/env python

import numpy as np
import os,sys,glob,shutil
import argparse
from skysat_stereo import asp_utils as asp
from skysat_stereo import skysat
from skysat_stereo import skysat_stereo_workflow as workflow
from p_tqdm import p_map
from imview import pltlib
import itertools
import ast
import matplotlib.pyplot as plt
from multiprocessing import cpu_count


def get_parser():
	parser = argparse.ArgumentParser(description='create browse image from input Skysat directory')
	parser.add_argument('-img_folder', help='Folder containing subdirectories of imagefiles', required=True)
	session_choice = ['rpc','pinhole']
	parser.add_argument('-session',choices = session_choice, default = 'rpc', help = 'Session for mapproject (defualt: %(default)s)')
	parser.add_argument('-out_folder',help='Folder where output orthoimages will be stored', required=True)
	parser.add_argument('-tr',help='Output image resolution',default=None)
	parser.add_argument('-tsrs',help='Output crs as EPSG code, example EPSG:32610')
	parser.add_argument('-DEM',help='Optional DEM for mapprojecting',default='WGS84')
	parser.add_argument('-delete_temporary_files',help='Delete temporary individual mapprojected files written to disc',default=True)
	map_choices = ['science','browse']
	parser.add_argument('-mode',choices=map_choices,default='browse',help='select mode for mapprojection default: %(default)s')
	parser.add_argument('-ba_prefix',default=None,help='bundle adjust prefix for rpc, or joiner for bundle adjusted pinhole cameras',required=False)
	parser.add_argument('-cam',default=None,help='camera folder containing list of tsai files for pinhole files',required=False)
	parser.add_argument('-frame_index',default=None,help="frame index to read frame's actual Ground sampling distance",required=False)
	orthomosaic_choice = [1,0]
	parser.add_argument('-orthomosaic',default=0,type=int,choices=orthomosaic_choice, help="if mode is science, enabling this (1) will also produce a final orthomosaic (default: %(default)s)")
	parser.add_argument('-copy_rpc',default=0,type=int,choices=orthomosaic_choice,help='if mode is science, enabling this (1) will copy rpc metadata in the orthoimage (default: %(default)s)')
	data_choices = ['video','triplet']
	parser.add_argument('-data',default='triplet',choices=data_choices,help="select if mosaicing video or triplet product in science mode (default: %(default)s)")
	parser.add_argument('-overlap_list', default=None,
		help='list containing pairs for which feature matching was restricted due during cross track bundle adjustment (not required during basic triplet processing)')
	return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.tr is not None:
        tr = str(args.tr)
    else:
        tr = None
    tsrs = args.tsrs
    dir = os.path.abspath(args.img_folder)
    outdir = os.path.abspath(args.out_folder)
    images = sorted(glob.glob(os.path.join(dir,'*.tif*')))
    if os.path.islink(images[0]):
        images = [os.readlink(x) for x in images]
    del_opt = args.delete_temporary_files
    dem=args.DEM
    cam_folder = args.cam
    ba_prefix = args.ba_prefix
    mode = args.mode
    workflow.execute_skysat_orhtorectification(images,outdir,data=args.data,dem=dem,
                                               tr=tr,tsrs=tsrs,del_opt=args.delete_temporary_files,
                                               cam_folder=cam_folder,ba_prefix=ba_prefix,mode=mode,
                                               session=args.session,overlap_list=args.overlap_list,
                                               frame_index_fn=args.frame_index,copy_rpc=args.copy_rpc,
                                               orthomosaic=args.orthomosaic)
    print("Script is complete!") 

if __name__=='__main__':
    main()
