#! /usr/bin/env python

import os,sys,glob,re
import argparse
from pygeotools.lib import iolib,malib
from skysat_stereo import asp_utils as asp
from skysat_stereo import skysat
from skysat_stereo import skysat_stereo_workflow as workflow
from p_tqdm import p_map
import numpy as np
from multiprocessing import cpu_count
import pandas as pd

def getparser():
    parser = argparse.ArgumentParser(description = 'Script for initialing frame cameras for Skysat triplet stereo and video, performing user defined video subsampling')
    modes = ['video','triplet']
    parser.add_argument('-mode',default='video',choices=modes, help='choose Skysat product to work with')
    session_choices = ['rpc','pinhole']
    parser.add_argument('-t',default='pinhole',choices=session_choices,help='choose between pinhole and rpc mode (default: %(default)s)')
    parser.add_argument('-img',default=None,help='folder containing images',required=True)
    sampling_mode_choices = ['sampling_interval', 'num_images']
    parser.add_argument('-video_sampling_mode', default = 'num_images', choices = sampling_mode_choices, required = False, help = 'Chose desired sampling procedure, either fixed sampling interval or by equally distributed user defined number of samples (default: %(default)s)')
    parser.add_argument('-sampler',default = 5 ,type = int, help = 'if video_sampling_mode: sampling_interval, this is the sampling interval, else this is the number of samples to be selected (default: %(default)s)')
    parser.add_argument('-outdir', default = None, required = True, help = 'Output folder to save cameras and GCPs')
    parser.add_argument('-frame_index',default=None,help='Frame index csv file provided with L1A video products, will be used for determining stereo combinations')
    parser.add_argument('-overlap_pkl',default=None,help='pkl dataframe containing entries of overlapping pairs for triplet run, obtained from skysat_overlap_parallel.py')
    parser.add_argument('-dem',default=None,help='Reference DEM to be used for frame camera initialisation')
    product_levels = ['l1a','l1b']
    parser.add_argument('-product_level', choices = product_levels,default='l1b',required = False, help = 'Product level being processed, (default: %(default)s)')
    return parser

def main():
    parser = getparser()
    args = parser.parse_args()
    mode = args.mode
    session = args.t
    img_folder = os.path.abspath(args.img)
    outdir = os.path.abspath(args.outdir)
    cam_gen_log = workflow.skysat_preprocess(img_folder,mode,sampling=args.video_sampling_mode,frame_index_fn=args.frame_index,
        product_level=args.product_level,sampler=args.sampler,overlap_pkl=args.overlap_pkl,dem=args.dem,
        outdir=args.outdir)
        
    from datetime import datetime
    now = datetime.now()
    log_fn = os.path.join(outdir,'camgen_{}.log'.format(now))
    print("saving subprocess camgen log at {}".format(log_fn))
    with open(log_fn,'w') as f:
        for log in cam_gen_log:
            f.write(log)
    print("Script is complete !")

if __name__=="__main__":
    main()
