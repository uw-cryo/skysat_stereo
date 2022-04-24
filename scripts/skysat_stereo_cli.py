#! /usr/bin/env python

from skysat_stereo import asp_utils as asp
from skysat_stereo import skysat
import numpy as np
import argparse
import os,sys,glob
from multiprocessing import cpu_count
from p_tqdm import p_map
from skysat_stereo import skysat_stereo_workflow as workflow
from tqdm import tqdm

def getparser():
    parser = argparse.ArgumentParser(description='Script for performing stereo jobs, generalised for skysat video and triplet stereo products')
    modes = ['video', 'triplet']
    parser.add_argument('-mode',default='video',choices=modes,help='choose Skysat product to work with')
    session_choices = ['rpc', 'nadirpinhole', 'rpcmaprpc', 'pinholemappinhole']
    # mapprojecting inputs are faster to process, and generally more complete
    # (less holes) + accurate (less blunders in stereo matching)
    parser.add_argument('-threads',default=cpu_count(),type=int,
            help='number of threads to use for each stereo process, (default: %(default)s)')
    entry_choice = ['pprc','corr','rfne','fltr','tri']
    parser.add_argument('-entry_point',type=str,default='pprc',help='start stereo from a particular stage (default: %(default)s)')
    parser.add_argument('-t',default='nadirpinhole',choices=session_choices,help='choose between pinhole and rpc mode (default: %(default)s)')
    parser.add_argument('-img',default=None,help='folder containing images',required=True)
    parser.add_argument('-cam',default=None,help='folder containing cameras, if using nadirpinhole/pinholemappinhole workflow',required=False)
    # note that the camera should contain similar names as images. We do a
    # simple string search to read appropriate camera.
    parser.add_argument('-ba_prefix',default=None, help='bundle adjust prefix for reading transforms from .adjust files, mainly for rpc runs, or for reading the correct cameras from a bundle adjustment directory containing multiple generations of pinhole cameras', required=False)
    parser.add_argument('-overlap_pkl',default=None,help='pkl dataframe containing entries of overlapping pairs for triplet run, obtained from skysat_overlap_parallel.py')
    parser.add_argument('-frame_index',default=None,help='Frame index csv file provided with L1A video products, will be used for determining stereo combinations')
    parser.add_argument('-sampling_interval',default=5,required=False,type=int,help='Sampling interval between stereo DEM input pairs, or the interval at which master images are picked for multiview stereo triangulation (default: %(default)s)')
    parser.add_argument('-dem',default=None,help='Reference DEM to be used in triangulation, if input images are mapprojected')
    texture_choices = ['low', 'normal']
    parser.add_argument('-texture',default='normal',choices=texture_choices,help='keyword to adapt processing for low texture surfaces, for example in case of fresh snow (default: %(default)s)',required=False)
    crop_ops = [1,0]
    parser.add_argument('-crop_map',default=1,type=int,choices=crop_ops,help='To crop mapprojected images to same resolution and extent or not before stereo')
    parser.add_argument('-outfol', default=None, help='output folder where stereo outputs will be saved', required=True)
    mvs_choices = [1, 0]
    parser.add_argument('-mvs', default=0, type=int, choices=mvs_choices, help='1: Use multiview stereo triangulation for video data, do matching with next 20 slave for each master image/camera (defualt: %(default)s')
    parser.add_argument('-block', default=0, type=int, choices=mvs_choices, help='1: use block matching instead of default MGM (default: %(default)s')
    parser.add_argument('-full_extent',type=int,choices = mvs_choices,default=1,
                        help='Selecting larger intervals can result in lower footprint output DEM, if 1: then DEMs with smaller interval image pairs will be padded at the begining and end of the video sequence (default: %(default)s)')
    parser.add_argument('-writeout_only', action='store_true', help='writeout_jobs to a text file, not run')
    parser.add_argument('-job_fn',type=str,help='text file to write stereo jobs to',default=None)
    parser.add_argument('-cross_track',action='store_true', help='attempt stereo for cross_track pairs as well')
    return parser


def main():
    parser = getparser()
    args = parser.parse_args()
    img = os.path.abspath(args.img)

    workflow.execute_skysat_stereo(img,args.outfol,args.mode,session=args.t,
        dem=args.dem,texture=args.texture,sampling_interval=args.sampling_interval,
        cam_folder=args.cam,ba_prefix=args.ba_prefix,writeout_only=args.writeout_only,
        mvs=args.mvs,block=args.block,crop_map=args.crop_map,full_extent=args.full_extent,
        entry_point=args.entry_point,threads=args.threads,overlap_pkl=args.overlap_pkl,
        frame_index=args.frame_index,job_fn=args.job_fn,cross_track=args.cross_track)
    
    print("Script is complete")

if __name__ == "__main__":
    main()

