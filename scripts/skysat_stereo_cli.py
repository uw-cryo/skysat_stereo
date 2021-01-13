#! /usr/bin/env python

from skysat_stereo import asp_utils as asp
from skysat_stereo import skysat
import numpy as np
import argparse
import os,sys,glob
from multiprocessing import cpu_count
from p_tqdm import p_map

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
    return parser

def main():
    parser = getparser()
    args = parser.parse_args()
    img = os.path.abspath(args.img)
    try:
        img_list = sorted(glob.glob(os.path.join(img, '*.tif')))
        temp = img_list[1]
    except BaseException:
        img_list = sorted(glob.glob(os.path.join(img, '*.tiff')))
    if len(img_list) == 0:
        print("No images in the specified folder, exiting")
        sys.exit()
    mode = args.mode
    session = args.t
    ba_prefix = args.ba_prefix
    overlap_list_fn = args.overlap_pkl
    frame_index = args.frame_index
    dem = args.dem
    texture = args.texture
    sampling_interval = args.sampling_interval
    if args.cam:
        cam_folder = args.cam
    if args.ba_prefix:
        ba_prefix = args.ba_prefix
    outfol = args.outfol
    if mode == 'video':
        # assume for now that we are still operating on a fixed image interval method
        # can accomodate different convergence angle function method here.
        frame_gdf = skysat.parse_frame_index(frame_index)
        # for now hardcording sgm,mgm,kernel params, should accept as inputs.
        # Maybe discuss with David with these issues/decisions when the overall
        # system is in place
        if args.mvs == 1:
            job_list = skysat.video_mvs(img,t=session,cam_fol=args.cam,ba_prefix=args.ba_prefix,dem=args.dem,sampling_interval=sampling_interval,texture=texture,outfol=outfol, block=args.block,frame_index=frame_gdf)
        else:
            if args.full_extent == 1:
                full_extent = True
            else:
                full_extent=False
            job_list = skysat.prep_video_stereo_jobs(img,t=session,cam_fol=args.cam,ba_prefix=args.ba_prefix,dem=args.dem,sampling_interval=sampling_interval,texture=texture,outfol=outfol,block=args.block,frame_index=frame_gdf,full_extent=full_extent,entry_point=args.entry_point)
    elif mode == 'triplet':
        if args.crop_map == 1:
            crop_map = True
        else: 
            crop_map = False
        job_list = skysat.triplet_stereo_job_list(t=args.t,
                threads = args.threads,overlap_list=args.overlap_pkl, img_list=img_list, ba_prefix=args.ba_prefix, cam_fol=args.cam, dem=args.dem, crop_map=crop_map,texture=texture, outfol=outfol, block=args.block,entry_point=args.entry_point)
    # decide on number of processes
    # if block matching, Plieades is able to handle 30-40 4 threaded jobs on bro node
    # if MGM/SGM, 25 . This stepup is arbitrariry, research on it more.
    # next build should accept no of jobs and stereo threads as inputs
    print(job_list[0])
    n_cpu = cpu_count()
    # no of parallel jobs with user specified threads per job
    jobs = int(n_cpu/args.threads)
    stereo_log = p_map(asp.run_cmd,['stereo']*len(job_list), job_list, num_cpus=jobs)
    stereo_log_fn = os.path.join(outfol,'stereo_log.log')
    print("Consolidated stereo log saved at {}".format(stereo_log_fn))
    #with open(stereo_log_fn,'w') as f:
     #   for logs in stereo_log:
      #      f.write(logs)
    print("Script is complete")

if __name__ == "__main__":
    main()
