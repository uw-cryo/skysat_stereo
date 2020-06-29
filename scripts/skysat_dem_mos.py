#! /usr/bin/env python

import os,sys,glob,shutil
import numpy as np
import argparse
from skysat_stereo import asp_utils as asp
from skysat_stereo import skysat
from p_tqdm import p_map
import itertools
from pygeotools.lib import iolib

def getparser():
    parser = argparse.ArgumentParser(description='Script to compute DEM mosaics from triplet output directory')
    parser.add_argument('-DEM_folder', help='Folder containing subdirectories of DEM', required=True)
    parser.add_argument('-out_folder', help='Where composite DEMs are to be saved, if none, creates a composite DEM directory in the input main directory', required=False)
    parser.add_argument('-identifier',help='if we want to mosaic individually aligned DEM which have been produced by skysat_coreg.py, place the identifiers here',required=False,default=None)
    mode_ch = ['video','triplet']
    parser.add_argument('-mode',default='triplet',choices=mode_ch,help="select if mosaicing video or triplet stereo output DEMs (default: %(default)s)")
    return parser 

def main():
    parser = getparser()
    args = parser.parse_args()
    dir = os.path.abspath(args.DEM_folder)
    if args.out_folder:
        out_folder = os.path.abspath(args.out_folder)
    else:
        out_folder = os.path.join(dir,'composite_dems')
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if args.identifier:
         # for indi align DEMs
         identifier = args.identifier
    else:
        identifier = ''
    if args.mode == 'triplet':
        dir_list = sorted(glob.glob(os.path.join(dir,'20*/')))
        valid_for_nadir_dir = []
        valid_for_aft_dir = []
        valid_nadir_aft_dir = []
        for for_nadir_dir in sorted(glob.glob(os.path.join(dir_list[0],'*/'))):
            try:
               D_sub = iolib.fn_getma(os.path.join(for_nadir_dir,'run-D_sub.tif'),3)
               stats = [np.percentile(D_sub.compressed(),(2,98)),np.mean(D_sub.compressed())]
               DEM = glob.glob(os.path.join(for_nadir_dir,f'run*{identifier}*-DEM.tif'))[0]
               valid_for_nadir_dir.append(for_nadir_dir)
            except:
               continue
        for for_aft_dir in sorted(glob.glob(os.path.join(dir_list[2],'*/'))):
            try:
               # see ASP issue for this dirty hack: https://github.com/NeoGeographyToolkit/StereoPipeline/issues/308
                D_sub = iolib.fn_getma(os.path.join(for_aft_dir,'run-D_sub.tif'),3)
                stats = [np.percentile(D_sub.compressed(),(2,98)),np.mean(D_sub.compressed())]
                DEM = glob.glob(os.path.join(for_aft_dir,f'run*{identifier}*-DEM.tif'))[0]
                valid_for_aft_dir.append(for_aft_dir)
            except:
                continue
        for nadir_aft_dir in sorted(glob.glob(os.path.join(dir_list[1],'*/'))):
            try:
                D_sub = iolib.fn_getma(os.path.join(nadir_aft_dir,'run-D_sub.tif'),3)
                stats = [np.percentile(D_sub.compressed(),(2,98)),np.mean(D_sub.compressed())]
                DEM = glob.glob(os.path.join(nadir_aft_dir,f'run*{identifier}*-DEM.tif'))[0]
                valid_nadir_aft_dir.append(nadir_aft_dir)
            except:
                continue
        for_nadir_list = [glob.glob(os.path.join(dir,f'run*{identifier}*-DEM.tif'))[0] for dir in valid_for_nadir_dir]
        nadir_aft_list = [glob.glob(os.path.join(dir,f'run*{identifier}*-DEM.tif'))[0] for dir in valid_nadir_aft_dir]
        for_aft_list = [glob.glob(os.path.join(dir,f'run*{identifier}*-DEM.tif'))[0] for dir in valid_for_aft_dir]
        total_dem_list = for_nadir_list+for_aft_list+nadir_aft_list
        stats_list = ['nmad','count','median']
        print(f'total dems are {len(total_dem_list)}')
        out_fn_list = [os.path.join(out_folder,f'triplet_{stat}_mos.tif') for stat in stats_list]
        print("Mosaicing output total per-pixel nmad, count, nmad and 3 DEMs from 3 stereo combinations in parallel")
        dem_mos_log = p_map(asp.dem_mosaic,[total_dem_list]*3+[for_aft_list,nadir_aft_list,for_nadir_list],out_fn_list+[os.path.join(out_folder,x) for x in ['for_aft_dem_median_mos.tif', 'nadir_aft_dem_median_mos.tif', 'for_nadir_dem_median_mos.tif']],['None']*6,[None]*6,stats_list+['median']*3)
        out_log_fn = os.path.join(out_folder,'skysat_triplet_dem_mos.log')
        print(f"Saving triplet DEM mosaic log at {out_log_fn}")
        with open(out_log_fn,'w') as f:
            for log in dem_mos_log:
                f.write(log) 
    elif args.mode=='video':
        dir_list = sorted(glob.glob(os.path.join(dir,'1*/')))
        valid_video_dir = []
        for video_dir in dir_list:
            try:
                D_sub = iolib.fn_getma(os.path.join(video_dir,'run-D_sub.tif'),3)
                stats = [np.percentile(D_sub.compressed(),(2,98)),np.mean(D_sub.compressed())]
                DEM = glob.glob(os.path.join(video_dir,f'run*{identifier}*-DEM.tif'))[0]
                valid_video_dir.append(video_dir)
            except:
                continue 
        video_dem_list = [glob.glob(os.path.join(dir,'*-DEM.tif'))[0] for dir in valid_video_dir]
        stats_list = ['nmad','count','median']
        print(f'total dems are {len(video_dem_list)}')
        out_fn_list = [os.path.join(out_folder,f'video_{stat}_mos.tif') for stat in stats_list]
        dem_mos_log = p_map(asp.dem_mosaic,[video_dem_list]*3,out_fn_list,['None']*3,[None]*3,stats_list) 
        out_log_fn = os.path.join(out_folder,'skysat_video_dem_mos.log')
        with open(out_log_fn,'w') as f:
            for log in dem_mos_log:
                f.write(log)
    print("Script complete")

if __name__=="__main__":
    main()

      
