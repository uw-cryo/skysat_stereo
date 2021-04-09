#! /usr/bin/env python

import os,sys,glob,shutil
import numpy as np
import argparse
from skysat_stereo import asp_utils as asp
from skysat_stereo import skysat
from tqdm import tqdm
from p_tqdm import p_map
import itertools
from pygeotools.lib import iolib,warplib

def getparser():
    parser = argparse.ArgumentParser(description='Script to compute DEM mosaics from triplet output directory')
    parser.add_argument('-DEM_folder', help='Folder containing subdirectories of DEM', required=True)
    parser.add_argument('-out_folder', help='Where composite DEMs are to be saved, if none, creates a composite DEM directory in the input main directory', required=False)
    parser.add_argument('-identifier',help='if we want to mosaic individually aligned DEM which have been produced by skysat_coreg.py, place the identifiers here',required=False,default=None)
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
        print(f"Number of combinations {len(dir_list)}")
        
        def valid_disp(direc):
            try:
                D_sub = iolib.fn_getma(os.path.join(direc,'run-D_sub.tif'),3)
                stats = np.percentile(D_sub.compressed(),(2,98))
                direc_out = glob.glob(os.path.join(direc,'run*{}*-DEM.tif'.format(identifier)))[0]
                out = (direc_out,True)
            except:
                out = (0,False)
            return out
        # find all valid DEMs
        total_cpu = iolib.cpu_count()
        n_proc = total_cpu - np.arange(len(dir_list))
        #this setup is so that old pools are discarded and new ones with new number of workers are created
        valid_dem_dir_list = []
        for idx,direc in enumerate(tqdm(dir_list)):
            comb_dir_list = sorted(glob.glob(os.path.join(direc,'*/')))
            results = p_map(valid_disp,comb_dir_list,num_cpus=n_proc[idx])
            t_val = [r[1] for r in results]
            valid_dirs = list(itertools.compress([r[0] for r in results],t_val))
            valid_dem_dir_list.append(valid_dirs)
        

        # naming logic for pairwise and triplet/multiview composites
        mdt1,t1,mdt2,t2 = [[],[],[],[]]
        combination_out_list = []
        if len(dir_list)>3:
            print("Input is cross-track")
        if dir_list[0][-1] == '/':
            dir_list = [x[:-1] for x in dir_list]
        for direc in dir_list:

            combination_out_list.append(os.path.join(out_folder,os.path.basename(direc)+'_wt_avg_mos.tif'))
            a,b,c,d = os.path.basename(direc).split('_')
            mdt1.append(a) #master date (year month date)
            t1.append(b) # time of day in seconds
            mdt2.append(c)
            t2.append(d)
        if len(direc)>3:
            composite_prefix = 'multiview_'+'_'.join(np.unique(mdt1+mdt2))+'__'+'_'.join(np.unique(t1+t2))
        else:
            composite_prefix = 'triplet_'+'_'.join(np.unique(mdt1+mdt2))+'__'+'_'.join(np.unique(t1+t2))

        # produce bistereo pairwise mosaics
        len_combinations = len(combination_out_list)
        tile_size = args.tile_size

        if len_combinations > 3:
            # force tiled processing in case of multiview mosaicking
            if not tile_size:
                tile_size = 400
       
        mos_log = p_map(asp.dem_mosaic,valid_dem_dir_list,combination_out_list,
                ['None']*len_combinations,[None]*len_combinations,
                [None]*len_combinations,[tile_size]*len_combinations)

        if len_combinations > 2:
            print("Producing triplet/multiview composites")
            total_dem_list = list(itertools.chain.from_iterable(valid_dem_dir_list))
            print(f"Mosaicing {len(total_dem_list)} DEM strips using median, wt_avg, count, nmad operators")
            stats_list = [None,'median','nmad','count']
            out_fn_list = [os.path.join(out_folder,
                            '{}_{}_mos.tif'.format(composite_prefix,stat)) for stat in ['wt_avg','median','nmad','count']]
            composite_mos_log = p_map(asp.dem_mosaic,[total_dem_list]*4,out_fn_list,['None']*4,[None]*4,stats_list,
                                [tile_size]*4,num_cpus=4)

        out_log_fn = os.path.join(out_folder,'skysat_triplet_dem_mos.log')
        print("Saving triplet DEM mosaic log at {}".format(out_log_fn))
        with open(out_log_fn,'w') as f:
            for log in mos_log+composite_mos_log:
                f.write(log) 
    elif args.mode=='video':
        dir_list = sorted(glob.glob(os.path.join(dir,'1*/')))
        valid_video_dir = []
        for video_dir in dir_list:
            try:
                D_sub = iolib.fn_getma(os.path.join(video_dir,'run-D_sub.tif'),3)
                stats = [np.percentile(D_sub.compressed(),(2,98)),np.mean(D_sub.compressed())]
                DEM = glob.glob(os.path.join(video_dir,'run*{}*-DEM.tif'.format(identifier)))[0]
                valid_video_dir.append(video_dir)
            except:
                continue 
        video_dem_list = [glob.glob(os.path.join(dir,f'run*{identifier}*-DEM.tif'))[0] for dir in valid_video_dir]
        stats_list = ['median','count','nmad']
        print('total dems are {}'.format(len(video_dem_list)))
        out_fn_list = [os.path.join(out_folder,'video_{}_mos.tif'.format(stat)) for stat in stats_list]
        dem_mos_log = p_map(asp.dem_mosaic,[video_dem_list]*3,out_fn_list,['None']*3,[None]*3,stats_list,[None]*3) 
        out_log_fn = os.path.join(out_folder,'skysat_video_dem_mos.log')
        with open(out_log_fn,'w') as f:
            for log in dem_mos_log:
                f.write(log)
        if args.filter_dem == 1:
            print("Filtering DEM using NMAD and count metrics")
            min_count = args.min_video_count
            max_nmad = args.max_video_nmad
            print(f"Filter will use min count of {min_count} and max NMAD of {max_nmad}")
            mos_ds_list = warplib.memwarp_multi_fn(out_fn_list)
            # Filtered array list contains dem_filtered,nmad_filtered, count_filtered in order
            filtered_array_list = skysat.filter_video_dem_by_nmad(mos_ds_list,min_count,max_nmad)
            trailing_str = f'_filt_max_nmad{max_nmad}_min_count{min_count}.tif'
            out_filter_fn_list = [os.path.splitext(fn)[0]+trailing_str for fn in out_fn_list]
            for idx,fn in enumerate(out_filter_fn_list):
                iolib.writeGTiff(filtered_array_list[idx],fn,mos_ds_list[idx])
    print("Script complete")

if __name__=="__main__":
    main()

      
