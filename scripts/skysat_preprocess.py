#! /usr/bin/env python

import os,sys,glob,re
import argparse
from pygeotools.lib import iolib,malib
from skysat_stereo import asp_utils as asp
from skysat_stereo import skysat
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
    parser.add_argument('-video_sampling_mode', default = 'sampling_interval', choices = sampling_mode_choices, required = False, help = 'Chose desired sampling procedure, either fixed sampling interval or by equally distributed user defined number of samples (default: %(default)s)')
    parser.add_argument('-sampler',default = 5 ,type = int, help = 'if video_sampling_mode: sampling_interval, this is the sampling interval, else this is the number of samples to be selected (default: %(default)s)')
    parser.add_argument('-outdir', default = None, required = True, help = 'Output folder to save cameras and GCPs')
    parser.add_argument('-frame_index',default=None,help='Frame index csv file provided with L1A video products, will be used for determining stereo combinations')
    parser.add_argument('-overlap_pkl',default=None,help='pkl dataframe containing entries of overlapping pairs for triplet run, obtained from skysat_overlap_parallel.py')
    parser.add_argument('-dem',default=None,help='Reference DEM to be used for frame camera initialisation')
    product_levels = ['l1a','l1b']
    parser.add_argument('-product_level', choices = product_levels,default='l1a',required = False, help = 'Product level being processed, (default: %(default)s)')
    return parser

def main():
    parser = getparser()
    args = parser.parse_args()
    mode = args.mode
    session = args.t
    img_folder = os.path.abspath(args.img)
    outdir = os.path.abspath(args.outdir)
    if not os.path.exists(outdir):
        try:
            os.makedir(outdir)
        except:
            os.makedirs(outdir)
    if mode == 'video':
        sampling = args.video_sampling_mode
        frame_index = skysat.parse_frame_index(args.frame_index,True)
        product_level = 'l1a'
        num_samples = len(frame_index)
        frames = frame_index.name.values
        sampler = args.sampler
        outdf = os.path.join(outdir,os.path.basename(args.frame_index))
        if sampling == 'sampling_interval':
            print("Hardcoded sampling interval results in frame exclusion at the end of the video sequence based on step size, better to chose the num_images mode and the program will equally distribute accordingly")
            idx = np.arange(0,num_samples,sampler)
            outdf = f'{os.path.splitext(outdf)[0]}_sampling_inteval_{sampler}.csv'
        else:
            print(f"Sampling {sampler} from {num_samples} of the input video sequence")
            idx = np.linspace(0,num_samples-1,sampler,dtype=int)
            outdf = f'{os.path.splitext(outdf)[0]}_sampling_inteval_aprox{idx[1]-idx[0]}.csv'
        sub_sampled_frames = frames[idx]
        sub_df = frame_index[frame_index['name'].isin(list(sub_sampled_frames))]
        sub_df.to_csv(outdf,sep=',',index=False)
        #this is camera/gcp initialisation
        n = len(sub_sampled_frames)
        img_list = [glob.glob(os.path.join(img_folder,f'{frame}*.tiff'))[0] for frame in sub_sampled_frames]
        pitch = [1]*n
        out_fn = [os.path.join(outdir,f'{frame}_frame_idx.tsai') for frame in sub_sampled_frames]
        out_gcp = [os.path.join(outdir,f'{frame}_frame_idx.gcp') for frame in sub_sampled_frames]
        frame_index = [args.frame_index]*n
       	camera = [None]*n

    elif mode == 'triplet':
        df = pd.read_pickle(args.overlap_pkl)
        img_list = list(np.unique(np.array(list(df.img1.values)+list(df.img2.values))))
        img_list = [os.path.splitext(os.path.basename(img))[0] for img in img_list]
        cam_list = [glob.glob(os.path.join(img_folder,f'{img}*.tif'))[0] for img in img_list]
        n = len(img_list)
        pitch = [0.8]*n
        out_fn = [os.path.join(outdir,f'{frame}_rpc.tsai') for frame in img_list]
        out_gcp = [os.path.join(outdir,f'{frame}_rpc.gcp') for frame in img_list]
        camera = cam_list
        frame_index = [None]*n
        img_list = cam_list
    fl = [553846.153846]*n
    cx = [1280]*n
    cy = [540]*n
    dem = args.dem
    ht_datum = [malib.get_stats_dict(iolib.fn_getma(dem))['median']]*n # use this value for height where DEM has no-data
    gcp_std = [1]*n
    datum = ['WGS84']*n
    refdem = [dem]*n
    n_proc = 30
    #n_proc = cpu_count()
    cam_gen_log = p_map(asp.cam_gen,img_list,fl,cx,cy,pitch,ht_datum,gcp_std,out_fn,out_gcp,datum,refdem,camera,frame_index,num_cpus = n_proc)
    print("writing gcp with basename removed")
    asp.clean_gcp(out_gcp,outdir)
    # saving subprocess consolidated log file
    from datetime import datetime
    now = datetime.now()
    log_fn = os.path.join(outdir,f'camgen_{now}.log')
    print(f"saving subprocess camgen log at {log_fn}")
    with open(log_fn,'w') as f:
        for log in cam_gen_log:
            f.write(log)
    print("Script is complete !")

if __name__=="__main__":
    main()
