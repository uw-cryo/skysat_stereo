#! /usr/bin/env python

import numpy as np
import os,sys,glob,shutil
import argparse
from skysat_stereo import asp_utils as asp
from skysat_stereo import skysat
from p_tqdm import p_map
from imview import pltlib
import itertools
import ast
import matplotlib.pyplot as plt 

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
	parser.add_argument('-frame_index',default='None',help="frame index to read frame's actual Ground sampling distance",required=False) 
	orthomosaic_choice = [1,0]
	parser.add_argument('-orthomosaic',default=0,type=int,choices=orthomosaic_choice, help="if mode is science, enabling this (1) will also produce a final orthomosaic (default: %(default)s)")
	data_choices = ['video','triplet']
	parser.add_argument('-data',default='triplet',choices=data_choices,help="select if mosaicing video or triplet product in science mode (default: %(default)s)")
	return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    tr = str(args.tr)
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
    if mode == 'browse':
        """
        this block creates low-res orthomosaics from RPC info for browsing purpose only
        """
        for_img_list,nadir_img_list,aft_img_list,for_time,nadir_time,aft_time = skysat.sort_img_list(images)
        for_out_dir = os.path.join(outdir,'for_map_browse')
        nadir_out_dir = os.path.join(outdir,'nadir_map_browse')
        aft_out_dir = os.path.join(outdir,'aft_map_browse')
        for_out_list = [os.path.join(for_out_dir,os.path.splitext(os.path.basename(img))[0]+'_browse_map.tif') for img in for_img_list]
        nadir_out_list = [os.path.join(nadir_out_dir,os.path.splitext(os.path.basename(img))[0]+'_browse_map.tif') for img in nadir_img_list]
        aft_out_list = [os.path.join(aft_out_dir,os.path.splitext(os.path.basename(img))[0]+'_browse_map.tif') for img in aft_img_list]
        for_count,nadir_count,aft_count = [len(for_img_list), len(nadir_img_list), len(aft_img_list)]
        print(f"Performing orthorectification for forward images {for_time}")
        for_map_log = p_map(asp.mapproject,for_img_list,for_out_list,[args.session]*for_count,['WGS84']*for_count,[None]*for_count,['EPSG:4326']*for_count,[None]*for_count,[None]*for_count)
        print(f"Performing orthorectification for nadir images {nadir_time}")
        nadir_map_log = p_map(asp.mapproject,nadir_img_list,nadir_out_list,[args.session]*nadir_count,['WGS84']*nadir_count,[None]*nadir_count,['EPSG:4326']*nadir_count,[None]*nadir_count,[None]*nadir_count)
        print(f"Performing orthorectification for aft images {aft_time}")
        aft_map_log = p_map(asp.mapproject,aft_img_list,aft_out_list,[args.session]*aft_count,['WGS84']*aft_count,[None]*aft_count,['EPSG:4326']*aft_count,[None]*aft_count,[None]*aft_count)
        ortho_log = os.path.join(outdir,'low_res_ortho.log')
        print(f"Orthorectification log saved at {ortho_log}")
        with open(ortho_log,'w') as f:
            total_ortho_log = for_map_log+nadir_map_log+aft_map_log
            for log in itertools.chain.from_iterable(total_ortho_log):
                f.write(log)
    
        # after orthorectification, now do mosaic
        for_out_mos = os.path.join(outdir,'for_map_mos_{}m.tif'.format(tr))
        for_map_list = sorted(glob.glob(os.path.join(for_out_dir,'*.tif')))
        nadir_out_mos = os.path.join(outdir,'nadir_map_mos_{}m.tif'.format(tr))
        nadir_map_list = sorted(glob.glob(os.path.join(nadir_out_dir,'*.tif')))
        aft_out_mos = os.path.join(outdir,'aft_map_mos_{}m.tif'.format(tr))
        aft_map_list = sorted(glob.glob(os.path.join(aft_out_dir,'*.tif')))
        print("Preparing forward browse orthomosaic")
        for_mos_log = asp.dem_mosaic(for_map_list,for_out_mos,tr,tsrs,'first')
        print("Preparing nadir browse orthomosaic")
        nadir_mos_log = asp.dem_mosaic(nadir_map_list, nadir_out_mos, tr, tsrs, 'first')
        print("Preparing aft browse orthomosaic")
        aft_mos_log = asp.dem_mosaic(aft_map_list, aft_out_mos, tr, tsrs, 'first')
        ## delete temporary files
        if del_opt:
            [shutil.rmtree(x) for x in [for_out_dir,nadir_out_dir,aft_out_dir]]	
        #Save figure to jpeg ?
        fig_title = os.path.basename(images[0]).split('_',15)[0]+'_'+for_time+'_'+nadir_time+'_'+aft_time
        fig,ax = plt.subplots(1,3,figsize=(10,10))
        pltlib.iv_fn(for_out_mos,full=True,ax=ax[0],cmap='gray',scalebar=True,title='Forward')
        pltlib.iv_fn(nadir_out_mos,full=True,ax=ax[1],cmap='gray',scalebar=True,title='NADIR')
        pltlib.iv_fn(aft_out_mos,full=True,ax=ax[2],cmap='gray',scalebar=True,title='Aft')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(fig_title)
        browse_img_fn = os.path.join(outdir,'browse_img_{}_{}m.jpg'.format(fig_title,tr))
        fig.savefig(browse_img_fn,dpi=300,bbox_inches='tight',pad_inches=0.1)
        print("Browse figure saved at {}".format(browse_img_fn))
   
    if mode == 'science':
        img_list = images
        if args.frame_index is not 'None':
            frame_index = skysat.parse_frame_index(args.frame_index)
            img_list = [glob.glob(os.path.join(dir,f'{frame}*.tiff'))[0] for frame in frame_index.name.values]
            print(f"no of images is {len(img_list)}")
        img_prefix = [os.path.splitext(os.path.basename(img))[0] for img in img_list]
        out_list = [os.path.join(outdir,img+'_map.tif') for img in img_prefix]
        session_list = 	[args.session]*len(img_list)
        dem_list = [dem]*len(img_list)
        tr_list = [args.tr]*len(img_list)
        if args.frame_index is not 'None':
            # this hack is for video
            df = skysat.parse_frame_index(args.frame_index)
            trunc_df = df[df['name'].isin(img_prefix)]
            tr_list = [str(gsd) for gsd in trunc_df.gsd.values]
        srs_list = [tsrs]*len(img_list)
        if args.session == 'pinhole':
            if ba_prefix:
                cam_list = [glob.glob(os.path.abspath(ba_prefix)+'-'+os.path.splitext(os.path.basename(x))[0]+'*.tsai')[0] for x in img_list]
                print(f"No of cameras is {len(cam_list)}")
            else:
                cam_list = [glob.glob(os.path.join(os.path.abspath(cam_fol),os.path.splitext(os.path.basename(x))[0]+'*.tsai'))[0] for x in img_list]
        else:
            cam_list = [None]*len(img_list)
            if ba_prefix:
                # not yet implemented
                ba_prefix_list = [ba_prefix]*len(img_list)
        print("Mapping given images")
        ortho_logs = p_map(asp.mapproject,img_list,out_list,session_list,dem_list,tr_list,srs_list,cam_list)
        ortho_log = os.path.join(outdir,'ortho.log')
        print(f"Saving Orthorectification log at {ortho_log}")
        with open(ortho_log,'w') as f:
            for log in ortho_logs:
                f.write(log)
        if args.orthomosaic == 1:
            print("Will also produce median, weighted average and highest resolution orthomosaic")
            if args.data == 'triplet':
                for_img_list,nadir_img_list,aft_img_list,for_time,nadir_time,aft_time = skysat.sort_img_list(out_list)
                res_sorted_list = skysat.res_sort(out_list)
                res_sorted_mosaic = os.path.join(outdir,f'{for_time}_{nadir_time}_{aft_time}_finest_orthomosaic.tif')
                median_mosaic = os.path.join(outdir,f'{for_time}_{nadir_time}_{aft_time}_median_orthomosaic.tif')
                wt_avg_mosaic = os.path.join(outdir,f'{for_time}_{nadir_time}_{aft_time}_wt_avg_orthomosaic.tif')
                print("producing finest resolution on top mosaic, per-pixel median and wt_avg mosaic")
                all_3_view_mos_logs = p_map(asp.dem_mosaic, [res_sorted_list]*3, [res_sorted_mosaic,median_mosaic,wt_avg_mosaic], ['None']*3, [None]*3, ['first','median',None])
                res_sorted_log = asp.dem_mosaic(res_sorted_list,res_sorted_mosaic,tr='None',stats='first')
                print("producing idependent mosaic for different views in parallel")
                for_mosaic = os.path.join(outdir,f'{for_time}_for_first_mosaic.tif')
                nadir_mosaic = os.path.join(outdir,f'{nadir_time}_nadir_first_mosaic.tif')
                aft_mosaic = os.path.join(outdir,f'{aft_time}_aft_first_mosaic.tif')
                # prepare mosaics in parallel
                indi_mos_log = p_map(asp.dem_mosaic,[for_img_list,nadir_img_list,aft_img_list], [for_mosaic,nadir_mosaic,aft_mosaic], ['None']*3, [None]*3, ['first']*3)
                out_log = os.path.join(outdir,'science_mode_ortho_mos.log')
                total_mos_log = all_3_view_mos_logs+indi_mos_log
                print(f"Saving orthomosaic log at {out_log}")
                with open(out_log,'w') as f:
                    for log in itertools.chain.from_iterable(total_mos_log):
                        f.write(log)
            if args.data == 'video':
                res_sorted_list = skysat.res_sort(out_list)
                print("producing orthomasaic with finest on top")
                res_sorted_mosaic = os.path.join(outdir,f'video_finest_orthomosaic.tif') 
                print("producing orthomasaic with per-pixel median stats")
                median_mosaic = os.path.join(outdir,f'video_median_orthomosaic.tif')
                print("producing orthomosaic with weighted average statistics")
                wt_avg_mosaic = os.path.join(outdir,f'video_wt_avg_orthomosaic.tif')
                print("Mosaicing will be done in parallel")
                all_3_view_mos_logs = p_map(asp.dem_mosaic, [res_sorted_list]*3, [res_sorted_mosaic,median_mosaic,wt_avg_mosaic], ['None']*3, [None]*3, ['first','median',None])
                out_log = os.path.join(outdir,'science_mode_ortho_mos.log')
                print(f"Saving orthomosaic log at {out_log}")
                with open(out_log,'w') as f:
                    for log in all_3_view_mos_logs:
                        f.write(log)
            print("Script is complete!")

if __name__=='__main__':
    main()
           
