#! /usr/bin/env python

import os,sys,glob
import numpy as np
from skysat_stereo import asp_utils as asp
from skysat_stereo import skysat
from p_tqdm import p_map,p_umap
from multiprocessing import cpu_count
import argparse

# TODO:
# Determine best parameters for RPC generation

def get_parser():
        parser = argparse.ArgumentParser(description="utility to grid and register DEMs, pinhole cameras to a referece DEM, using ASP's ICP algorithm")
        mode_choice = ['gridding_only', 'classic_dem_align', 'multi_align', 'align_cameras']
        parser.add_argument('-mode', help='operation mode', choices=mode_choice, required=True)
        # gridding only choices
        parser.add_argument('-tr', default=2, type=float, help='DEM gridding resolution (default: %(default)s)')
        parser.add_argument('-tsrs', default='EPSG:325610', help='Projection for gridded DEM (default: %(default)s)')
        parser.add_argument('-point_cloud_list', nargs='*', help='List of pointclouds for gridding')
        # classic dem align options, also carried forward to multi_align
        align_choice = ['point-to-point', 'point-to-plane']
        parser.add_argument('-align', choices=align_choice, default='point-to-plane', help='ICP Alignment algorithm (defualt: %(default)s)')
        parser.add_argument('-max_displacement', default=100.0, type=float, help='Maximum allowable displacement between two DEMs (defualt: %(default)s)')
        trans_choice = [0, 1]
        parser.add_argument('-trans_only', default=0, type = int, choices=trans_choice, help='1: compute translation only, (default: %(default)s)')
        parser.add_argument('-outprefix', default=None, help='outprefix for alignment results')
        parser.add_argument('-refdem',default=None,help='DEM used as refrence for alignment')
        parser.add_argument('-source_dem',default = None,help = 'DEM to be aligned')
        parser.add_argument('-source_dem_list', nargs='*', help='List of source DEMs to be aligned to a common reference, used in multi_align operation mode')
        # Camera align args
        parser.add_argument('-transform', help='transfrom.txt file written by pc_align, used to align transform cameras to correct locations')
        parser.add_argument('-cam_list',nargs='*', help='list of cameras to be transformed')
        parser.add_argument('-outfol',help='folder where aligned cameras will be saved')
        parser.add_argument('-rpc', choices=trans_choice, default=0, type = int, help='1: also write out updated RPC, (default: %(default)s)')
        parser.add_argument('-dem', default='None', help='DEM used for generating RPC')
        parser.add_argument('-img_list', nargs='*', help='list of images for which RPC will be generated')
        return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    mode = args.mode
    if mode == 'gridding_only':
        tr = args.tr
        tsrs = args.tsrs
        point2dem_opts = asp.get_point2dem_opts(tr=tr, tsrs=tsrs)
        pc_list = args.point_cloud_list
        job_list = [point2dem_opts + [pc] for pc in pc_list]
        p2dem_log = p_map(asp.run_cmd,['point2dem'] * len(job_list), job_list, num_cpus = cpu_count())
        print(p2dem_log)
    if mode == 'classic_dem_align':
        ref_dem=args.refdem
        source_dem=args.source_dem
        max_displacement=args.max_displacement
        outprefix=args.outprefix
        align=args.align
        if args.trans_only == 0:
            trans_only=False
        else:
            trans_only=True
        asp.dem_align(ref_dem, source_dem, max_displacement, outprefix, align, trans_only)
    if mode == 'multi_align':
        """ Align multiple DEMs to a single source DEM """
        ref_dem=args.refdem
        source_dem_list=args.source_dem_list
        max_displacement=args.max_displacement
        outprefix_list=[f'{os.path.splitext(source_dem)[0]}_aligned_to{os.path.splitext(os.path.basename(ref_dem))[0]}' for source_dem in source_dem_list]
        align=args.align
        if args.trans_only == 0:
            trans_only=False
        else:
            trans_only=True
        n_source=len(source_dem_list)
        ref_dem_list=[ref_dem] * n_source
        max_disp_list=[max_displacement] * n_source
        align_list=[align] * n_source
        trans_list=[trans_only] * n_source
        p_umap(asp.dem_align,ref_dem_list,source_dem_list,max_disp_list,outprefix_list,align_list,trans_list,num_cpus = cpu_count())
    if mode == 'align_cameras':
        transform_txt = args.transform
        input_camera_list = args.cam_list
        n_cam=len(input_camera_list)
        if (args.rpc == 1) & (args.dem != 'None'):
            print("will also write rpc files")
            dem=args.dem
            img_list=arg.img_list
            rpc=True
        else:
            dem=None
            img_list=[None] * n_cam
            rpc=False
        transform_list=[transform_txt] * n_cam
        outfolder = args.outfol
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        outfolder=[outfolder] * n_cam
        write=[True] * n_cam
        rpc=[rpc] * n_cam
        dem=[dem] * n_cam
        p_umap(asp.align_cameras,input_camera_list,transform_list,outfolder,write,rpc,dem,img_list,num_cpus = cpu_count())

if __name__=="__main__":
    main()
