#! /usr/bin/env python

import os,sys,glob
import numpy as np
from skysat_stereo import asp_utils as asp
from skysat_stereo import skysat
from skysat_stereo import skysat_stereo_workflow as workflow
from p_tqdm import p_map,p_umap
import psutil
import argparse
from rpcm import geo
from pygeotools.lib import iolib,geolib



def get_parser():
        parser = argparse.ArgumentParser(description="utility to grid and register DEMs, pinhole cameras to a referece DEM, using ASP's ICP algorithm")
        mode_choice = ['gridding_only', 'classic_dem_align', 'multi_align', 'align_cameras']
        parser.add_argument('-mode', help='operation mode', choices=mode_choice, required=True)
        # gridding only choices
        parser.add_argument('-tr', default=2, type=float, help='DEM gridding resolution (default: %(default)s)')
        parser.add_argument('-tsrs', default=None, help='Projection for gridded DEM if not using local UTM (default: %(default)s)')
        parser.add_argument('-point_cloud_list', nargs='*', help='List of pointclouds for gridding')
        # classic dem align options, also carried forward to multi_align
        align_choice = ['point-to-point', 'point-to-plane']
        parser.add_argument('-align', choices=align_choice, default='point-to-plane', help='ICP Alignment algorithm (defualt: %(default)s)')
        parser.add_argument('-initial_align',default=None,type=str,help='Alignment transform from initial PC align run')
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
        workflow.grdding_wrapper(args.point_cloud_list,args.tr,args.tsrs)
    elif mode == 'classic_dem_align':
        workflow.alignment_wrapper_single(args.refdem,args.source_dem,args.max_displacement,args.outprefix,
                             args.align,args.trans_only,initial_align=args.initial_align)
    elif mode == 'multi_align':
        workflow.alignment_wrapper_multi(args.refdem,args.source_dem_list,args.max_displacement,args.align,
                            trans_only=args.trans_only,initial_align=args.initial_align)
    elif mode == 'align_cameras':
        workflow.align_cameras_wrapper(args.cam_list,args.transform,args.outfol,rpc=args.rpc,
                                       dem=args.dem,img_list=args.img_list)
if __name__=="__main__":
    main()