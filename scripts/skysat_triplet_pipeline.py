#! /usr/bin/env python
import subprocess
import argparse
import os,sys.glob
from distutils.spawn import find_executable
from skysat_stereo import asp_utils as asp

"""
Script for running the full pipeline based on workflow described in ISPRS 2020 submission
Need to specify input image folder, input refrence DEM folder
"""
def getparser():
    parser = argparse.ArgumentParser(description='Wrapper script to run full triplet stereo workflow')
    parser.add_argument('-in_img',default=None,type=str,help='path to Folder containing L1B imagery')
    parser.add_argument('-refdem',default=None,type=str,help='path to Reference DEM to use in processing')
    parser.add_argument('-outfolder',default=None,type=str,help='path to output folder to save results in')
    bin_choice = [1,0]
    parser.add_argument('-full_workflow,choices=bin_choice,type=int,default=1,help='Specify 1 to run full workflow (default: %(default)s)')
    parser.add_argument('-partial_workflow_steps',nargs='*',help='specify steps of workflow to run')
    return parser

def main():
    parser = getparser()
    args = parser.parse_args()
    img_folder = args.img_folder
    refdem = args.refdem
    outfol = os.path.join(args.outfolder,'proc_out')
    
    #Universal Args
    map = True
    if map:
        init_stereo_session = 'rpcmaprpc'
        init_ortho_session = 'rpc'
        final_stereo_session = 'pinholemappinhole'
        final_ortho_session = 'pinhole'
    else:
        init_stereo_session = 'rpc'
        init_ortho_session = 'rpc'
        final_stereo_session, final_ortho_session = ['nadirpinhole','pinhole']

    # For consistency, lets hardcode expected file names,folder names here :)
    # step1 outputs
    overlap_full_txt = os.path.join(outfol,'overlap.txt')
    overlap_full_pkl = os.path.splitext(overlap_full_txt)[0]+'_with_overlap_perc.pkl'
    overlap_stereo_pkl = os.path.splitext(overlap_full_pkl)[0]+'_stereo_only.pkl'
    overlap_stereo_txt = os.path.splitext(overlap_full_pkl)[0]+'_stereo_only.txt'
    bound_fn = os.path.splitext(overlap_full_txt)[0]+'_bound.gpkg'
    
    # step2 outputs
    cam_gcp_directory = os.path.join(out_fol,'camgen_cam_gcp')
    
    # step3 outputs
    init_ortho_dir = os.path.join(out_fol,'init_rpc_ortho')
    init_stereo_dir = os.path.join(out_fol,'init_rpc_stereo')
    
    # step4 bundle_adjust dense matches
    init_ba = os.path.join(out_fol,ba_pinhole)
  
    if args.full_workflow == 1:
         steps2run = np.arange(0,10) # run the entire 9 steps
    else:
        steps2run = np.array(args.partial_workflow_steps).astype(int)

	#workflow_steps
    # create output directory
    if not.os.path.exists(outfolder):
        os.makedirs(outfolder)
    if 1 in steps2run:
    print("Computing overlapping pairs")
    # Step 1 Compute overlapping pairs
    # Inputs: Image directory, minimum overlap percentage
    cmd = ['skysat_overlap.py','-img_folder',img_folder,'-percentage',overlap_perc,'-outfn',overlap_full_txt]
    asp.run_cmd(cmd)
    if 2 in steps2run:
        print("Generating Frame Cameras")
    frame_cam_cmd = ['skysat_preprocess.py','-mode','triplet','-t','rpc','-img',img_folder,'-outdir',cam_gcp_directory,
                    '-overlap_pkl',overlap_stereo_pkl,'-dem',refdem]
    asp.run_cmd(frame_cam_cmd)
    if 3 in steps2run:
        # specify whether to run using maprojected sessions or not
        map=True # switch this to False
        if map:
            # orthorectify all the images first
            print("Orthorectifying images using RPC camera")
            ortho_cmd = ['skysat_orthorectify.py', '-img_folder',img_folder,'-session',init_rpc_session,'-out_folder',init_rpc_ortho,
                        '-tsrs',tsrs,'-DEM',refdem,'-mode','science','-orthomosaic','0','-copy_rpc.py','1','-data','triplet']
            asp.run_cmd(ortho_cmd)
        print("Running stereo using RPC cameras")
        stereo_cmd = ['skysat_stereo_cli.py','-mode','triplet','-threads','4','-t','rpcmaprpc','-img',init_rpc_ortho,
                     '-overlap_pkl',overlap_stereo_pkl,'-dem',refdem,'-block','1','-crop_map','0','-outfol',init_stereo_dir]
        asp.run_cmd(stereo_cmd)

        # copy dense match file to ba directory
        dense_match_cmd = ['prep_dense_ba_run.py','-img'python ~/sw/src/skysat_3d/scripts/prep_dense_ba_run.py -img in_img/ -orig_pickle proc_out_rainier_lidar/overlap_with_overlap_perc.pkl -stereo_dir proc_out_rainier_lidar/init_rpc_stereo/ -ba_dir proc_out_rainier_lidar/ba_pinhole  -modify_overlap 0
   if 4 in steps2run:
       # copy  dense match to ba directory
       
