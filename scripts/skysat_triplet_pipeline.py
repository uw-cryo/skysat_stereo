#! /usr/bin/env python
import subprocess
import argparse
import os,sys,glob
from rpcm import geo
from distutils.spawn import find_executable
from skysat_stereo import asp_utils as asp

"""
Script for running the full pipeline based on workflow described in ISPRS 2020 submission
Need to specify input image folder, input refrence DEM folder
"""
def getparser():
    parser = argparse.ArgumentParser(description='Wrapper script to run full triplet stereo workflow')
    parser.add_argument('-in_img',default=None,type=str,help='path to Folder containing L1B imagery')
    parser.add_argument('-orthodem',default=None,type=str,help='path to Reference DEM to use in orthorectification and camera resection, if not provided, will use coregdem')
    parser.add_argument('-coregdem',default=None,type=str,help='path to reference DEM to use in coregisteration')
    parser.add_argument('-ortho_workflow',default=1,type=int,choices=[1,0],help='option to orthorectify before stereo or not')
    parser.add_argument('-block_matching',default=0,type=int,choices=[1,0],help='whether to use block matching in final stereo matching, default is 0 (not)')
    parser.add_argument('-job_name',default=None,type=str,help='identifier for output folder and final composite products')
    parser.add_argument('-outfolder',default=None,type=str,help='path to output folder to save results in')
    bin_choice = [1,0]
    parser.add_argument('-full_workflow',choices=bin_choice,type=int,default=1,help='Specify 1 to run full workflow (default: %(default)s)')
    parser.add_argument('-partial_workflow_steps',nargs='*',help='specify steps of workflow to run')
    return parser

def main():
    parser = getparser()
    args = parser.parse_args()
    img_folder = args.img_folder
    coerg_dem = args.coregdem
    if args.orthodem is not None:
        ortho_dem = args.orthodem
    else:
        ortho_dem = coreg_dem
    outfol = os.path.join(args.outfolder,'proc_out')
    job_name = args.job_name
    
    #Universal Args
    if args.orthoworkflow == 1:
        map = True
    else:
        map = False
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
    
    # step5 stereo_args
    intermediate_ortho_dir = os.path.join(out_fol,'intermediate_pinhole_ortho')
    final_stereo_dir = os.path.join(out_fol,'final_pinhole_stereo')

    # step 6, dem gridding and mosaicing
    mos_dem_dir = os.path.join(final_stereo_dir,'composite_dems')

    # step 7. dem_alignment
    alignment_dir = os.path.join(out_fol,'georegistered_dem_mos')
			
    # step 8, camera alignment
    aligned_cam_dir = os.path.join(out_fol,'georegistered_cameras')
			
    # step 9, final orthorectification
    final_ortho_dir = os.path.join(out_fol,'georegisterd_orthomosaics')

    # step 10, experimental rpc production
    
    if args.full_workflow == 1:
         steps2run = np.arange(0,10) # run the entire 9 steps
    else:
        steps2run = np.array(args.partial_workflow_steps).astype(int)

	#workflow_steps
    # create output directory
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    if 1 in steps2run:
        print("Computing overlapping pairs")
        # Step 1 Compute overlapping pairs
        # Inputs: Image directory, minimum overlap percentage 
        overlap_perc = 0.01 # 1 percent essentially
        cmd = ['skysat_overlap.py','-img_folder',img_folder,'-percentage',overlap_perc,'-outfn',overlap_full_txt]
        asp.run_cmd(cmd)
    print("Computing Target UTM zones for orthorectification")
    gdf = gpd.read_file(bound_fn)
    clon,clat = gdf.to_crs({'init':'epsg:4326'}).centroid
    epsg_code = f'EPSG:{geo.compute_epsg(clon,clat)}'
       
    
    if 2 in steps2run:
        print("Generating Frame Cameras")
        frame_cam_cmd = ['skysat_preprocess.py','-mode','triplet','-t','rpc','-img',img_folder,'-outdir',cam_gcp_directory,
                    '-overlap_pkl',overlap_stereo_pkl,'-dem',ortho_dem]
        asp.run_cmd(frame_cam_cmd)
    if 3 in steps2run:
        # specify whether to run using maprojected sessions or not
   
        if map:
            # orthorectify all the images first
            print("Orthorectifying images using RPC camera")
            ortho_cmd = ['skysat_orthorectify.py', '-img_folder',img_folder,'-session',init_ortho_session,'-out_folder',init_rpc_ortho,
                        '-tsrs',epsg_code,'-DEM',ortho_dem,'-mode','science','-orthomosaic','0','-copy_rpc.py','1','-data','triplet']
            #Note above, copy_rpc = 1, because we want the orthoimages to have RPC info embedded in gdal header for stereo later
            asp.run_cmd(ortho_cmd)
            init_stereo_input_img_folder = init_rpc_ortho
        else:
            init_stereo_input_img_folder = img_folder
        print("Running stereo using RPC cameras")
        stereo_cmd = ['skysat_stereo_cli.py','-mode','triplet','-threads','2','-session',init_stereo_session,'-img',init_stereo_input_img_folder,
                     '-overlap_pkl',overlap_stereo_pkl,'-dem',ortho_dem,'-block','1','-crop_map','0','-outfol',init_stereo_dir]
        # Note crop_map = 0 option, this does not do warping to common extent and resolution for orthoimages before stereo, because we want to 
        # presrve this crucail information for correctly unwarped dense match points
        asp.run_cmd(stereo_cmd)

        # copy dense match file to ba directory
        dense_match_cmd = ['prep_dense_ba_run.py','-img', img_folder, '-orig_pickle', overlap_stereo_pkl, '-stereo_dir', init_rpc_stereo, 
                          '-ba_dir', init_ba, '-modify_overlap','0']
        asp.run_cmd(dense_match_cmd)

    if 4 in steps2run:
        # this is bundle adjustment step
        # we use dense files copied from previous step
        ba_prefix = os.path.join(init_ba,'run')
        ba_cmd = ['ba_skysat.py', '-mode', 'full_triplet', '-t', 'nadirpinhole', 'img', img_folder, 
                  '-cam', cam_gcp_directory, '-overlap_list', overlap_stereo_txt, 'num_iter', '700', '-num_pass', '2','-ba_prefix',ba_prefix]
        asp.run_cmd(ba_cmd)
    if 5 in steps2run:
        # this is where final stereo will take place
        # first we orthorectify again, if map = True
        if map:
            ortho_cmd = ['skysat_orthorectify.py', '-img_folder',img_folder,'-session',final_ortho_session,'-out_folder',intermediate_pin_ortho,
                        '-tsrs',tsrs,'-DEM',ortho_dem,'-mode','science','-orthomosaic','0','-data','triplet','-ba_prefix',ba_prefix+'-run']
            asp.run_cmd(ortho_cmd)
            final_stereo_input_img_folder = intermediate_pin_ortho
        else:
            final_stereo_input_img_folder = img_folder
        # now run stereo
        stereo_cmd = ['skysat_stereo_cli.py','-mode','triplet','-threads','2','-t',final_stereo_session,'-img',final_stereo_input_img_folder,
                     '-overlap_pkl',overlap_stereo_pkl,'-dem',refdem,'-block', '0', '-crop_map','1', '-outfol', final_stereo_dir, 
                      '-ba_prefix',ba_prefix+'-run','-block',args.block_matching]
        asp.run_cmd(stereo_cmd)
   
    if 6 in steps2run:
        pc_list = sorted(glob.glob(os.path.join(final_stereo_dir,'20*/2*/run-PC.tif'))) 
        # this is dem gridding followed by mosaicing
        dem_grid_cmd = ['skysat_pc_cam.py', '-mode','gridding_only', '-tr', '2', '-point_cloud_list', pc_list]
        asp.run_cmd(dem_grid_cmd)
        dem_mos_cmd = ['skysat_dem_mos.py','-mode','triplet','-DEM_folder',final_stereo_dir,'-out_folder',mos_dem_dir]
        asp.run_cmd(dem_mos_cmd)
    if 7 in steps2run:
        # this is DEM alignment step
        # add option to mask coreg_dem for static surfaces
        # might want to remove glaciers, forest et al. before coregisteration
        # this can potentially be done in asp_utils step
        median_mos_dem = glob.glob(os.path.join(mos_dem_dir,'triplet_median_mos.tif'))[0]
        dem_align_cmd = ['skysat_pc_cam.py','-mode','classic_dem_align','-max_displacement','100','-refdem',coreg_dem,
                         '-srcdem',median_mos_dem,'-outprefix',os.path.join(alignment_dir,'run')]
      
        asp.run_cmd(dem_align_cmd)
    if 8 in steps2run:
        # this steps aligns the frame camera models
        camera_list = sorted(glob.glob(os.path.join(init_ba,'run-run-*.tsai')))
        alignment_vector = glob.glob(os.path.join(alignment_dir,'alignment_vector.txt'))[0]
        camera_align_cmd = ['skysat_pc_cam.py','-mode','align_cameras','-transform',alignment_vector,
                            '-cam_list',camera_list,'-outfol',aligned_cam_dir]
        asp.run_cmd(camera_align_cmd)
    if 9 in steps2run:
        # this produces final georegistered orthomosaics
        georegistered_median_dem = glob.glob(os.path.join(alignment_dir,'run-trans_*DEM.tif'))[0]
        ortho_cmd = ['skysat_orthorectify.py', '-img_folder',img_folder,'-session',final_ortho_session,'-out_folder',final_ortho_dir,
                        '-tsrs',tsrs,'-DEM',georegistered_median_dem,'-mode','science','-orthomosaic','1','-data','triplet',
		     '-ba_prefix',os.path.join(aligned_cam_dir,'run-run')]
        asp.run_cmd(ortho_cmd)
