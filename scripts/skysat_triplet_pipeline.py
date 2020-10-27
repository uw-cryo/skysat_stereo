#! /usr/bin/env python
import subprocess
import argparse
import os,sys,glob
from rpcm import geo
import numpy as np
import geopandas as gpd
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
    parser.add_argument('-mask_dem',default=1,type=int,choices=[1,0],help='mask reference DEM for static surfaces before coreg (default: %(default)s'))
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
    img_folder = args.in_img
    coreg_dem = args.coregdem
    if args.orthodem is not None:
        ortho_dem = args.orthodem
    else:
        ortho_dem = coreg_dem
    # Check for input files
    img_list = glob.glob(os.path.join(img_folder,'*.tif'))+glob.glob(os.path.join(img_folder,'*.tiff'))
    if len(img_list)<2:
        print(f"Only {len(img_list)} images detected, exiting")
    if not os.path.exists(coreg_dem):
        print(f"Coreg dem {coreg_dem} could not be located, exiting")
    if not os.path.exists(ortho_dem):
        print(f"Ortho dem {ortho_dem} could not be located, exiting")
    
    # structure for output folder
    out_fol = os.path.join(args.outfolder,'proc_out')
    job_name = args.job_name
    
    #Universal Args
    if args.ortho_workflow == 1:
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
    overlap_full_txt = os.path.join(out_fol,'overlap.txt')
    overlap_full_pkl = os.path.splitext(overlap_full_txt)[0]+'_with_overlap_perc.pkl'
    overlap_stereo_pkl = os.path.splitext(overlap_full_pkl)[0]+'_stereo_only.pkl'
    overlap_stereo_txt = os.path.splitext(overlap_full_pkl)[0]+'_stereo_only.txt'
    bound_fn = os.path.splitext(overlap_full_txt)[0]+'_bound.gpkg'
    bound_buffer_fn = os.path.splitext(bound_fn)[0]+'_1km_buffer.gpkg'

    # step2 outputs
    cam_gcp_directory = os.path.join(out_fol,'camgen_cam_gcp')
    
    # step3 outputs
    init_ortho_dir = os.path.join(out_fol,'init_rpc_ortho')
    init_stereo_dir = os.path.join(out_fol,'init_rpc_stereo')
    
    # step4 bundle_adjust dense matches
    init_ba = os.path.join(out_fol,'ba_pinhole')
    ba_prefix = os.path.join(init_ba,'run')
    
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
    if not os.path.exists(out_fol):
        os.makedirs(out_fol)


    if 1 in steps2run:
        print("Computing overlapping pairs")
        # Step 1 Compute overlapping pairs
        # Inputs: Image directory, minimum overlap percentage 
        overlap_perc = 0.01 # 1 percent essentially
        cmd = ['-img_folder',img_folder,'-percentage',str(overlap_perc),'-outfn',overlap_full_txt]
        asp.run_cmd('skysat_overlap.py',cmd)
    print("Computing Target UTM zones for orthorectification")
    gdf = gpd.read_file(bound_fn)
    clon,clat = [gdf.centroid.x.values,gdf.centroid.y.values]
    epsg_code = f'EPSG:{geo.compute_epsg(clon,clat)}'
    print(f"Detected UTM zone is {epsg_code}")
    if not os.path.exists(bound_buffer_fn):
        print("Creating buffered shapefile")
        gdf_proj = gdf.to_crs(epsg_code)
        gdf_proj['geometry'] = gdf_proj.buffer(1000)
        gdf_proj.to_file(bound_buffer_fn,driver='GPKG')

    print("Cropping reference DEMs to extent of SkySat footprint + 1 km buffer")
    asp.run_cmd('clip_raster_by_shp.py',[coreg_dem,bound_buffer_fn])
    asp.run_cmd('trim_ndv.py',[os.path.splitext(coreg_dem)[0]+'_shpclip.tif'])
    coreg_dem = os.path.splitext(coreg_dem)[0]+'_shpclip_trim.tif'
    if ortho_dem != coreg_dem:
        clip_log = asp.run_cmd('clip_raster_by_shp.py',[ortho_dem,bound_buffer_fn])
        print(clip_log)
        asp.run_cmd('trim_ndv.py',[os.path.splitext(ortho_dem)[0]+'_shpclip.tif'])
        ortho_dem = os.path.splitext(ortho_dem)[0]+'_shpclip_trim.tif'    
    else:
        ortho_dem = coreg_dem


    if 2 in steps2run:
        print("Generating Frame Cameras")
        frame_cam_cmd = ['-mode','triplet','-t','rpc','-img',img_folder,'-outdir',cam_gcp_directory,
                    '-overlap_pkl',overlap_stereo_pkl,'-dem',ortho_dem]
        asp.run_cmd('skysat_preprocess.py',frame_cam_cmd)
    if 3 in steps2run:
        # specify whether to run using maprojected sessions or not
   
        if map:
            # orthorectify all the images first
            print("Orthorectifying images using RPC camera")
            ortho_cmd = ['-img_folder',img_folder,'-session',init_ortho_session,'-out_folder',init_ortho_dir,
                        '-tsrs',epsg_code,'-DEM',ortho_dem,'-mode','science','-orthomosaic','0','-copy_rpc','1','-data','triplet']
            #Note above, copy_rpc = 1, because we want the orthoimages to have RPC info embedded in gdal header for stereo later
            asp.run_cmd('skysat_orthorectify.py',ortho_cmd)
            init_stereo_input_img_folder = init_ortho_dir
        else:
            init_stereo_input_img_folder = img_folder
        print("Running stereo using RPC cameras")
        stereo_cmd = ['-mode','triplet','-threads','2','-t',init_stereo_session,'-img',init_stereo_input_img_folder,
                     '-overlap_pkl',overlap_stereo_pkl,'-dem',ortho_dem,'-block','1','-crop_map','0','-outfol',init_stereo_dir]
        # Note crop_map = 0 option, this does not do warping to common extent and resolution for orthoimages before stereo, because we want to 
        # presrve this crucail information for correctly unwarped dense match points
        asp.run_cmd('skysat_stereo_cli.py',stereo_cmd)

        # copy dense match file to ba directory
        dense_match_cmd = ['-img', img_folder, '-orig_pickle', overlap_stereo_pkl, '-stereo_dir', init_stereo_dir, 
                          '-ba_dir', init_ba, '-modify_overlap','0']
        asp.run_cmd('prep_dense_ba_run.py',dense_match_cmd)


    if 4 in steps2run:
        # this is bundle adjustment step
        # we use dense files copied from previous step
        ba_prefix = os.path.join(init_ba,'run')
        ba_cmd = ['-mode', 'full_triplet', '-t', 'nadirpinhole', '-img', img_folder, 
                  '-cam', cam_gcp_directory, '-overlap_list', overlap_stereo_txt, '-num_iter', '700', '-num_pass', '2','-ba_prefix',ba_prefix]
        print("running bundle adjustment")
        asp.run_cmd('ba_skysat.py',ba_cmd)


    if 5 in steps2run:
        # this is where final stereo will take place
        # first we orthorectify again, if map = True
        if map:
            print("Running intermediate orthorectification with bundle adjusted pinhole cameras")
            ortho_cmd = ['-img_folder',img_folder,'-session',final_ortho_session,'-out_folder',intermediate_ortho_dir,
                        '-tsrs',epsg_code,'-DEM',ortho_dem,'-mode','science','-orthomosaic','0','-data','triplet','-ba_prefix',ba_prefix+'-run']
            asp.run_cmd('skysat_orthorectify.py',ortho_cmd)
            final_stereo_input_img_folder = intermediate_ortho_dir
        else:
            final_stereo_input_img_folder = img_folder
        # now run stereo
        stereo_cmd = ['-mode','triplet','-threads','2','-t',final_stereo_session,'-img',final_stereo_input_img_folder,
                     '-overlap_pkl',overlap_stereo_pkl,'-dem',ortho_dem, '-crop_map','1', '-outfol', final_stereo_dir, 
                      '-ba_prefix',ba_prefix+'-run','-block',str(args.block_matching)]
        print("Running final stereo reconstruction")
        asp.run_cmd('skysat_stereo_cli.py',stereo_cmd)

   
    if 6 in steps2run:

        pc_list = sorted(glob.glob(os.path.join(final_stereo_dir,'20*/2*/run-PC.tif'))) 
        print(f"Identified {len(pc_list)} clouds")
        # this is dem gridding followed by mosaicing
        dem_grid_cmd = ['-mode','gridding_only', '-tr', '2', '-point_cloud_list'] + pc_list
        
        asp.run_cmd('skysat_pc_cam.py',dem_grid_cmd)
        print("Mosaicing DEMs")
        dem_mos_cmd = ['-mode','triplet','-DEM_folder',final_stereo_dir,'-out_folder',mos_dem_dir]
        asp.run_cmd('skysat_dem_mos.py',dem_mos_cmd)


    if 7 in steps2run:

        # this is DEM alignment step
        # add option to mask coreg_dem for static surfaces
        # might want to remove glaciers, forest et al. before coregisteration
        # this can potentially be done in asp_utils step
        # actually use dem_mask.py with options of nlcd, nlcd_filter (not_forest) and of course RGI glacier polygons

        median_mos_dem = glob.glob(os.path.join(mos_dem_dir,'triplet_median_mos.tif'))[0]
        dem_align_cmd = ['-mode','classic_dem_align','-max_displacement','100','-refdem',coreg_dem,
                         '-source_dem',median_mos_dem,'-outprefix',os.path.join(alignment_dir,'run')]
        print("Aligning DEMs")
        asp.run_cmd('skysat_pc_cam.py',dem_align_cmd)
    if 8 in steps2run:
        # this steps aligns the frame camera models
        camera_list = sorted(glob.glob(os.path.join(init_ba,'run-run-*.tsai')))
        print(f"Detected {len(camera_list)} cameras to be registered to DEM")
        alignment_vector = glob.glob(os.path.join(alignment_dir,'alignment_vector.txt'))[0]
        camera_align_cmd = ['-mode','align_cameras','-transform',alignment_vector,
                            '-outfol',aligned_cam_dir,'-cam_list']+camera_list
        print("Aligning cameras")
        asp.run_cmd('skysat_pc_cam.py',camera_align_cmd)
    if 9 in steps2run:
        # this produces final georegistered orthomosaics
        georegistered_median_dem = glob.glob(os.path.join(alignment_dir,'run-trans_*DEM.tif'))[0]
        ortho_cmd = ['-img_folder',img_folder,'-session',final_ortho_session,'-out_folder',final_ortho_dir,
                        '-tsrs',epsg_code,'-DEM',georegistered_median_dem,'-mode','science','-orthomosaic','1','-data','triplet',
		     '-ba_prefix',os.path.join(aligned_cam_dir,'run-run')]
        print("Running final orthomsaic creation")
        asp.run_cmd('skysat_orthorectify.py',ortho_cmd)

if __name__ == '__main__':
    main()
