#! /usr/bin/env python
import subprocess
import argparse
from datetime import datetime
import os,sys,glob,shutil
from rpcm import geo
import numpy as np
import geopandas as gpd
from distutils.spawn import find_executable
from skysat_stereo import misc_geospatial as misc
from skysat_stereo import asp_utils as asp
from skysat_stereo import skysat_stereo_workflow as workflow

"""
Script for running the full pipeline based on workflow described in ISPRS 2020 submission
Need to specify input image folder, input refrence DEM folder
"""
#TODO:
# Add an option of cleaning up the lots of intermediate files produced

def getparser():
    parser = argparse.ArgumentParser(description='Wrapper script to run full triplet stereo workflow')
    parser.add_argument('-in_img',default=None,type=str,help='path to Folder containing L1B imagery')
    parser.add_argument('-orthodem',default=None,type=str,help='path to Reference DEM to use in orthorectification and camera resection, if not provided, will use coregdem')
    parser.add_argument('-coregdem',default=None,type=str,help='path to reference DEM to use in coregisteration')
    parser.add_argument('-mask_dem',default=1,type=int,choices=[1,0],help='mask reference DEM for static surfaces before coreg (default: %(default)s)')
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
        sys.exit()
    if not os.path.exists(coreg_dem):
        print(f"Coreg dem {coreg_dem} could not be located, exiting")
        sys.exit()
    if not os.path.exists(ortho_dem):
        print(f"Ortho dem {ortho_dem} could not be located, exiting")
        sys.exit()

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
    final_ortho_dir = os.path.join(out_fol,'georegistered_orthomosaics')
    
    # step 10, plot figure
    final_figure = os.path.join(out_fol,f"{job_name}_result.jpg")
    
    # step 11, experimental rpc production
    
    if args.full_workflow == 1:
         steps2run = np.arange(0,11) # run the entire 9 steps
    else:
        steps2run = np.array(args.partial_workflow_steps).astype(int)

	#workflow_steps
    # create output directory
    if not os.path.exists(out_fol):
        os.makedirs(out_fol)
    
    #copy reference DEM(s) to refdem directory
    # if parallel runs on different nodes use the same DEM, then will have issues
    refdem_dir = os.path.join(out_fol,'refdem')
    if not os.path.exists(refdem_dir):
        os.makedirs(refdem_dir)
    shutil.copy2(coreg_dem,os.path.join(refdem_dir,os.path.basename(coreg_dem)))
    if not coreg_dem == ortho_dem:
        shutil.copy2(ortho_dem,os.path.join(refdem_dir,os.path.basename(ortho_dem)))
    # replace old variable names
    coreg_dem = os.path.join(refdem_dir,os.path.basename(coreg_dem))
    ortho_dem = os.path.join(refdem_dir,os.path.basename(ortho_dem))


    if 1 in steps2run:
        print("Computing overlapping pairs")
        # Step 1 Compute overlapping pairs
        # Inputs: Image directory, minimum overlap percentage 
        overlap_perc = 0.01 # 1 percent essentially
        workflow.prepare_stereopair_list(img_folder,overlap_perc,overlap_full_txt)
        

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
    misc.clip_raster_by_shp_disk(coreg_dem,bound_buffer_fn)
    misc.ndvtrim_function(os.path.splitext(coreg_dem)[0]+'_shpclip.tif')
    coreg_dem = os.path.splitext(coreg_dem)[0]+'_shpclip_trim.tif'

    if ortho_dem != coreg_dem:
        misc.clip_raster_by_shp_disk(ortho_dem,bound_buffer_fn)
        misc.ndvtrim_function(os.path.splitext(ortho_dem)[0]+'_shpclip.tif')
        ortho_dem = os.path.splitext(ortho_dem)[0]+'_shpclip_trim.tif'    
    else:
        ortho_dem = coreg_dem


    if 2 in steps2run:
        print("Generating Frame Cameras")
        cam_gen_log = workflow.skysat_preprocess(img_folder,mode='triplet',
        product_level='l1b',overlap_pkl=overlap_stereo_pkl,dem=ortho_dem,
        outdir=cam_gcp_directory)

        now = datetime.now()
        log_fn = os.path.join(cam_gcp_directory,'camgen_{}.log'.format(now))
        print("saving subprocess camgen log at {}".format(log_fn))
        with open(log_fn,'w') as f:
        for log in cam_gen_log:
            f.write(log)

    if 3 in steps2run:
        # specify whether to run using maprojected sessions or not
   
        if map:
            # orthorectify all the images first
            print("Orthorectifying images using RPC camera")
            workflow.execute_skysat_orhtorectification(images=images_list,data='triplet',session=init_ortho_session,
                                                       outdir=init_ortho_dir,tsrs=epsg_code,dem=ortho_dem,mode='science',
                                                       overlap_list=None,copy_rpc=1,orthomosaic=0)
            init_stereo_input_img_folder = init_ortho_dir
        else:
            init_stereo_input_img_folder = img_folder
        print("Running stereo using RPC cameras")
        # Note crop_map = 0 option, this does not do warping to common extent and resolution for orthoimages before stereo, because we want to 
        # presrve this crucail information for correctly unwarped dense match points
        workflow.execute_skysat_stereo(init_stereo_input_img_folder,init_stereo_dir,
                                       mode='triplet',session=init_stereo_session,
                                       dem=ortho_dem,texture='normal',writeout_only=False,
                                       block=1,crop_map=0,threads=2,overlap_pkl=overlap_stereo_pkl,
                                       cross_track=False)
        

        # copy dense match file to ba directory
        workflow.dense_match_wrapper(stereo_master_dir=os.path.abspath(init_stereo_dir),
                                     ba_dir=os.path.abspath(init_ba),modify_overlap=0) 


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
            workflow.execute_skysat_orhtorectification(images=images_list,data='triplet',session=final_ortho_session,
                                                       outdir=intermediate_ortho_dir,tsrs=epsg_code,dem=ortho_dem,
                                                       ba_prefix=ba_prefix+'-run',mode='science',overlap_list=None,
                                                       copy_rpc=1,orthomosaic=0)
            print("Running intermediate orthorectification with bundle adjusted pinhole cameras")
            
            final_stereo_input_img_folder = intermediate_ortho_dir
        else:
            final_stereo_input_img_folder = img_folder
        # now run stereo
        print("Running final stereo reconstruction")
        workflow.execute_skysat_stereo(final_stereo_input_img_folder,
                                       final_stereo_dir,ba_prefix=ba_prefix+'-run',
                                       mode='triplet',session=final_stereo_session,
                                       dem=ortho_dem,texture='normal',writeout_only=False,
                                       block=args.block_matching,crop_map=1,threads=2,overlap_pkl=overlap_stereo_pkl,
                                       cross_track=False)
        

   
    if 6 in steps2run:

        pc_list = sorted(glob.glob(os.path.join(final_stereo_dir,'20*/2*/run-PC.tif'))) 
        print(f"Identified {len(pc_list)} clouds")
        
        # this is dem gridding followed by mosaicing
        workflow.gridding_wrapper(pc_list,tr=2)
        
        print("Mosaicing DEMs")
        
        workflow.dem_mosaic_wrapper(dir=os.path.abspath(final_stereo_dir),mode='triplet',
                                    out_folder=os.path.abspath(mos_dem_dir))

    if 7 in steps2run:
        # this is DEM alignment step
        # add option to mask coreg_dem for static surfaces
        # might want to remove glaciers, forest et al. before coregisteration
        # this can potentially be done in asp_utils step
        # actually use dem_mask.py with options of nlcd, nlcd_filter (not_forest) and of course RGI glacier polygons
        if args.mask_dem == 1: 
            # this might change for non-US sites, best to use bareground files
            mask_list = ['nlcd','glaciers']
            print("Masking reference DEM to static surfaces") 
            misc.dem_mask_disk(mask_list,os.path.abspath(coreg_dem))
            coreg_dem = os.path.splitext(coreg_dem)[0]+'_ref.tif'
        
        #now perform alignment
        median_mos_dem = glob.glob(os.path.join(mos_dem_dir,'multiview_*_median_mos.tif'))[0]
        print("Aligning DEMs")
        workflow.alignment_wrapper_single(coreg_dem,source_dem=median_mos_dem,max_displacement=40,
                                          outprefix=os.path.join(alignment_dir,'run'))
        
    if 8 in steps2run:
        # this steps aligns the frame camera models
        camera_list = sorted(glob.glob(os.path.join(init_ba,'run-run-*.tsai')))
        print(f"Detected {len(camera_list)} cameras to be registered to DEM")
        alignment_vector = glob.glob(os.path.join(alignment_dir,'alignment_vector.txt'))[0]
        print("Aligning cameras")
        workflow.align_cameras_wrapper(input_camera_list=camera_list,transform_txt=alignment_vector,
                                       outfolder=aligned_cam_dir)

    if 9 in steps2run:
        # this produces final georegistered orthomosaics
        georegistered_median_dem = glob.glob(os.path.join(alignment_dir,'run-trans_*DEM.tif'))[0]
        print("Running final orthomsaic creation")
        workflow.execute_skysat_orhtorectification(images=images_list,data='triplet',session=final_ortho_session,
                                                       outdir=final_ortho_dir,tsrs=epsg_code,dem=georegistered_median_dem,
                                                       ba_prefix=os.path.join(aligned_cam_dir,'run-run'),mode='science',
                                                       overlap_list=None,copy_rpc=0,orthomosaic=1)
        
    if 10 in steps2run:
        # this produces a final plot of orthoimage,DEM, NMAD and countmaps
        ortho = glob.glob(os.path.join(final_ortho_dir,'*finest_orthomosaic.tif'))[0]
        count = glob.glob(os.path.join(mos_dem_dir,'*count*.tif'))[0]
        nmad = glob.glob(os.path.join(mos_dem_dir,'*nmad*.tif'))[0]
        georegistered_median_dem = glob.glob(os.path.join(alignment_dir,'run-trans_*DEM.tif'))[0]
        print("plotting final figure")
        misc.plot_composite_fig(ortho,georegistered_median_dem,count,nmad,outfn=final_figure)

if __name__ == '__main__':
    main()
