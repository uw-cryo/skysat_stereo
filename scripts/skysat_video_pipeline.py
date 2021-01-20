#! /usr/bin/env python
import subprocess
import argparse
import os,sys,glob,shutil
from rpcm import geo
import numpy as np
import geopandas as gpd
from distutils.spawn import find_executable
from skysat_stereo import misc_geospatial as misc
from skysat_stereo import asp_utils as asp
from skysat_stereo import skysat

"""
Script for running the full pipeline based on workflow described in ISPRS 2020 submission
Need to specify input image folder, input refrence DEM folder
"""
def getparser():
    parser = argparse.ArgumentParser(description='Wrapper script to run full video workflow')
    parser.add_argument('-in_img',default=None,type=str,help='path to Folder containing L1A imagery')
    parser.add_argument('-frame_index',default=None,type=str,help='path to frame_index.csv containing atitude and ephmeris information')
    parser.add_argument('-orthodem',default=None,type=str,help='path to Reference DEM to use in orthorectification and camera resection, if not provided, will use coregdem')
    parser.add_argument('-produce_low_res_for_ortho',type=int,choices=[1,0],default = 1,
                       help='use hole-filled low res DEM produced from bundle-adjusted camera for orthorectification, (default: %(default)s)')
    parser.add_argument('-coregdem',default=None,type=str,help='path to reference DEM to use in coregisteration')
    parser.add_argument('-mask_dem',default=1,type=int,choices=[1,0],help='mask reference DEM for static surfaces before coreg (default: %(default)s)')
    parser.add_argument('-ortho_workflow',default=1,type=int,choices=[1,0],help='option to orthorectify before stereo or not')
    parser.add_argument('-block_matching',default=0,type=int,choices=[1,0],help='whether to use block matching in final stereo matching, default is 0 (not)')
    parser.add_argument('-mvs', default=0, type=int, choices=[1,0], help='1: Use multiview stereo triangulation for video data\
                         , do matching with next 20 slave for each master image/camera (defualt: %(default)s')
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
    frame_index = args.frame_index

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
    if not os.path.exists(frame_index):
        print(f"Input frame index file {frame_index} file could not be located, exiting")
        sys.exit()

    # structure for output folder
    out_fol = os.path.join(args.outfolder,'proc_out')
    job_name = args.job_name
    
    #Universal Args
    if args.ortho_workflow == 1:
        map = True
    else:
        map = False

    # For consistency, lets hardcode expected file names,folder names here 
    bound_fn = os.path.join(out_fol,job_name+'_bound_2km.gpkg')

    # step1 outputs
    # this is preprocessing step
    cam_gcp_directory = os.path.join(out_fol,'camgen_cam_gcp')
    
    # step2 outputs
    # this is bundle_adjustment step
    init_ba = os.path.join(out_fol,'ba_pinhole')
    ba_prefix = os.path.join(init_ba,'run')
    
    # step3 outputs
    # this is stereo reconstruction step
    init_ortho_dir = os.path.join(out_fol,'init_ortho')
    init_stereo_dir = os.path.join(out_fol,'init_block_stereo')
    intermediate_ortho_dir = os.path.join(out_fol,'intermediate_ortho')
    final_stereo_dir = os.path.join(out_fol,'final_pinhole_stereo')

    # step4, dem gridding and mosaicing
    mos_dem_dir = os.path.join(final_stereo_dir,'composite_dems')

    # step5, dem_alignment
    alignment_dir = os.path.join(out_fol,'georegistered_dem_mos')
			
    # step6, camera alignment
    aligned_cam_dir = os.path.join(out_fol,'georegistered_cameras')
			
    # step7, final orthorectification
    final_ortho_dir = os.path.join(out_fol,'georegistered_orthomosaics')

    # step 8, plot figure
    final_figure = os.path.join(out_fol,f"{job_name}_result.jpg") 

    # step 10, experimental rpc production
    
    if args.full_workflow == 1:
         steps2run = np.arange(0,10) # run the entire 9 steps
    else:
        steps2run = np.array(args.partial_workflow_steps).astype(int)

	#workflow_steps
    # create output directory
    if not os.path.exists(out_fol):
        os.makedirs(out_fol)
    # copy coreg_dem and ortho_dem to folder
    # if parallel runs on different nodes use the same DEM, then will have issues
    refdem_dir = os.path.join(out_fol,'refdem')
    if not os.path.exists(refdem_dir):
        os.makedirs(refdem_dir)
    shutil.copy2(coreg_dem,os.path.join(refdem_dir,os.path.basename(coreg_dem)))
    shutil.copy2(ortho_dem,os.path.join(refdem_dir,os.path.basename(ortho_dem)))
    # replace old variable names
    coreg_dem = os.path.join(refdem_dir,os.path.basename(coreg_dem))
    ortho_dem = os.path.join(refdem_dir,os.path.basename(ortho_dem))

    print("Computing Target UTM zones for orthorectification")
    gdf_frame_index = skysat.parse_frame_index(frame_index)
    gdf_buffer = gpd.GeoDataFrame({'idx':[0],'geometry':gdf_frame_index.unary_union},crs={'init':'epsg:4326'})
    clon,clat = [gdf_buffer.centroid.x.values[0],gdf_buffer.centroid.y.values[0]]
    epsg_code = f'EPSG:{geo.compute_epsg(clon,clat)}'
    print(f"Detected UTM zone is {epsg_code}")
    if not os.path.exists(bound_fn):
        print("Creating buffered shapefile")
        gdf_proj = gdf_buffer.to_crs(epsg_code)
        # buffer by 2 km
        gdf_proj['geometry'] = gdf_proj.buffer(2000)
        gdf_proj.to_file(bound_fn,driver='GPKG')

    print("Cropping reference DEMs to extent of SkySat footprint + 1 km buffer")
    asp.run_cmd('clip_raster_by_shp.py',[coreg_dem,bound_fn])
    asp.run_cmd('trim_ndv.py',[os.path.splitext(coreg_dem)[0]+'_shpclip.tif'])
    coreg_dem = os.path.splitext(coreg_dem)[0]+'_shpclip_trim.tif'
    if ortho_dem != coreg_dem:
        clip_log = asp.run_cmd('clip_raster_by_shp.py',[ortho_dem,bound_fn])
        print(clip_log)
        asp.run_cmd('trim_ndv.py',[os.path.splitext(ortho_dem)[0]+'_shpclip.tif'])
        ortho_dem = os.path.splitext(ortho_dem)[0]+'_shpclip_trim.tif'    
    else:
        ortho_dem = coreg_dem


    if 1 in steps2run:
        print("Sampling video sequence and generating Frame Cameras")
        frame_cam_cmd = ['-mode','video','-t','pinhole','-img',img_folder,'-outdir',cam_gcp_directory,
                    '-video_sampling_mode','num_images','-sampler','60','-frame_index',frame_index,
                    '-product_level', 'l1a','-dem',ortho_dem]
        print(frame_cam_cmd)
        asp.run_cmd('skysat_preprocess.py',frame_cam_cmd)
    # read the frame_index.csv which contains the info for sampled scenes only
    print(cam_gcp_directory)
    frame_index = glob.glob(os.path.join(cam_gcp_directory,'*frame*.csv'))[0]

    if 2 in steps2run:
        # this is bundle adjustment step
        ba_cmd = ['-mode', 'full_video', '-t', 'nadirpinhole', '-img', img_folder, '-gcp',cam_gcp_directory,
                  '-cam', cam_gcp_directory, '-frame_index', frame_index, '-num_iter', '2000', 
                  '-num_pass', '3','-ba_prefix',ba_prefix]
        print("Running bundle adjustment for the input video sequence")
        asp.run_cmd('ba_skysat.py',ba_cmd)
        

    if 3 in steps2run:
        # this is stereo step
        # we need to check for 2 steps
        # is map turned to true ?
        # if map true, is low resolution block matching DEM to be used in stereo ?
        # so lets process first assuming map is untrue
        if not map:
            if args.mvs == 1:
                print("MVS not implemented on non-orthorectified scenes, exiting")
                sys.exit()
            stereo_cmd = ['-mode','video','-threads','2','-t','nadirpinhole','-img',img_folder,
                     '-frame_index',frame_index,'-outfol', final_stereo_dir, '-sampling_interval','10',
                      '-ba_prefix',ba_prefix+'-run','-full_extent','1']
            asp.run_cmd(stereo_cmd)
        else:
            if args.produce_low_res_for_ortho == 1:
                # will need to produce low res dem using block matching on L1A images and bundle adjusted cameras
                # this was used for the 2 St. Helen's case studies in SkySat stereo manuscript
                stereo_cmd = ['-mode','video','-threads','2','-t','nadirpinhole','-img',img_folder,
                         '-frame_index',frame_index,'-outfol', init_stereo_dir, '-sampling_interval','10',
                         '-ba_prefix',ba_prefix+'-run','-full_extent','1','-block','1','-texture','low']
                print("Running stereo with block matching for producing orthorectification DEM")
                asp.run_cmd('skysat_stereo_cli.py',stereo_cmd)
                # query point clouds
                pc_list = sorted(glob.glob(os.path.join(init_stereo_dir,'12*/run-PC.tif')))
                # grid into DEMs
                grid_cmd = ['-mode','gridding_only','-tr','4','-tsrs',epsg_code,'-point_cloud_list'] + pc_list
                print("Gridding block matching point clouds")
                asp.run_cmd('skysat_pc_cam.py',grid_cmd)
                dem_list = sorted(glob.glob(os.path.join(init_stereo_dir,'12*/run-DEM.tif')))
                hole_filled_low_res_dem = os.path.join(init_stereo_dir,'block_matching_hole_filled_dem_mos.tif')
                mos_cmd = ['--dem-blur-sigma','9','--median','-o', hole_filled_low_res_dem]
                asp.run_cmd('dem_mosaic', mos_cmd+dem_list)
                dem_for_ortho = hole_filled_low_res_dem

            else:
                # this argument will use input orhtodem (used for camera resection) as input for orthorectification
                dem_for_ortho = ortho_dem

            print("Running intermediate orthorectification")
            ortho_cmd = ['-img_folder',img_folder,'-session','pinhole','-out_folder',intermediate_ortho_dir,
                        '-tsrs',epsg_code,'-DEM',dem_for_ortho,'-mode','science','-orthomosaic','0','-data','video',
                        '-frame_index',frame_index,'-ba_prefix',ba_prefix+'-run']
            asp.run_cmd('skysat_orthorectify.py',ortho_cmd)

            ## Now run final stereo
            stereo_cmd = ['-mode','video','-threads','2','-t','pinholemappinhole','-img',intermediate_ortho_dir,
                         '-frame_index',frame_index,'-outfol', final_stereo_dir, '-sampling_interval','10',
                         '-ba_prefix',ba_prefix+'-run','-full_extent','1','-dem',dem_for_ortho,
                         '-mvs',str(args.mvs),'-block',str(args.block_matching)]

            print("Running final stereo reconstruction")
            asp.run_cmd('skysat_stereo_cli.py',stereo_cmd)
           
   
    if 4 in steps2run:
        pc_list = sorted(glob.glob(os.path.join(final_stereo_dir,'12*/run-PC.tif'))) 
        print(f"Identified {len(pc_list)} clouds")
        # this is dem gridding followed by mosaicing
        dem_grid_cmd = ['-mode','gridding_only', '-tr', '2', '-tsrs',epsg_code,'-point_cloud_list'] + pc_list
        asp.run_cmd('skysat_pc_cam.py',dem_grid_cmd)
        print("Mosaicing DEMs")
        dem_mos_cmd = ['-mode','video','-DEM_folder',final_stereo_dir,'-out_folder',mos_dem_dir]
        asp.run_cmd('skysat_dem_mos.py',dem_mos_cmd)


    if 5 in steps2run:

        # this is DEM alignment step
        # add option to mask coreg_dem for static surfaces
        # might want to remove glaciers, forest et al. before coregisteration
        # this can potentially be done in asp_utils step
        # actually use dem_mask.py with options of nlcd, nlcd_filter (not_forest) and of course RGI glacier polygons
        if args.mask_dem == 1:
            # this might change for non-US sites, best to use bareground files
            mask_dem_cmd = ['--nlcd','--glaciers']
            print("Masking reference DEM to static surfaces")
            asp.run_cmd('dem_mask.py',mask_dem_cmd+[os.path.abspath(coreg_dem)])
            coreg_dem = os.path.splitext(coreg_dem)[0]+'_ref.tif'

        # now perform alignment
        median_mos_dem = glob.glob(os.path.join(mos_dem_dir,'video_median_mos.tif'))[0]
        # use the composite filtered by count and NMAD metrics
        median_mos_dem_filt = glob.glob(os.path.join(mos_dem_dir,'video_median_mos_filt*.tif'))[0]
        dem_align_cmd = ['-mode','classic_dem_align','-max_displacement','100','-refdem',coreg_dem,
                         '-source_dem',median_mos_dem_filt,'-outprefix',os.path.join(alignment_dir,'run')]
        print("Aligning DEMs")
        asp.run_cmd('skysat_pc_cam.py',dem_align_cmd)

    if 6 in steps2run:
        # this steps aligns the frame camera models
        camera_list = sorted(glob.glob(os.path.join(init_ba,'run-run-*.tsai')))
        print(f"Detected {len(camera_list)} cameras to be registered to DEM")
        alignment_vector = glob.glob(os.path.join(alignment_dir,'alignment_vector.txt'))[0]
        camera_align_cmd = ['-mode','align_cameras','-transform',alignment_vector,
                            '-outfol',aligned_cam_dir,'-cam_list']+camera_list
        print("Aligning cameras")
        asp.run_cmd('skysat_pc_cam.py',camera_align_cmd)

    if 7 in steps2run:
        # this produces final georegistered orthomosaics
        georegistered_median_dem = glob.glob(os.path.join(alignment_dir,'run-trans_*DEM.tif'))[0]
        ortho_cmd = ['-img_folder',img_folder,'-session','pinhole','-out_folder',final_ortho_dir,
                        '-tsrs',epsg_code,'-DEM',georegistered_median_dem,'-mode','science','-orthomosaic','1','-data','video',
		     '-ba_prefix',os.path.join(aligned_cam_dir,'run-run'),'-frame_index',frame_index]
        print("Running final orthomsaic creation")
        asp.run_cmd('skysat_orthorectify.py',ortho_cmd)

    if 8 in steps2run:
        # this produces a final plot of orthoimage,DEM, NMAD and countmaps
        ortho = glob.glob(os.path.join(final_ortho_dir,'*median_orthomosaic.tif'))[0]
        count = glob.glob(os.path.join(mos_dem_dir,'*count*.tif'))[0]
        nmad = glob.glob(os.path.join(mos_dem_dir,'*nmad*.tif'))[0]
        georegistered_median_dem = glob.glob(os.path.join(alignment_dir,'run-trans_*DEM.tif'))[0]
        print("plotting final figure")
        misc.plot_composite_fig(ortho,georegistered_median_dem,count,nmad,outfn=final_figure,product='video')

if __name__ == '__main__':
    main()
