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
from skysat_stereo import bundle_adjustment_lib as ba
from skysat_stereo import skysat_stereo_workflow as workflow

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
    mask_opt = ['glaciers','glaciers+nlcd']
    parser.add_argument('-mask_dem_opt',default='glaciers',choices=mask_opt,help='surfaces to mask if -mask_dem=1, default is glaciers which uses RGI polygons.\
                        If processing in CONUS, the option of glaciers+nlcd also additionaly masks out forest surfaces')
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
    if coreg_dem != ortho_dem:
        diff_dem = True
        shutil.copy2(ortho_dem,os.path.join(refdem_dir,os.path.basename(ortho_dem)))
    else:
        diff_dem = False
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
    misc.clip_raster_by_shp_disk(coreg_dem,bound_fn)
    misc.ndvtrim_function(os.path.splitext(coreg_dem)[0]+'_shpclip.tif')
    coreg_dem = os.path.splitext(coreg_dem)[0]+'_shpclip_trim.tif'
    if diff_dem:
        misc.clip_raster_by_shp_disk(ortho_dem,bound_buffer_fn)
        misc.ndvtrim_function(os.path.splitext(ortho_dem)[0]+'_shpclip.tif')
        ortho_dem = os.path.splitext(ortho_dem)[0]+'_shpclip_trim.tif'    
    else:
        ortho_dem = coreg_dem


    if 1 in steps2run:
        print("Sampling video sequence and generating Frame Cameras")
        cam_gen_log = workflow.skysat_preprocess(img_folder,mode='video',product_level='l1a',
            outdir=cam_gcp_directory,sampler=60,sampling='num_images',frame_index_fn=frame_index,
            dem=ortho_dem)
    # read the frame_index.csv which contains the info for sampled scenes only
    print(cam_gcp_directory)
    #now point to the subsampled frame_index file
    frame_index = glob.glob(os.path.join(cam_gcp_directory,'*frame*.csv'))[0]

    if 2 in steps2run:
        # this is bundle adjustment step
        print("Running bundle adjustment for the input video sequence")

        ba.bundle_adjust_stable(img=img_folder,ba_prefix=ba_prefix,cam=cam_gcp_directory,
            session='nadirpinhole',num_iter=2000,num_pass=3,gcp=cam_gcp_directory,
            frame_index=frame_index,mode='full_video')
                

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
            workflow.execute_skysat_stereo(img_folder,final_stereo_dir,
                ba_prefix=ba_prefix+'-run',mode='video',threads=2,
                session='nadirpinhole',frame_index=frame_index,sampling_interval=10,
                full_extent=1)
            
        else:
            if args.produce_low_res_for_ortho == 1:
                # will need to produce low res dem using block matching on L1A images and bundle adjusted cameras
                # this was used for the 2 St. Helen's case studies in SkySat stereo manuscript
                print("Running stereo with block matching for producing consistent orthorectification DEM")
                workflow.execute_skysat_stereo(img_folder,init_stereo_dir,
                    ba_prefix=ba_prefix+'-run',mode='video',threads=2,
                    session='nadirpinhole',frame_index=frame_index,sampling_interval=10,
                    full_extent=1,block=1,texture='low')
                
                # query point clouds
                pc_list = sorted(glob.glob(os.path.join(init_stereo_dir,'12*/run-PC.tif')))
                # grid into DEMs
                print("Gridding block matching point clouds")
                workflow.gridding_wrapper(pc_list,tr=4,tsrs=epsg_code)
                dem_list = sorted(glob.glob(os.path.join(init_stereo_dir,'12*/run-DEM.tif')))
                hole_filled_low_res_dem = os.path.join(init_stereo_dir,'block_matching_hole_filled_dem_mos.tif')
                workflow.dem_mosaic_holefill_wrapper(input_dem_list=dem_list,
                    output_dem_path=hole_filled_low_res_dem)
                dem_for_ortho = hole_filled_low_res_dem
            else:
                # this argument will use input orhtodem (used for camera resection) as input for orthorectification
                dem_for_ortho = ortho_dem

            print("Running intermediate orthorectification")
            workflow.execute_skysat_orhtorectification(images=img_folder,session='pinhole',
                outfolder=intermediate_ortho_dir,frame_index_fn=frame_index,tsrs=epsg_code,
                dem=dem_for_ortho,mode='science',data='video',ba_prefix=ba_prefix+'-run')
            ## Now run final stereo

            print("Running final stereo reconstruction")
            workflow.execute_skysat_stereo(intermediate_ortho_dir,final_stereo_dir,
                ba_prefix=ba_prefix+'-run',mode='video',session='pinholemappinhole',
                frame_index=frame_index,sampling_interval=10,full_extent=1,
                dem=dem_for_ortho,mvs=args.mvs,block=args.block_matching)
            
           
   
    if 4 in steps2run:
        pc_list = sorted(glob.glob(os.path.join(final_stereo_dir,'12*/run-PC.tif'))) 
        print(f"Identified {len(pc_list)} clouds")
        # this is dem gridding followed by mosaicing
        workflow.gridding_wrapper(pc_list,tr=2,tsrs=epsg_code)
        print("Mosaicing DEMs")
        workflow.dem_mosaic_wrapper(final_stereo_dir,mode='video',out_folder=mos_dem_dir) 


    if 5 in steps2run:

        # this is DEM alignment step
        # add option to mask coreg_dem for static surfaces
        # might want to remove glaciers, forest et al. before coregisteration
        # this can potentially be done in asp_utils step
        # actually use dem_mask.py with options of nlcd, nlcd_filter (not_forest) and of course RGI glacier polygons
        if args.mask_dem == 1:
            # this might change for non-US sites, best to use bareground files
            if args.mask_dem_opt == 'glaciers':
                mask_list = ['glaciers']
            elif args.msak_dem_opt == 'glaciers+nlcd':
                mask_list = ['nlcd','glaciers'] 
            print("Masking reference DEM to static surfaces") 
            misc.dem_mask_disk(mask_list,os.path.abspath(coreg_dem))
            coreg_dem = os.path.splitext(coreg_dem)[0]+'_ref.tif'

        # now perform alignment
        median_mos_dem = glob.glob(os.path.join(mos_dem_dir,'video_median_mos.tif'))[0]
        # use the composite filtered by count and NMAD metrics
        median_mos_dem_filt = glob.glob(os.path.join(mos_dem_dir,'video_median_mos_filt*.tif'))[0]
        print("Aligning DEMs")
        workflow.alignment_wrapper_single(coreg_dem,source_dem=median_mos_dem_filt,
            max_displacement=100,outprefix=os.path.join(alignment_dir,'run'))
           

    if 6 in steps2run:
        # this steps aligns the frame camera models
        camera_list = sorted(glob.glob(os.path.join(init_ba,'run-run-*.tsai')))
        print(f"Detected {len(camera_list)} cameras to be registered to DEM")
        alignment_vector = glob.glob(os.path.join(alignment_dir,'alignment_vector.txt'))[0]
        if not os.path.exists(aligned_cam_dir):
            os.makedirs(aligned_cam_dir)
        print("Aligning cameras")
        workflow.align_cameras_wrapper(input_camera_list=camera_list,transform_txt=alignment_vector,
            outfolder=aligned_cam_dir)
        
    
    if 7 in steps2run:
        # this produces final georegistered orthomosaics
        georegistered_median_dem = glob.glob(os.path.join(alignment_dir,'run-trans_*DEM.tif'))[0]
        print("Running final orthomsaic creation")
        workflow.execute_skysat_orhtorectification(images=img_folder,session='pinhole',
            out_folder=final_ortho_dir,tsrs=epsg_code,dem=georegistered_median_dem,
            mode='science',orthomosaic=1,data='video',ba_prefix=os.path.join(aligned_cam_dir,'run-run'))

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
