#! /usr/bin/env python
import os,sys,glob,shutil
import subprocess
import argparse
from distutils.spawn import find_executable
from pygeotools.lib import iolib,malib
import geopandas as gpd
import numpy as np
from datetime import datetime
import pandas as pd
from multiprocessing import cpu_count

def run_cmd(bin, args, **kw):
    # Note, need to add full executable
    # from dshean/vmap.py
    #binpath = os.path.join('/home/sbhushan/src/StereoPipeline/bin',bin)
    #binpath = find_executable(bin)
    binpath = '/nobackupp16/swbuild3/sbhusha1/StereoPipeline-3.1.1-alpha-2022-10-31-x86_64-Linux/bin/bundle_adjust'
    if binpath is None:
        msg = ("Unable to find executable %s\n"
        "Install ASP and ensure it is in your PATH env variable\n"
       "https://ti.arc.nasa.gov/tech/asr/intelligent-robotics/ngt/stereo/")
        sys.exit(msg)
    # binpath = os.path.join('/opt/StereoPipeline/bin/',bin)
    call = [binpath, ]
    print(call)
    call.extend(args)
    print(call)
	# print(type(call))
    # print(' '.join(call))
    try:
        code = subprocess.call(call, shell=False)
    except OSError as e:
        raise Exception('%s: %s' % (binpath, e))
    if code != 0:
        raise Exception('ASP step ' + kw['msg'] + ' failed')
        

def get_ba_opts(ba_prefix, ip_per_tile=4000,camera_weight=None,translation_weight=0.4,rotation_weight=0,fixed_cam_idx=None,overlap_list=None, robust_threshold=None, overlap_limit=None, initial_transform=None, input_adjustments=None, flavor='general_ba', session='nadirpinhole', gcp_transform=False,num_iterations=2000,num_pass=2,lon_lat_limit=None,elevation_limit=None):
    ba_opt = []
    # allow CERES to use multi-threads
    ba_opt.extend(['--threads', str(cpu_count())])
    #ba_opt.extend(['--threads', '1'])
    ba_opt.extend(['-o', ba_prefix])

    # keypoint-finding args
    # relax triangulation error based filters to account for initial camera errors
    ba_opt.extend(['--min-matches', '4'])
    ba_opt.extend(['--disable-tri-ip-filter'])
    ba_opt.extend(['--force-reuse-match-files'])
    ba_opt.extend(['--ip-per-tile', str(ip_per_tile)])
    ba_opt.extend(['--ip-inlier-factor', '0.2'])
    ba_opt.extend(['--ip-num-ransac-iterations', '1000'])
    ba_opt.extend(['--skip-rough-homography'])
    ba_opt.extend(['--min-triangulation-angle', '0.0001'])

    # Save control network created from match points
    ba_opt.extend(['--save-cnet-as-csv'])

    # Individually normalize images to properly stretch constrant 
    # Helpful in keypoint detection
    ba_opt.extend(['--individually-normalize'])

    if robust_threshold is not None:
        # make the solver focus more on mininizing very high reporjection errors
        ba_opt.extend(['--robust-threshold', str(robust_threshold)])

    if camera_weight is not None:
        # this generally assigns weight to penalise movement of camera parameters (Default:0)
        ba_opt.extend(['--camera-weight', str(camera_weight)])
    else:
        # this is more fine grained, will pinalize translation but allow rotation parameters update
        ba_opt.extend(['--translation-weight',str(translation_weight)])
        ba_opt.extend(['--rotation-weight',str(rotation_weight)])

    if fixed_cam_idx is not None:
        # parameters for cameras at the specified indices will not be floated during optimisation
        ba_opt.extend(['--fixed-camera-indices',' '.join(fixed_cam_idx.astype(str))])
    ba_opt.extend(['-t', session])
    
    # filter points based on reprojection errors before running a new pass
    ba_opt.extend(['--remove-outliers-params', '75 3 5 6'])
    
    # How about adding num random passes here ? Think about it, it might help if we are getting stuck in local minima :)
    if session == 'nadirpinhole':
        ba_opt.extend(['--inline-adjustments'])
        # write out a new camera model file with updated parameters

    # specify number of passes and maximum iterations per pass
    ba_opt.extend(['--num-iterations', str(num_iterations)])
    ba_opt.extend(['--num-passes', str(num_pass)])
    #ba_opt.extend(['--parameter-tolerance','1e-14'])

    if gcp_transform:
        ba_opt.extend(['--transform-cameras-using-gcp'])
       
    if initial_transform:
        ba_opt.extend(['--initial-transform', initial_transform])
    if input_adjustments:
        ba_opt.extend(['--input-adjustments', input_adjustments])

    # these 2 parameters determine which image pairs to use for feature matching
    # only the selected pairs are used in formation of the bundle adjustment control network
    # video is a sequence of overlapping scenes, so we use an overlap limit
    # triplet stereo uses list of overlapping pairs
    if overlap_limit:
        ba_opt.extend(['--overlap-limit',str(overlap_limit)])
    if overlap_list:
        ba_opt.extend(['--overlap-list', overlap_list])
   
    # these two params are not used generally.
    if lon_lat_limit:
        ba_opt.extend(['--lon-lat-limit',str(lon_lat_limit[0]),str(lon_lat_limit[1]),str(lon_lat_limit[2]),str(lon_lat_limit[3])])
    if elevation_limit:
        ba_opt.extend(['--elevation-limit',str(elevation_limit[0]),str(elevation_limit[1])])

    return ba_opt

def bundle_adjust_stable(img,ba_prefix,cam=None,session='rpc',initial_transform=None,
                        input_adjustments=None,overlap_list=None,gcp=None,
                        mode='full_triplet',bound=None,camera_param2float='trans+rot',
                        dem=None,num_iter=2000,num_pass=2,frame_index=None):
    """
    """
    img_list = sorted(glob.glob(os.path.join(img,'*.tif')))
    if len(img_list) < 2:
        img_list = sorted(glob.glob(os.path.join(img, '*.tiff')))
        #img_list = [os.path.basename(x) for x in img_list]
        if os.path.islink(img_list[0]):
            img_list = [os.readlink(x) for x in img_list] 
    if overlap_list is not None:
        # need to remove images and cameras which are not optimised during bundle adjustment
        # read pairs from input overlap list
        initial_count = len(img_list)
        with open(overlap_list) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        l_img = [x.split(' ')[0] for x in content]
        r_img = [x.split(' ')[1] for x in content]
        total_img = l_img + r_img
        uniq_idx = np.unique(total_img, return_index=True)[1]
        img_list = [total_img[idx] for idx in sorted(uniq_idx)]
        print(f"Out of the initial {initial_count} images, {len(img_list)} will be orthorectified using adjusted cameras")

        if cam is not None:
            #cam = os.path.abspath(cam)
            if 'run' in os.path.basename(cam):
                cam_list = [glob.glob(cam+'-'+os.path.splitext(os.path.basename(x))[0]+'*.tsai')[0] for x in img_list]
                print("No of cameras is {}".format(len(cam_list)))
                
            else:
                cam_list = [glob.glob(os.path.join(cam,os.path.splitext(os.path.basename(x))[0]+'*.tsai'))[0] for x in img_list]
        if gcp is not None:
            gcp_list = sorted(glob.glob(os.path.join(args.gcp, '*.gcp')))
    if bound:
        bound = gpd.read_file(args.bound)
        geo_crs = {'init':'epsg:4326'}
        if bound.crs is not geo_crs:
            bound = bound.to_crs(geo_crs)
        lon_min,lat_min,lon_max,lat_max = bound.total_bounds    
    if camera_param2float == 'trans+rot':
        cam_wt = 0
    else:
        # this will invoke adjustment with rotation weight of 0 and translation weight of 0.4
        cam_wt = None
    print(f"Camera weight is {cam_wt}")
    
    if dem:
        dem = iolib.fn_getma(dem)
        dem_stats = malib.get_stats_dict(dem)
        min_elev,max_elev = [dem_stats['min']-500,dem_stats['max']+500] 
        dem = None
    if mode == 'full_triplet':
        if overlap_list is None:
             print(
                "Attempted bundle adjust will be expensive, will try to find matches in each and every pair")
        # the concept is simple
        #first 3 cameras, and then corresponding first three cameras from next collection are fixed in the first go
        # these serve as a kind of #GCP, preventing a large drift in the triangulated points/camera extrinsics during optimization
        img_time_identifier_list = np.array([os.path.basename(img).split('_')[1] for img in img_list])
        img_time_unique_list = np.unique(img_time_identifier_list)
        second_collection_list = np.where(img_time_identifier_list == img_time_unique_list[1])[0][[0,1,2]]
        fix_cam_idx = np.array([0,1,2]+list(second_collection_list))
        print(type(fix_cam_idx)) 
        round1_opts = get_ba_opts(
            ba_prefix, session=session,num_iterations=num_iter,num_pass=num_pass,
            fixed_cam_idx=fix_cam_idx,overlap_list=overlap_list,camera_weight=cam_wt)
        # enter round2_opts here only ?
        if session == 'nadirpinhole':
            ba_args = img_list+ cam_list
        else:
            ba_args = img_list
        print("Running round 1 bundle adjustment for given triplet stereo combination")
        run_cmd('bundle_adjust', round1_opts+ba_args)
        
        # Save the first and foremost bundle adjustment reprojection error file
        init_residual_fn_def = sorted(glob.glob(ba_prefix+'*initial*residuals*pointmap*.csv'))[0]
        init_residual_fn = os.path.splitext(init_residual_fn_def)[0]+'_initial_reproj_error.csv' 
        init_per_cam_reproj_err = sorted(glob.glob(ba_prefix+'-*initial_residuals_raw_pixels.txt'))[0]
        init_per_cam_reproj_err_disk = os.path.splitext(init_per_cam_reproj_err)[0]+'_initial_per_cam_reproj_error.txt'
        init_cam_stats = sorted(glob.glob(ba_prefix+'-*initial_residuals_stats.txt'))[0]
        init_cam_stats_disk = os.path.splitext(init_cam_stats)[0]+'_initial_camera_stats.txt'
        shutil.copy2(init_residual_fn_def,init_residual_fn)
        shutil.copy2(init_per_cam_reproj_err,init_per_cam_reproj_err_disk)
        shutil.copy2(init_cam_stats,init_cam_stats_disk)
        
        if session == 'nadirpinhole':
            identifier = os.path.basename(cam_list[0]).split('_',14)[0][:2]
            print(ba_prefix+'-{}*.tsai'.format(identifier))
            cam_list = sorted(glob.glob(os.path.join(ba_prefix+ '-{}*.tsai'.format(identifier))))
            ba_args = img_list+cam_list
            fixed_cam_idx2 = np.delete(np.arange(len(img_list),dtype=int),fix_cam_idx)
            round2_opts = get_ba_opts(ba_prefix, overlap_list=overlap_list,session=session,
                                      fixed_cam_idx=fixed_cam_idx2,camera_weight=cam_wt)
        else:
            # round 1 is adjust file
            # Only camera model parameters for the first three stereo pairs float in this round
            input_adjustments = ba_prefix
            round2_opts = get_ba_opts(
                ba_prefix, overlap_limit, input_adjustments=ba_prefix, flavor='2round_gcp_2', session=session,
                elevation_limit=[min_elev,max_elev],lon_lat_limit=[lon_min,lat_min,lon_max,lat_max])
            ba_args = img_list+gcp_list
        
        
        print("running round 2 bundle adjustment for given triplet stereo combination")
        run_cmd('bundle_adjust', round2_opts+ba_args)
        
        # Save state for final condition reprojection errors for the sparse triangulated points
        final_residual_fn_def = sorted(glob.glob(ba_prefix+'*final*residuals*pointmap*.csv'))[0]
        final_residual_fn = os.path.splitext(final_residual_fn_def)[0]+'_final_reproj_error.csv'
        shutil.copy2(final_residual_fn_def,final_residual_fn)
        final_per_cam_reproj_err = sorted(glob.glob(ba_prefix+'-*final_residuals_raw_pixels.txt'))[0]
        final_per_cam_reproj_err_disk = os.path.splitext(final_per_cam_reproj_err)[0]+'_final_per_cam_reproj_error.txt'
        final_cam_stats = sorted(glob.glob(ba_prefix+'-*final_residuals_stats.txt'))[0]
        final_cam_stats_disk = os.path.splitext(final_cam_stats)[0]+'_final_camera_stats.txt'
        shutil.copy2(final_per_cam_reproj_err,final_per_cam_reproj_err_disk)
        shutil.copy2(final_cam_stats,final_cam_stats_disk)

    elif mode == 'full_video':
        df = pd.read_csv(frame_index)
        # block to determine automatically overlap limit of 40 seconds for computing match points
        df['dt'] = [datetime.strptime(date.split('+00:00')[0],'%Y-%m-%dT%H:%M:%S.%f') for date in df.datetime.values]
        delta = (df.dt.values[1]-df.dt.values[0])/np.timedelta64(1, 's')
        # i hardocde overlap limit to have 40 seconds coverage
        overlap_limit = np.int(np.ceil(40/delta))
        print("Calculated overlap limit as {}".format(overlap_limit))
        img_list = [glob.glob(os.path.join(img,'*{}*.tiff'.format(x)))[0] for x in df.name.values]
        cam_list = [glob.glob(os.path.join(cam,'*{}*.tsai'.format(x)))[0] for x in df.name.values]
        gcp_list = [glob.glob(os.path.join(gcp,'*{}*.gcp'.format(x)))[0] for x in df.name.values]
        #also append the clean gcp here
        print(os.path.join(gcp,'*clean*_gcp.gcp'))
        gcp_list.append(glob.glob(os.path.join(gcp,'*clean*_gcp.gcp'))[0])
        round1_opts = get_ba_opts(
            ba_prefix, overlap_limit=overlap_limit, flavor='2round_gcp_1', session=session,ip_per_tile=4000,
            num_iterations=num_iter,num_pass=num_pass,camera_weight=cam_wt,fixed_cam_idx=None,robust_threshold=None)
        print("Running round 1 bundle adjustment for input video sequence")
        if session == 'nadirpinhole':
            ba_args = img_list+cam_list
        else:
            ba_args = img_list
        # Check if this command executed till last
        print('Running bundle adjustment round1')
        run_cmd('bundle_adjust', round1_opts+ba_args)

        # Make files used to evaluate solution quality
        init_residual_fn_def = sorted(glob.glob(ba_prefix+'*initial*residuals*pointmap*.csv'))[0]
        init_per_cam_reproj_err = sorted(glob.glob(ba_prefix+'-*initial_residuals_*raw_pixels.txt'))[0]
        init_per_cam_reproj_err_disk = os.path.splitext(init_per_cam_reproj_err)[0]+'_initial_per_cam_reproj_error.txt'
        init_residual_fn = os.path.splitext(init_residual_fn_def)[0]+'_initial_reproj_error.csv' 
        shutil.copy2(init_residual_fn_def,init_residual_fn)
        shutil.copy2(init_per_cam_reproj_err,init_per_cam_reproj_err_disk)

        # Copy final reprojection error files before transforming cameras
        final_residual_fn_def = sorted(glob.glob(ba_prefix+'*final*residuals*pointmap*.csv'))[0]
        final_residual_fn = os.path.splitext(final_residual_fn_def)[0]+'_final_reproj_error.csv'
        final_per_cam_reproj_err = sorted(glob.glob(ba_prefix+'-*final_residuals*_raw_pixels.txt'))[0]
        final_per_cam_reproj_err_disk = os.path.splitext(final_per_cam_reproj_err)[0]+'_final_per_cam_reproj_error.txt'
        shutil.copy2(final_residual_fn_def,final_residual_fn)
        shutil.copy2(final_per_cam_reproj_err,final_per_cam_reproj_err_disk)

        if session == 'nadirpinhole':
            # prepare for second run to apply a constant transform to the self-consistent models using initial ground footprints
            identifier = os.path.basename(cam_list[0]).split(df.name.values[0])[0]
            print(ba_prefix+identifier+'-{}*.tsai'.format(df.name.values[0]))
            cam_list = [glob.glob(ba_prefix+identifier+'-{}*.tsai'.format(img))[0] for img in df.name.values]
            print(len(cam_list))
            ba_args = img_list+cam_list+gcp_list

        round2_opts = get_ba_opts(
                ba_prefix, overlap_limit = overlap_limit, flavor='2round_gcp_2', session=session, gcp_transform=True,camera_weight=0,
                num_iterations=0,num_pass=1)
        
        print("running round 2 bundle adjustment for input video sequence")
        run_cmd('bundle_adjust', round2_opts+ba_args)
        