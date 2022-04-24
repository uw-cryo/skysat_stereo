#! /usr/bin/env python

import os,sys,glob
import numpy as np
import geopandas as gpd
import pandas as pd
from pygeotools.lib import iolib,malib
from p_tqdm import p_umap, p_map
from skysat_stereo import skysat
from skysat_stereo import asp_utils as asp
from skysat_stereo import misc_geospatial as geo
from shapely.geometry import Polygon
from itertools import combinations,compress

def prepare_stereopair_list(img_folder,perc_overlap,out_fn,aoi_bbox=None,cross_track=False):
    """
    """ 
    geo_crs = 'EPSG:4326'
    # populate img list
    try:
        img_list = sorted(glob.glob(os.path.join(img_folder,'*.tif')))
        print("Number of images {}".format(len(img_list)))
    except:
        print ("No images found in the directory. Make sure they end with a .tif extension")
        sys.exit()
    out_shp = os.path.splitext(out_fn)[0]+'_bound.gpkg'
    n_proc = iolib.cpu_count()
    shp_list = p_umap(skysat.skysat_footprint,img_list,num_cpus=2*n_proc)
    merged_shape = geo.shp_merger(shp_list)
    bbox = merged_shape.total_bounds
    merged_shape = geo.shp_merger(shp_list)
    bbox = merged_shape.total_bounds
    print (f'Bounding box lon_lat is:{bbox}')
    print (f'Bounding box lon_lat is:{bbox}')
    bound_poly = Polygon([[bbox[0],bbox[3]],[bbox[2],bbox[3]],[bbox[2],bbox[1]],[bbox[0],bbox[1]]])
    bound_shp = gpd.GeoDataFrame(index=[0],geometry=[bound_poly],crs=geo_crs)
    bound_centroid = bound_shp.centroid
    cx = bound_centroid.x.values[0]
    cy = bound_centroid.y.values[0]
    pad = np.ptp([bbox[3],bbox[1]])/6.0
    lat_1 = bbox[1]+pad
    lat_2 = bbox[3]-pad
    #local_ortho = '+proj=ortho +lat_0={} +lon_0={}'.format(cy,cx)
    local_aea = "+proj=aea +lat_1={} +lat_2={} +lat_0={} +lon_0={} +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs".format(lat_1,lat_2,cy,cx)
    print ('Local Equal Area coordinate system is : {} \n'.format(local_aea))
    print('Saving bound shapefile at {} \n'.format(out_shp))
    bound_shp.to_file(out_shp,driver='GPKG')
    
    # condition to check bbox_aoi
    if aoi_bbox is not None:
        bbox = gpd.read_file(aoi_bbox)
        mask = merged_shape.to_crs(bbox.crs).intersects(bbox)
        img_list = merged_shape[mask].img.values

    img_combinations = list(combinations(img_list,2))
    n_comb = len(img_combinations)
    perc_overlap = np.ones(n_comb,dtype=float)*perc_overlap
    proj = local_aea
    tv = p_map(skysat.frame_intsec, img_combinations, [proj]*n_comb, perc_overlap,num_cpus=4*n_proc)
    # result to this contains truth value (0 or 1, overlap percentage)
    truth_value = [tvs[0] for tvs in tv]
    overlap = [tvs[1] for tvs in tv]
    valid_list = list(compress(img_combinations,truth_value))
    overlap_perc_list = list(compress(overlap,truth_value))
    print('Number of valid combinations are {}, out of total {}  input images making total combinations {}\n'.format(len(valid_list),len(img_list),n_comb))
    with open(out_fn, 'w') as f:
        img1_list = [x[0] for x in valid_list]
        img2_list = [x[1] for x in valid_list]
        for idx,i in enumerate(valid_list):
            #f.write("%s %s\n" % i) 
            f.write(f"{os.path.abspath(img1_list[idx])} {os.path.abspath(img2_list[idx])}\n")
    out_fn_overlap = os.path.splitext(out_fn)[0]+'_with_overlap_perc.pkl'
    img1_list = [x[0] for x in valid_list]
    img2_list = [x[1] for x in valid_list]
    out_df = pd.DataFrame({'img1':img1_list,'img2':img2_list,'overlap_perc':overlap_perc_list})
    out_df.to_pickle(out_fn_overlap)
    
    out_fn_stereo = os.path.splitext(out_fn_overlap)[0]+'_stereo_only.pkl'
    stereo_only_df = skysat.prep_trip_df(out_fn_overlap,cross_track=cross_track)
    stereo_only_df.to_pickle(out_fn_stereo)
    out_fn_stereo_ba = os.path.splitext(out_fn_overlap)[0]+'_stereo_only.txt'
    stereo_only_df[['img1','img2']].to_csv(out_fn_stereo_ba,sep=' ',header=False,index=False)
    
    return stereo_only_df, out_df

def skysat_preprocess(img_folder,mode,sampling=None,frame_index=None,product_level='l1a',
        sampler=5,overlap_pkl=None,dem=None,outdir=None):
    """
    """
    if not os.path.exists(outdir):
        try:
            os.makedir(outdir)
        except:
            os.makedirs(outdir)
    if mode == 'video':
        sampling = args.video_sampling_mode
        frame_index = skysat.parse_frame_index(frame_index,True)
        product_level = 'l1a'
        num_samples = len(frame_index)
        frames = frame_index.name.values
        outdf = os.path.join(outdir,os.path.basename(frame_index))
        if sampling == 'sampling_interval':
            print("Hardcoded sampling interval results in frame exclusion at the end of the video sequence based on step size, better to chose the num_images mode and the program will equally distribute accordingly")
            idx = np.arange(0,num_samples,sampler)
            outdf = '{}_sampling_inteval_{}.csv'.format(os.path.splitext(outdf)[0],sampler)
        else:
            print("Sampling {} from {} of the input video sequence".format(sampler,num_samples))
            idx = np.linspace(0,num_samples-1,sampler,dtype=int)
            outdf = '{}_sampling_inteval_aprox{}.csv'.format(os.path.splitext(outdf)[0],idx[1]-idx[0])
        
        sub_sampled_frames = frames[idx]
        sub_df = frame_index[frame_index['name'].isin(list(sub_sampled_frames))]
        sub_df.to_csv(outdf,sep=',',index=False)

        #this is camera/gcp initialisation
        n = len(sub_sampled_frames)
        img_list = [glob.glob(os.path.join(img_folder,'{}*.tiff'.format(frame)))[0] for frame in sub_sampled_frames]
        pitch = [1]*n
        out_fn = [os.path.join(outdir,'{}_frame_idx.tsai'.format(frame)) for frame in sub_sampled_frames]
        out_gcp = [os.path.join(outdir,'{}_frame_idx.gcp'.format(frame)) for frame in sub_sampled_frames]
        frame_index = [frame_index]*n
       	camera = [None]*n
        gcp_factor = 4

    elif mode == 'triplet':
        df = pd.read_pickle(overlap_pkl)
        img_list = list(np.unique(np.array(list(df.img1.values)+list(df.img2.values))))
        img_list = [os.path.splitext(os.path.basename(img))[0] for img in img_list]
        cam_list = [glob.glob(os.path.join(img_folder,'{}*.tif'.format(img)))[0] for img in img_list]
        n = len(img_list)
        if product_level == 'l1b':
            pitch = [0.8]*n
        else:
            pitch = [1.0]*n
        out_fn = [os.path.join(outdir,'{}_rpc.tsai'.format(frame)) for frame in img_list]
        out_gcp = [os.path.join(outdir,'{}_rpc.gcp'.format(frame)) for frame in img_list]
        camera = cam_list
        frame_index = [None]*n
        img_list = cam_list
        gcp_factor = 8

    fl = [553846.153846]*n
    cx = [1280]*n
    cy = [540]*n
    ht_datum = [malib.get_stats_dict(iolib.fn_getma(dem))['median']]*n # use this value for height where DEM has no-data
    gcp_std = [1]*n
    datum = ['WGS84']*n
    refdem = [dem]*n
    n_proc = 30
    #n_proc = cpu_count()
    print("Starting camera resection procedure")
    cam_gen_log = p_map(asp.cam_gen,img_list,fl,cx,cy,pitch,ht_datum,gcp_std,out_fn,out_gcp,datum,refdem,camera,frame_index,num_cpus = n_proc)
    print("writing gcp with basename removed")
    # count expexted gcp 
    print(f"Total expected GCP {gcp_factor*n}")    
    asp.clean_gcp(out_gcp,outdir)
    return cam_gen_log   

def execute_skysat_stereo(img,mode,session,overlap_list_fn,frame_index,dem,texture,
    sampling_interval,cam_folder,outfol,ba_prefix=None,writeout_only=False,mvs=0,block=1,full_extent=1,
    entry_point=0,threads=2,overlap_pkl=None,job_fn=None):
    """
    """
    img = os.path.abspath(img)
    try:
        img_list = sorted(glob.glob(os.path.join(img, '*.tif')))
        temp = img_list[1]
    except BaseException:
        img_list = sorted(glob.glob(os.path.join(img, '*.tiff')))
    if len(img_list) == 0:
        print("No images in the specified folder, exiting")
        sys.exit()

    if mode == 'video':
        # assume for now that we are still operating on a fixed image interval method
        # can accomodate different convergence angle function method here.
        frame_gdf = skysat.parse_frame_index(frame_index)
        # for now hardcording sgm,mgm,kernel params, should accept as inputs.
        # Maybe discuss with David with these issues/decisions when the overall
        # system is in place
        if mvs == 1:
            job_list = skysat.video_mvs(img,t=session,cam_fol=cam_fol,ba_prefix=ba_prefix,dem=dem,
                           sampling_interval=sampling_interval,texture=texture,
                           outfol=outfol,block=block,frame_index=frame_gdf)

        else:
            if full_extent == 1:
                full_extent = True
            else:
                full_extent = False
            job_list = skysat.prep_video_stereo_jobs(img,t=session,cam_fol=cam_fol,ba_prefix=ba_prefix,
                dem=dem,sampling_interval=sampling_interval,texture=texture,outfol=outfol,block=block,
                frame_index=frame_gdf,full_extent=full_extent,entry_point=entry_point)
    elif mode == 'triplet':
        if crop_map == 1:
            crop_map = True
        else: 
            crop_map = False
            
        job_list = skysat.triplet_stereo_job_list(cross_track=cross_track,t=session,
            threads = threads,overlap_list=overlap_pkl, img_list=img_list, ba_prefix=ba_prefix, 
            cam_fol=cam_fol, dem=dem, crop_map=crop_map,texture=texture, outfol=outfol, block=block,
            entry_point=entry_point)
    if not writeout_only:
        # decide on number of processes
        # if block matching, Plieades is able to handle 30-40 4 threaded jobs on bro node
        # if MGM/SGM, 25 . This stepup is arbitrariry, research on it more.
        # next build should accept no of jobs and stereo threads as inputs
    
        print(job_list[0])
        n_cpu = iolib.cpu_count()
        # no of parallel jobs with user specified threads per job
        jobs = int(n_cpu/args.threads)
        stereo_log = p_map(asp.run_cmd,['stereo']*len(job_list), job_list, num_cpus=jobs)
        stereo_log_fn = os.path.join(outfol,'stereo_log.log')
        print("Consolidated stereo log saved at {}".format(stereo_log_fn))
    else:
        print(f"Writng jobs at {job_fn}")
        with open(args.job_fn,'w') as f:
            for idx,job in enumerate(tqdm(job_list)):
                try:                
                    job_str = 'stereo ' + ' '.join(job) + '\n'
                    f.write(job_str)
                except:
                    continue
