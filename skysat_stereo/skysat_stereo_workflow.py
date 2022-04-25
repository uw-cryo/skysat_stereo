#! /usr/bin/env python

import os,sys,glob,re,shutil
import numpy as np
import geopandas as gpd
import pandas as pd
from pygeotools.lib import iolib,malib
from tqdm import tqdm
from p_tqdm import p_umap, p_map
from skysat_stereo import skysat
from skysat_stereo import asp_utils as asp
from rpcm import geo
from skysat_stereo import misc_geospatial as misc
from shapely.geometry import Polygon
import itertools
from osgeo import osr
from pyproj import Transformer


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
    shp_list = p_map(skysat.skysat_footprint,img_list,num_cpus=2*n_proc)
    merged_shape = misc.shp_merger(shp_list)
    bbox = merged_shape.total_bounds
    merged_shape = misc.shp_merger(shp_list)
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

    img_combinations = list(itertools.combinations(img_list,2))
    n_comb = len(img_combinations)
    perc_overlap = np.ones(n_comb,dtype=float)*perc_overlap
    proj = local_aea
    tv = p_map(skysat.frame_intsec, img_combinations, [proj]*n_comb, perc_overlap,num_cpus=4*n_proc)
    # result to this contains truth value (0 or 1, overlap percentage)
    truth_value = [tvs[0] for tvs in tv]
    overlap = [tvs[1] for tvs in tv]
    valid_list = list(itertools.compress(img_combinations,truth_value))
    overlap_perc_list = list(itertools.compress(overlap,truth_value))
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

def execute_skysat_orhtorectification(images,outdir,dem='WGS84',tr=None,tsrs=None,del_opt=False,cam_folder=None,ba_prefix=None,
    mode='science',session=None,overlap_list=None,frame_index_fn=None,copy_rpc=1,orthomosaic=0):
    """
    """
    if mode == 'browse':
        """
        this block creates low-res orthomosaics from RPC info for browsing purpose only
        """
        for_img_list,nadir_img_list,aft_img_list,for_time,nadir_time,aft_time = skysat.sort_img_list(images)
        for_out_dir = os.path.join(outdir,'for_map_browse')
        nadir_out_dir = os.path.join(outdir,'nadir_map_browse')
        aft_out_dir = os.path.join(outdir,'aft_map_browse')
        for_out_list = [os.path.join(for_out_dir,os.path.splitext(os.path.basename(img))[0]+'_browse_map.tif') for img in for_img_list]
        nadir_out_list = [os.path.join(nadir_out_dir,os.path.splitext(os.path.basename(img))[0]+'_browse_map.tif') for img in nadir_img_list]
        aft_out_list = [os.path.join(aft_out_dir,os.path.splitext(os.path.basename(img))[0]+'_browse_map.tif') for img in aft_img_list]
        for_count,nadir_count,aft_count = [len(for_img_list), len(nadir_img_list), len(aft_img_list)]
        print("Performing orthorectification for forward images {}".format(for_time))
        for_map_log = p_map(asp.mapproject,for_img_list,for_out_list,[session]*for_count,['WGS84']*for_count,[None]*for_count,
            ['EPSG:4326']*for_count,[None]*for_count,[None]*for_count,[None]*for_count)
        print("Performing orthorectification for nadir images {}".format(nadir_time))
        nadir_map_log = p_map(asp.mapproject,nadir_img_list,nadir_out_list,[session]*nadir_count,['WGS84']*nadir_count,[None]*nadir_count,
            ['EPSG:4326']*nadir_count,[None]*nadir_count,[None]*nadir_count,[None]*nadir_count)
        print("Performing orthorectification for aft images {}".format(aft_time))
        aft_map_log = p_map(asp.mapproject,aft_img_list,aft_out_list,[session]*aft_count,['WGS84']*aft_count,[None]*aft_count,
            ['EPSG:4326']*aft_count,[None]*aft_count,[None]*aft_count,[None]*aft_count)
        ortho_log = os.path.join(outdir,'low_res_ortho.log')
        print("Orthorectification log saved at {}".format(ortho_log))
        with open(ortho_log,'w') as f:
            total_ortho_log = for_map_log+nadir_map_log+aft_map_log
            for log in itertools.chain.from_iterable(total_ortho_log):
                f.write(log)

        # after orthorectification, now do mosaic
        for_out_mos = os.path.join(outdir,'for_map_mos_{}m.tif'.format(tr))
        for_map_list = sorted(glob.glob(os.path.join(for_out_dir,'*.tif')))
        nadir_out_mos = os.path.join(outdir,'nadir_map_mos_{}m.tif'.format(tr))
        nadir_map_list = sorted(glob.glob(os.path.join(nadir_out_dir,'*.tif')))
        aft_out_mos = os.path.join(outdir,'aft_map_mos_{}m.tif'.format(tr))
        aft_map_list = sorted(glob.glob(os.path.join(aft_out_dir,'*.tif')))
        print("Preparing forward browse orthomosaic")
        for_mos_log = asp.dem_mosaic(for_map_list,for_out_mos,tr,tsrs,stats='first',tile_size=None)
        print("Preparing nadir browse orthomosaic")
        nadir_mos_log = asp.dem_mosaic(nadir_map_list, nadir_out_mos, tr, tsrs,stats='first',tile_size=None)
        print("Preparing aft browse orthomosaic")
        aft_mos_log = asp.dem_mosaic(aft_map_list, aft_out_mos, tr, tsrs,stats='first',tile_size=None)
        ## delete temporary files
        if del_opt:
            [shutil.rmtree(x) for x in [for_out_dir,nadir_out_dir,aft_out_dir]]
        #Save figure to jpeg ?
        fig_title = os.path.basename(images[0]).split('_',15)[0]+'_'+for_time+'_'+nadir_time+'_'+aft_time
        fig,ax = plt.subplots(1,3,figsize=(10,10))
        pltlib.iv_fn(for_out_mos,full=True,ax=ax[0],cmap='gray',scalebar=True,title='Forward')
        pltlib.iv_fn(nadir_out_mos,full=True,ax=ax[1],cmap='gray',scalebar=True,title='NADIR')
        pltlib.iv_fn(aft_out_mos,full=True,ax=ax[2],cmap='gray',scalebar=True,title='Aft')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(fig_title)
        browse_img_fn = os.path.join(outdir,'browse_img_{}_{}m.jpg'.format(fig_title,tr))
        fig.savefig(browse_img_fn,dpi=300,bbox_inches='tight',pad_inches=0.1)
        print("Browse figure saved at {}".format(browse_img_fn))
        
    if mode == 'science':
        img_list = images
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


        if frame_index_fn is not None:
            frame_index = skysat.parse_frame_index(frame_index_fn)
            img_list = [glob.glob(os.path.join(dir,'{}*.tiff'.format(frame)))[0] for frame in frame_index.name.values]
            print("no of images is {}".format(len(img_list)))
        img_prefix = [os.path.splitext(os.path.basename(img))[0] for img in img_list]
        out_list = [os.path.join(outdir,img+'_map.tif') for img in img_prefix]
        session_list = [session]*len(img_list)
        dem_list = [dem]*len(img_list)
        tr_list = [tr]*len(img_list)
        if frame_index_fn is not None:
            # this hack is for video
            df = skysat.parse_frame_index(frame_index_fn)
            trunc_df = df[df['name'].isin(img_prefix)]
            tr_list = [str(gsd) for gsd in trunc_df.gsd.values]
        srs_list = [tsrs]*len(img_list)

        if session == 'pinhole':
            if ba_prefix:
                cam_list = [glob.glob(os.path.abspath(ba_prefix)+'-'+os.path.splitext(os.path.basename(x))[0]+'*.tsai')[0] for x in img_list]
                print("No of cameras is {}".format(len(cam_list)))
            else:
                print(os.path.join(os.path.abspath(cam_folder),os.path.splitext(os.path.basename(img_list[0]))[0]+'*.tsai'))
                cam_list = [glob.glob(os.path.join(os.path.abspath(cam_folder),os.path.splitext(os.path.basename(x))[0]+'*.tsai'))[0] for x in img_list]
        else:
            cam_list = [None]*len(img_list)
            if ba_prefix:
                # not yet implemented
                ba_prefix_list = [ba_prefix]*len(img_list)

        print("Mapping given images")
        ortho_logs = p_map(asp.mapproject,img_list,out_list,session_list,dem_list,tr_list,srs_list,cam_list,
            [None]*len(img_list),[None]*len(img_list),num_cpus=int(iolib.cpu_count()/4))
        ortho_log = os.path.join(outdir,'ortho.log')
        print("Saving Orthorectification log at {}".format(ortho_log))
        with open(ortho_log,'w') as f:
            for log in ortho_logs:
                f.write(log)
        if copy_rpc == 1:
            print("Copying RPC from native image to orthoimage in parallel")
            copy_rpc_out = p_map(skysat.copy_rpc,img_list,out_list,num_cpus=cpu_count())
        if orthomosaic == 1:
            print("Will also produce median, weighted average and highest resolution orthomosaic")
            if data == 'triplet':
                # sort images based on timestamps and resolutions
                img_list, time_list = skysat.sort_img_list(out_list)
                res_sorted_list = skysat.res_sort(out_list)

                # define mosaic prefix containing timestamps of inputs
                mos_prefix = '_'.join(np.unique([t.split('_')[0] for t in time_list]))+'__'+'_'.join(np.unique([t.split('_')[1] for t in time_list]))

                # define output filenames
                res_sorted_mosaic = os.path.join(outdir,'{}_finest_orthomosaic.tif'.format(mos_prefix))
                median_mosaic = os.path.join(outdir,'{}_median_orthomosaic.tif'.format(mos_prefix))
                wt_avg_mosaic = os.path.join(outdir,'{}_wt_avg_orthomosaic.tif'.format(mos_prefix))
                indi_mos_list = [os.path.join(outdir,f'{time}_first_orthomosaic.tif') for time in time_list]


                print("producing finest resolution on top mosaic, per-pixel median and wt_avg mosaic")
                all_3_view_mos_logs = p_map(asp.dem_mosaic, [res_sorted_list]*3, [res_sorted_mosaic,median_mosaic,wt_avg_mosaic], 
                                          ['None']*3, [None]*3, ['first','median',None],[None]*3,num_cpus=4)

                print("producing idependent mosaic for different views in parallel")
                indi_mos_count = len(time_list)
                if indi_mos_count>3:
                    tile_size = 400
                else:
                    tile_size = None

                indi_mos_log = p_map(asp.dem_mosaic,img_list, indi_mos_list, ['None']*indi_mos_count, [None]*indi_mos_count, 
                    ['first']*indi_mos_count,[tile_size]*indi_mos_count)

                # write out log files
                out_log = os.path.join(outdir,'science_mode_ortho_mos.log')
                total_mos_log = all_3_view_mos_logs+indi_mos_log
                print("Saving orthomosaic log at {}".format(out_log))
                with open(out_log,'w') as f:
                    for log in itertools.chain.from_iterable(total_mos_log):
                        f.write(log)
            if data == 'video':
                res_sorted_list = skysat.res_sort(out_list)
                print("producing orthomasaic with finest on top")
                res_sorted_mosaic = os.path.join(outdir,'video_finest_orthomosaic.tif')
                print("producing orthomasaic with per-pixel median stats")
                median_mosaic = os.path.join(outdir,'video_median_orthomosaic.tif')
                print("producing orthomosaic with weighted average statistics")
                wt_avg_mosaic = os.path.join(outdir,'video_wt_avg_orthomosaic.tif')
                print("Mosaicing will be done in parallel")
                all_3_view_mos_logs = p_map(asp.dem_mosaic, [res_sorted_list]*3, [res_sorted_mosaic,median_mosaic,wt_avg_mosaic], ['None']*3, [None]*3, ['first','median',None],[None]*3)
                out_log = os.path.join(outdir,'science_mode_ortho_mos.log')
                print("Saving orthomosaic log at {}".format(out_log))
                with open(out_log,'w') as f:
                    for log in all_3_view_mos_logs:
                        f.write(log)

def execute_skysat_stereo(img,outfol,mode,session='rpc',dem=None,texture='high',
    sampling_interval=None,cam_folder=None,ba_prefix=None,writeout_only=False,mvs=0,block=1,crop_map=0,
    full_extent=1,entry_point=0,threads=2,overlap_pkl=None,frame_index=None,job_fn=None,cross_track=False):
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
            job_list = skysat.video_mvs(img,t=session,cam_fol=cam_folder,ba_prefix=ba_prefix,dem=dem,
                           sampling_interval=sampling_interval,texture=texture,
                           outfol=outfol,block=block,frame_index=frame_gdf)

        else:
            if full_extent == 1:
                full_extent = True
            else:
                full_extent = False
            job_list = skysat.prep_video_stereo_jobs(img,t=session,cam_fol=cam_folder,ba_prefix=ba_prefix,
                dem=dem,sampling_interval=sampling_interval,texture=texture,outfol=outfol,block=block,
                frame_index=frame_gdf,full_extent=full_extent,entry_point=entry_point)
    elif mode == 'triplet':
        if crop_map == 1:
            crop_map = True
        else: 
            crop_map = False
            
        job_list = skysat.triplet_stereo_job_list(cross_track=cross_track,t=session,
            threads = threads,overlap_list=overlap_pkl, img_list=img_list, ba_prefix=ba_prefix, 
            cam_fol=cam_folder, dem=dem, crop_map=crop_map,texture=texture, outfol=outfol, block=block,
            entry_point=entry_point)
    if not writeout_only:
        # decide on number of processes
        # if block matching, Plieades is able to handle 30-40 4 threaded jobs on bro node
        # if MGM/SGM, 25 . This stepup is arbitrariry, research on it more.
        # next build should accept no of jobs and stereo threads as inputs
    
        print(job_list[0])
        n_cpu = iolib.cpu_count()
        # no of parallel jobs with user specified threads per job
        jobs = int(n_cpu/threads)
        stereo_log = p_map(asp.run_cmd,['stereo']*len(job_list), job_list, num_cpus=jobs)
        stereo_log_fn = os.path.join(outfol,'stereo_log.log')
        print("Consolidated stereo log saved at {}".format(stereo_log_fn))
    else:
        print(f"Writng jobs at {job_fn}")
        with open(job_fn,'w') as f:
            for idx,job in enumerate(tqdm(job_list)):
                try:                
                    job_str = 'stereo ' + ' '.join(job) + '\n'
                    f.write(job_str)
                except:
                    continue

                    
def grdding_wrapper(pc_list,tr,tsrs=None):
    if tsrs is None:
        print("Projected Target CRS not provided, reading from the first point cloud")
        
        #fetch the PC-center.txt file instead
        # should probably make this default after more tests and confirmation with Oleg
        pc_center = os.path.splitext(pc_list[0])[0]+'-center.txt'
        with open(pc_center,'r') as f:
            content = f.readlines()
        X,Y,Z = [np.float(x) for x in content[0].split(' ')[:-1]]
        ecef_proj = 'EPSG:4978'
        geo_proj = 'EPSG:4326'
        ecef2wgs = Transformer.from_crs(ecef_proj,geo_proj)
        clat,clon,h = ecef2wgs.transform(X,Y,Z)
        epsg_code = f'EPSG:{geo.compute_epsg(clon,clat)}'
        print(f"Detected EPSG code from point cloud {epsg_code}") 
        tsrs = epsg_code
    n_cpu = iolib.cpu_count()    
    point2dem_opts = asp.get_point2dem_opts(tr=tr, tsrs=tsrs,threads=1)
    job_list = [point2dem_opts + [pc] for pc in pc_list]
    p2dem_log = p_map(asp.run_cmd,['point2dem'] * len(job_list), job_list, num_cpus = n_cpu)
    print(p2dem_log)
    
    
def alignment_wrapper_single(ref_dem,source_dem,max_displacement,outprefix,
                             align,trans_only=0,initial_align=None):
    if trans_only == 0:
        trans_only = False
    else:
        trans_only = True
    asp.dem_align(ref_dem,source_dem,max_displacement,outprefix,align,
                  trans_only,threads=iolib.cpu_count(),intial_align=initial_align)
    
def alignment_wrapper_multi(ref_dem,source_dem_list,max_displacement,align,initial_align=None,
                            trans_only=0):
    outprefix_list=['{}_aligned_to{}'.format(os.path.splitext(source_dem)[0],os.path.splitext(os.path.basename(ref_dem))[0]) for source_dem in source_dem_list]
    if trans_only == 0:
        trans_only = False
    else:
        trans_only = True
    n_source = len(source_dem_list)
    
    initial_align = [initial_align]*n_source
    ref_dem_list=[ref_dem] * n_source
    max_disp_list=[max_displacement] * n_source
    align_list=[align] * n_source
    trans_list=[trans_only] * n_source
    p_umap(asp.dem_align,ref_dem_list,source_dem_list,max_disp_list,outprefix_list,
           align_list,trans_list,[1]*n_source,initial_align,num_cpus = iolib.cpu_count())
    
def align_cameras_wrapper(input_camera_list,transform_txt,outfolder,rpc=0,dem='None',img_list=None):
    n_cam=len(input_camera_list)
    if (rpc == 1) & (dem != 'None'):
        print("Will also write RPC files")
        rpc = True
    else:
        dem = None
        img_list = [None] * n_cam
        rpc = False
    transform_list = [transform_txt]*n_cam
    outfolder = [outfolder] * n_cam
    write = [True] * n_cam
    rpc = [rpc] * n_cam
    dem = [dem] * n_cam
    
    p_umap(asp.align_cameras,input_camera_list,transform_list,outfolder,write,rpc,dem,
           img_list,num_cpus = iolib.cpu_count())
    

def dem_mosaic_wrapper(dir,mode='triplet',out_folder=None,identifier=None,tile_size=None,filter_dem=1,min_video_count=2,max_video_nmad=5):
    if out_folder is None:
        out_folder = os.path.join(dir,'composite_dems')
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if identifier is None:
        identifier = ''
    if mode == 'triplet':
        dir_list = sorted(glob.glob(os.path.join(dir,'20*/')))
        print(f"Number of combinations {len(dir_list)}")
        
        def valid_disp(direc):
            try:
                D_sub = iolib.fn_getma(os.path.join(direc,'run-D_sub.tif'),3)
                stats = np.percentile(D_sub.compressed(),(2,98))
                direc_out = glob.glob(os.path.join(direc,'run*{}*-DEM.tif'.format(identifier)))[0]
                out = (direc_out,True)
            except:
                out = (0,False)
            return out

    
        # find all valid DEMs
        total_cpu = iolib.cpu_count()
        n_proc = total_cpu - np.arange(len(dir_list))
        #this setup is so that old pools are discarded and new ones with new number of workers are created
        valid_dem_dir_list = []
        for idx,direc in enumerate(tqdm(dir_list)):
            comb_dir_list = sorted(glob.glob(os.path.join(direc,'*/')))
            results = p_map(valid_disp,comb_dir_list,num_cpus=n_proc[idx])
            t_val = [r[1] for r in results]
            valid_dirs = list(itertools.compress([r[0] for r in results],t_val))
            valid_dem_dir_list.append(valid_dirs)
        
        # naming logic for pairwise and triplet/multiview composites
        mdt1,t1,mdt2,t2 = [[],[],[],[]]
        combination_out_list = []
        if len(dir_list)>3:
            print("Input is cross-track")
        if dir_list[0][-1] == '/':
            dir_list = [x[:-1] for x in dir_list]
        for direc in dir_list:
            combination_out_list.append(os.path.join(out_folder,os.path.basename(direc)+'_wt_avg_mos.tif'))
            a,b,c,d = os.path.basename(direc).split('_')
            mdt1.append(a) #master date (year month date)
            t1.append(b) # time of day in seconds
            mdt2.append(c)
            t2.append(d)
        if len(direc)>3:
            composite_prefix = 'multiview_'+'_'.join(np.unique(mdt1+mdt2))+'__'+'_'.join(np.unique(t1+t2))
        else:
            composite_prefix = 'triplet_'+'_'.join(np.unique(mdt1+mdt2))+'__'+'_'.join(np.unique(t1+t2))
            
        # produce bistereo pairwise mosaics
        len_combinations = len(combination_out_list)
        
        
        if len_combinations > 3:
            # force tiled processing in case of multiview mosaicking
            if not tile_size:
                tile_size = 400
        mos_log = p_map(asp.dem_mosaic,valid_dem_dir_list,combination_out_list,
                ['None']*len_combinations,[None]*len_combinations,
                [None]*len_combinations,[tile_size]*len_combinations)

        if len_combinations > 2:
            print("Producing triplet/multiview composites")
            total_dem_list = list(itertools.chain.from_iterable(valid_dem_dir_list))
            print(f"Mosaicing {len(total_dem_list)} DEM strips using median, wt_avg, count, nmad operators")
            stats_list = [None,'median','nmad','count']
            out_fn_list = [os.path.join(out_folder,
                            '{}_{}_mos.tif'.format(composite_prefix,stat)) for stat in ['wt_avg','median','nmad','count']]
            composite_mos_log = p_map(asp.dem_mosaic,[total_dem_list]*4,out_fn_list,['None']*4,[None]*4,stats_list,
                                [tile_size]*4,num_cpus=4)
        out_log_fn = os.path.join(out_folder,'skysat_triplet_dem_mos.log')
        print("Saving triplet DEM mosaic log at {}".format(out_log_fn))
        with open(out_log_fn,'w') as f:
            for log in mos_log+composite_mos_log:
                f.write(log) 
    if mode == 'video':
        dir_list = sorted(glob.glob(os.path.join(dir,'1*/')))
        valid_video_dir = []
        for video_dir in dir_list:
            try:
                D_sub = iolib.fn_getma(os.path.join(video_dir,'run-D_sub.tif'),3)
                stats = [np.percentile(D_sub.compressed(),(2,98)),np.mean(D_sub.compressed())]
                DEM = glob.glob(os.path.join(video_dir,'run*{}*-DEM.tif'.format(identifier)))[0]
                valid_video_dir.append(video_dir)
            except:
                continue 
        video_dem_list = [glob.glob(os.path.join(dir,f'run*{identifier}*-DEM.tif'))[0] for dir in valid_video_dir]
        stats_list = ['median','count','nmad']
        print('total dems are {}'.format(len(video_dem_list)))
        out_fn_list = [os.path.join(out_folder,'video_{}_mos.tif'.format(stat)) for stat in stats_list]
        dem_mos_log = p_map(asp.dem_mosaic,[video_dem_list]*3,out_fn_list,['None']*3,[None]*3,stats_list,[None]*3) 
        out_log_fn = os.path.join(out_folder,'skysat_video_dem_mos.log')
        with open(out_log_fn,'w') as f:
            for log in dem_mos_log:
                f.write(log)
        if filter_dem == 1:
            print("Filtering DEM using NMAD and count metrics")
            print(f"Filter will use min count of {min_video_count} and max NMAD of {max_video_nmad}")
            mos_ds_list = warplib.memwarp_multi_fn(out_fn_list)
            # Filtered array list contains dem_filtered,nmad_filtered, count_filtered in order
            filtered_array_list = skysat.filter_video_dem_by_nmad(mos_ds_list,min_video_count,max_video_nmad)
            trailing_str = f'_filt_max_nmad{max_video_nmad}_min_count{min_video_count}.tif'
            out_filter_fn_list = [os.path.splitext(fn)[0]+trailing_str for fn in out_fn_list]
            for idx,fn in enumerate(out_filter_fn_list):
                iolib.writeGTiff(filtered_array_list[idx],fn,mos_ds_list[idx])
                

                
                
def dense_match_wrapper(stereo_master_dir,ba_dir,modify_overlap=0,img_fol=None,orig_pickle=None,dense_match_pickle=None,stereo_dir=None,out_overlap_fn=None):
    """
    """
    triplet_stereo_matches = sorted(glob.glob(os.path.join(stereo_master_dir,'20*/*/run*-*disp*.match')))
    print('Found {} dense matches'.format(len(triplet_stereo_matches)))
    if  not os.path.isdir(ba_dir):
        os.makedirs(ba_dir)
    out_dense_match_list = [os.path.join(ba_dir,'run-'+os.path.basename(match).split('run-disp-',15)[1]) for match in triplet_stereo_matches]
    for idx,match in tqdm(enumerate(triplet_stereo_matches)):
        shutil.copy2(match, out_dense_match_list[idx])
    print("Copied all files successfully")
    
    if modify_overlap == 1:
        orig_df = pd.read_pickle(orig_pickle)
        dense_df = pd.read_pickle(dense_match_pickle)
        dense_img1 = list(dense_df.img1.values)
        dense_img2 = list(dense_df.img2.values)
        prioirty_list = list(zip(dense_img1,dense_img2))
        regular_img1 = [os.path.basename(x) for x in orig_df.img1.values]
        regular_img2 = [os.path.basename(x) for x in orig_df.img2.values]
        secondary_list = list(zip(regular_img1,regular_img2))
        # adapted from https://www.geeksforgeeks.org/python-extract-unique-tuples-from-list-order-irrespective/
        # note that I am using the more inefficient answer on purpose, because I want to use image pair order from the dense match overlap list
        total_list = priority_list + secondary_list
        final_overlap_set = set()
        temp = [final_overlap_set.add((a, b)) for (a, b) in total_list
              if (a, b) and (b, a) not in final_overlap_set]
        new_img1 = [os.path.join(img_fol,pair[0]) for pair in list(final_overlap_set)]
        new_img2 = [os.path.join(img_fol,pair[1]) for pair in list(final_overlap_set)]
        if not out_overlap_fn:
            out_overlap = os.path.join(ba_dir,'overlap_list_adapted_from_dense_matches.txt')
        else:
            out_overlap = os.path.join(ba_dir,out_overlap_fn)
        
        print("Saving adjusted overlap list at {}".format(out_overlap))
        with open(out_overlap,'w') as foo:
            for idx,img1 in enumerate(new_img1):
                out_str = '{} {}\n'.format(img1,new_img2[idx])
                f.write(out_str)
                

            
