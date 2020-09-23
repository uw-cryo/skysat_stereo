#! /usr/bin/env python

from pygeotools.lib import iolib,geolib,warplib,malib
from shapely.geometry import Polygon, Point
import geopandas as gpd
from skysat_stereo import asp_utils
import numpy as np
import pandas as pd
import gdal
import os,sys,glob
from shapely import wkt
import gdalconst
from progressbar import ProgressBar
import re
from tqdm import tqdm
from datetime import datetime
from multiprocessing import cpu_count


def skysat_footprint(img_fn,incrs=None):
    """
    Define ground corner footprint from RPC model
    Parameters
    ----------
    img_fn: str
        path to image with embedded RPC info in tiff tag
    incrs: dict
        crs to convert the final footprint into, by default the footprint is returned in geographic coordinates (EPSG:4326)
    Returns
    ----------
    footprint_shp: geopandas geodataframe
        geodataframe containg ground footprints in specified incrs
    """

    if os.path.islink(img_fn):
        img_ds = iolib.fn_getds(os.readlink(img_fn))
    else:
        img_ds = iolib.fn_getds(img_fn)
    nx = img_ds.RasterXSize
    ny = img_ds.RasterYSize
    #img_coord (0,0), (nx,0), (nx,ny), (0,ny) correspond to ul,ur,lr,ll
    z = np.float(img_ds.GetMetadata('RPC')['HEIGHT_OFF'])
    #z = np.float(ht.split(' ',1)[1].splitlines()[0])
    img_x = [0,nx,nx,0]
    img_y = [0,0,ny,ny]
    img_z = [z,z,z,z] #should ideally accept a common height above datum, read from rpc #done
    mx,my = asp_utils.rpc2map(img_fn,img_x,img_y,img_z)
    coord_list = list(zip(mx,my))
    footprint_poly = Polygon(coord_list)
    geo_crs = {'init':'epsg:4326'}
    footprint_shp = gpd.GeoDataFrame(index=[0],geometry=[footprint_poly],crs=geo_crs)
    if incrs:
        footprint_shp = footprint_shp.to_crs(incrs)
    return footprint_shp

def parse_frame_index(frame_index,df_only=False):
    """
    Parse L1A frame_index.csv as a geodataframe/dataframe
    Parameters
    ----------
    frame_index: str
        Path to frame_index.csv file
    df_only: bool
        if True, the the function returns a dataframe, else, it returns a geodataframe
    Returns
    ----------
    out: GeoPandas GeoDataframe/ Pandas Dataframe
        output type depends on input to df_only argument
    """
    df = pd.read_csv(frame_index)
    geo_crs = {'init':'epsg:4326'}
    df_rec = df.copy()
    df_rec['geom'] = df_rec['geom'].apply(fix_polygon_wkt)
    gdf = gpd.GeoDataFrame(df_rec,geometry='geom',crs=geo_crs)
    if df_only:
        out = df
    else:
        out = gdf
    return out

def fix_polygon_wkt(string):
    # from Scott Henderson's notebook
    """
    returns shapely geometry from reformatted WKT
    Parameters
    ----------
    string: str
        wkt string to be formatted
    Returns
    ----------
    out: str
        fixed wkt string
    """
    pre = string[:-2]
    first_point = string.split(',')[0].split('(')[-1]
    fixed = '{},{}))'.format(pre,first_point)
    return wkt.loads(fixed)

def copy_rpc(in_img,out_img):
    """
    Copy rpc info from 1 image to target image
    Parameters
    ----------
    in_img: str
        Path to input image from which RPC will be copied
    out_img: str
        Path to output image to which RPC will be copied
    """
    rpc_fn = in_img
    non_rpc_fn = out_img
    rpc_img = gdal.Open(rpc_fn, gdalconst.GA_ReadOnly)
    non_rpc_img = gdal.Open(non_rpc_fn, gdalconst.GA_Update)
    rpc_data = rpc_img.GetMetadata('RPC')
    non_rpc_img.SetMetadata(rpc_data,'RPC')
    print("Copying rpc from {} to {}".format(in_img,out_img))
    del(rpc_img)
    del(non_rpc_img)

def crop_sim_res_extent(img_list, outfol, vrt=False,rpc=False):
    """
    Warp images to common 'finest' resolution and intersecting extent
    This is useful for stereo processing with mapprojected imagery with the skysat pairs

    Parameters
    ----------
    img_list: list
        list containing two images
    outfol: str
        path to folder where warped images will be saved
    vrt: bool
        Produce warped VRT instead of geotiffs if True
    rpc: bool
        Copy RPC information to warped images if True
    Returns
    ----------
    out: list
        list containing the two warped images, first entry (left image) is the image which was of finer resolution (more nadir) initially
        If the images do not intersect, two None objects are returned in the list
    """
    resample_alg = 'lanczos'
    img1 = img_list[0]
    img2 = img_list[1]
    img1_ds = iolib.fn_getds(img1)
    img2_ds = iolib.fn_getds(img2)
    res1 = geolib.get_res(img1_ds, square=True)[0]
    res2 = geolib.get_res(img2_ds, square=True)[0]
    # set left image as higher resolution, this is repeated for video, but
    # good for triplet with no gsd information
    if res1 < res2:
        l_img = img1
        r_img = img2
        res = res1
    else:
        l_img = img2
        r_img = img1
        res = res2
    # ASP stereo command expects the input to be .tif/.tiff, complains for .vrt
    # Try to save with vrt driver but a tif extension ?
    l_img_warp = os.path.join(outfol, os.path.splitext(os.path.basename(l_img))[0] + '_warp.tif')
    r_img_warp = os.path.join(outfol, os.path.splitext(os.path.basename(r_img))[0] + '_warp.tif')
    if not (os.path.exists(l_img_warp)):
        # can turn on verbose during qa/qc
        # Better to turn off during large runs, writing takes time
        verbose = False
        if not os.path.exists(outfol):
            os.makedirs(outfol)
        try:
            #this will simply break and continue if the images do not intersect
            ds_list = warplib.memwarp_multi_fn([l_img,r_img], r=resample_alg, verbose=verbose, res='min', extent = 'intersection')
            if vrt:
                extent = geolib.ds_extent(ds_list[0])
                res = geolib.get_res(ds_list[0], square=True)
                vrt_options = gdal.BuildVRTOptions(resampleAlg='average',resolution='user',xRes=res[0],yRes=res[1],outputBounds=tuple(extent))
                l_vrt = gdal.BuildVRT(l_img_warp, [l_img, ], options=vrt_options)
                r_vrt = gdal.BuildVRT(r_img_warp, [r_img, ], options=vrt_options)
                # close vrt to save to disk
                l_vrt = None
                r_vrt = None
                out = [l_img_warp, r_img_warp]
            else:
                # I am opting out of writing out vrt, to prevent correlation
                # artifacts. GeoTiffs will be written out in the meantime
                l_img_ma = iolib.ds_getma(ds_list[0])
                r_img_ma = iolib.ds_getma(ds_list[1])
                iolib.writeGTiff(l_img_ma, l_img_warp, ds_list[0])
                iolib.writeGTiff(r_img_ma, r_img_warp, ds_list[1])
                out = [l_img_warp, r_img_warp]
                del(ds_list)
                if rpc:
                    copy_rpc(l_img,l_img_warp)
                    copy_rpc(r_img,r_img_warp)
        except BaseException:
            out = None
    else:
        out = [l_img_warp, r_img_warp]
    return out

def video_mvs(img_folder,t,cam_fol=None,ba_prefix=None,dem=None,sampling_interval=None,texture=None,crop_map=False,outfol=None,frame_index=None,block=0):
    """
    Builds subprocess job list for video collection multiview implementation (explained below) adapted from dAngelo 2016
    - this is mvs implementation
    - each input master view will be jointly triangulated with the next 20 views
    - Sampling interval will be used to select master views
    # Note: ASP's mutliview only supports homography alignment. This will likely fail over steep terrain  given the small skysat footprint.
    Therefore I prefer to use Mapprojected images with alignment=None option ! and results have only been evaluated for mapprojected images

    Parameters
    ----------
    img_folder: str
        Path to image folder
    t: str
        Session to use for stereo processing
    cam_fol: str
        Folder containing tsai camera models (None if using RPC models or using bundle adjusted tsai cameras)
    ba_prefix: str
        ba_prefix for locating the refined tsai camera models, or for locating the *.adjust files for RPC bundle adjusted cameras
    dem: str
        Path to DEM used for mapprojection
    sampling_interval: int
        Number of equally spaced master views to be selected
    texture: str
        use option 'low' input image texture is low, 'normal' for normal textured images. This is used for determining the correlation and refinement kernel
    crop_map: bool
        crop images to map extent if True. Should always be False for video dataset
    outfol: str
        Path to master output folder where the stereo results will be saved
    frame_index: Geopandas GeoDataframe or Pandas Dataframe
        dataframe/geodataframe formed from the truncated frame_index.csv written by skysat_preprocess.py. Will be used in determining images to be processed
    block: int
        Select 0 for the defualt MGM matching, 1 for block matching

    Returns
    ----------
    job_list: list
        list of stereo jobs build on the given parameters
    """
    # only experimented with frame camera models
    job_list = []
    img_list = [glob.glob(os.path.join(img_folder,'{}*.tif'.format(frame)))[0] for frame in frame_index.name.values] # read images
    # Read Cameras
    if ba_prefix:
        cam_list = [glob.glob(ba_prefix + '-' + frame + '*.tsai')[0] for frame in frame_index.name.values]
    else:
        cam_list = [glob.glob(os.path.join(cam_fol, frame + '*.tsai'))[0] for frame in frame_index.name.values]
    num_pairs = 20 # can be accepted as input
    # Compute equally spaced indices for the master images to be chosen
    ref_idx = np.linspace(0,len(img_list)-1-num_pairs,sampling_interval,dtype=np.int)
    source_idexs = [list(np.arange(idx+1,idx+1+num_pairs)) for idx in ref_idx] #this is list of list containing source_ids for corresponding reference_id
    if os.path.islink(img_list[0]):
        symlink = True
    else:
        symlink = False
    # This loop prepares the jobs
    for i in tqdm(range(0, len(ref_idx))):
        total_images = len(img_list)
        if symlink:
            ref_image = os.path.islink(img_list[ref_idx[i]])
            source_images = [os.path.islink(img_list[source_idexs[i][k]]) for k in range(len(source_idexs[i]))]
        else:
            ref_image = img_list[ref_idx[i]]
            source_images = [img_list[source_idexs[i][k]] for k in range(len(source_idexs[i]))]
        ref_camera = cam_list[ref_idx[i]]
        source_cameras = [cam_list[source_idexs[i][k]] for k in range(len(source_idexs[i]))]
        print('Number of source images: {}'.format(len(source_images)))
        ref_prefix = os.path.splitext(os.path.basename(ref_image))[0]
        outstr = ref_prefix + '_mvs'
        outfolder = os.path.join(outfol, outstr)
        source_prefixes = [os.path.splitext(os.path.basename(x))[0] for x in source_images]
        if 'map' in ref_prefix:
            ref_prefix = ref_prefix.split('_map', 15)[0]
            source_prefixes = [x.split('_map', 15)[0] for x in source_prefixes]
        ba = None
        stereo_args = [ref_image] + source_images + [ref_camera] + source_cameras
        if block == 1:
            spm = 2
            stereo_mode = 0
            corr_tile_size = 1024
            if texture == 'low':
                rfne_kernel = [21, 21]
                corr_kernel = [35, 35]
                lv = 5
            else:
                rfne_kernel = [15, 15]
                corr_kernel = [21, 21]
                lv = 5
        else:
            spm = 2
            stereo_mode = 2
            corr_tile_size = 6400
            if texture == 'low':
                rfne_kernel = [21, 21]
                corr_kernel = [9, 9]
                lv = 5
            else:
                rfne_kernel = [15, 15]
                corr_kernel = [7, 7]
                lv = 5
        stereo_opt = asp_utils.get_stereo_opts(session=t,align='None',lv=lv,corr_kernel=corr_kernel,rfne_kernel=rfne_kernel,stereo_mode=stereo_mode,spm=spm,cost_mode=4,corr_tile_size=corr_tile_size,mvs=True)
        outfolder = outfolder + '/run'
        print(outfolder)
        stereo_args = stereo_args + [outfolder]
        if 'map' in t:
            stereo_args = stereo_args + [dem]
        job_list.append(stereo_opt + stereo_args)
    return job_list

def prep_video_stereo_jobs(img_folder,t,threads=4,cam_fol=None,ba_prefix=None,dem=None,sampling_interval=None,texture=None,crop_map=False,outfol=None,frame_index=None,block=0,full_extent=False):
    """
    Builds subprocess job list for video collection pairwise implementation

    Parameters
    ----------
    img_folder: str
        Path to image folder
    threads: int
        number of threads to use for each stereo step
    t: str
        Session to use for stereo processing
    cam_fol: str
        Folder containing tsai camera models (None if using RPC models or using bundle adjusted tsai cameras)
    ba_prefix: str
        ba_prefix for locating the refined tsai camera models, or for locating the *.adjust files for RPC bundle adjusted cameras
    dem: str
        Path to DEM used for mapprojection
    sampling_interval: int
        interval at which source images will be chosen for sequential reference images in the list
    texture: str
        use option 'low' input image texture is low, 'normal' for normal textured images. This is used for determining the correlation and refinement kernel
    crop_map: bool
        crop images to map extent if True. Cropping to common resolution and extent should give best results in mapprojected images
    outfol: str
        Path to master output folder where the stereo results will be saved
    frame_index: Geopandas GeoDataframe or Pandas Dataframe
        dataframe/geodataframe formed from the truncated frame_index.csv written by skysat_preprocess.py. Will be used in determining images to be processed
    block: int
        Select 0 for the defualt MGM matching, 1 for block matching
    full_extent: bool
        If True, stereo pairs with smaller baselines (sampling interval of 5) will be padded at the beginning and end of the video sequence

    Returns
    ----------
    job_list: list
        list of stereo jobs build on the given parameters
    """

     #check here if the sampling is greater than 10 seconds
     # if this is the case, then output DEM is not going to capture the full extent of the video
     # so after initial pairing at user defined rates, append to list pairs at this range at the starting and end.
     # check the interval first
    try:
        img_list = [glob.glob(os.path.join(img_folder,'{}*.tiff'.format(frame)))[0] for frame in frame_index.name.values]
    except:
        img_list = [glob.glob(os.path.join(img_folder,'{}*.tif'.format(frame)))[0] for frame in frame_index.name.values]
    if 'pinhole' in t:
        if ba_prefix:
            cam_list = [glob.glob(ba_prefix + '-' + frame + '*.tsai')[0] for frame in frame_index.name.values]
        else:
            cam_list = [glob.glob(os.path.join(cam_fol, frame + '*.tsai'))[0] for frame in frame_index.name.values]
    print("Sampling interval is {}".format(sampling_interval))
    #find min baseline between first and next in seconds
    dt_list = [datetime.strptime(date.split('+00:00')[0],'%Y-%m-%dT%H:%M:%S.%f') for date in frame_index.datetime.values]
    min_sec = (dt_list[sampling_interval]-dt_list[0]).total_seconds() #this is interval between 1 and next stereo pair image
    succesive_sec = (dt_list[1]-dt_list[0]).total_seconds() #this is interval between 2 images in the sequence

    end_point = len(img_list)-1-sampling_interval
    ref_idx = np.linspace(0,end_point,num=end_point+1,dtype=int)
    source_idx = ref_idx+sampling_interval
    print("seleceted {} stereo pairs by default".format(len(source_idx)))
    # if the sampling interval causes the image pairs to be formed at higher interval, and full_extent is chosen, then this will be used for padding stereo pairs (See docstring).
    if (min_sec > 10) & full_extent:
        #need to maintain 10 seconds interval minimum
        #stereo resulst are poor for lower intervals than that
        secondary_interval = np.int(np.round(10/succesive_sec))
        print("will buffer start and end frames with interval of {}".format(secondary_interval))
        end_point1 = source_idx[0]-secondary_interval
        ref_1 = np.linspace(ref_idx[0],end_point1,num=end_point1-ref_idx[0]+1,dtype=int)
        source_1 = ref_1+secondary_interval
        end_point2 = source_idx[-1]-secondary_interval
        ref_2 = np.linspace(ref_idx[-1],end_point2,num=end_point2-ref_idx[-1]+1,dtype=int)
        source_2 = ref_2+secondary_interval
        ref_idx = list(ref_idx)+list(ref_1)+list(ref_2)
        source_idx = list(source_idx)+list(source_1)+list(source_2)
        print("added additional {} stereo pairs".format(len(source_1)+len(source_2)))
    job_list = []
    if os.path.islink(img_list[0]):
        symlink = True
    else:
        symlink = False
    # Build jobs
    for i in tqdm(range(0, len(ref_idx))):
        if symlink:
            in_img_1 = os.readlink(img_list[ref_idx])
            in_img_2 = os.readlink(img_list[source_idx])
        else:
            in_img_1 = img_list[ref_idx[i]]
            in_img_2 = img_list[source_idx[i]]
        img1 = os.path.basename(in_img_1)
        img2 = os.path.basename(in_img_2)
        pref_1 = os.path.splitext(img1)[0]
        pref_2 = os.path.splitext(img2)[0]
        # print(pref_1)
        if 'map' in t:
            pref_1 = pref_1.split('_PAN', 15)[0] + '_PAN'
            pref_2 = pref_2.split('_PAN', 15)[0] + '_PAN'
        mask1 = frame_index['name'] == pref_1
        mask2 = frame_index['name'] == pref_2
        df_img1 = frame_index[mask1]
        df_img2 = frame_index[mask2]
        gsd1 = df_img1.gsd.values[0]
        gsd2 = df_img2.gsd.values[0]
        if 'pinhole' in t:
            cam_1 = cam_list[ref_idx[i]]
            cam_2 = cam_list[source_idx[i]]
        # always map stereo disparity with finer resolution image as reference
        if gsd1 < gsd2:
            in_img1 = in_img_1
            in_img2 = in_img_2
            pref1 = pref_1
            pref2 = pref_2
            if 'pinhole' in t:
                cam1 = cam_1
                cam2 = cam_2
        else:
            in_img1 = in_img_2
            in_img2 = in_img_1
            pref1 = pref_2
            pref2 = pref_1
            if 'pinhole' in t:
                cam2 = cam_1
                cam1 = cam_2
        convergence = np.round(asp_utils.convergence_angle(df_img1.sat_az.values[0],df_img1.sat_elev.values[0],df_img2.sat_az.values[0],df_img2.sat_elev.values[0]),2)
        outstr = '{}_{}_{}'.format(pref1,pref2,convergence)
        outfolder = os.path.join(outfol, outstr)
        if 'pinhole' in t:
            if t == 'pinholemappinhole':
                in_img1, in_img2 = crop_sim_res_extent(
                    [in_img1, in_img2], outfolder)
            ba = None
            outfolder = outfolder + '/run'
            stereo_args = [in_img1, in_img2, cam1, cam2, outfolder]
        else:
            stereo_args = [in_img1, in_img2, outfolder]
            if ba_prefix:
                ba = ba_prefix
            else:
                ba = None
        if 'map' in t:
            align = 'None'
            stereo_args.append(dem)
        else:
            align = 'AffineEpipolar'
        # add as list ?
        if block == 1:
            print("Performing block matching")
            spm = 2 #Bayes EM
            stereo_mode = 0 #Block matching
            cost_mode = 2 #NCC
            corr_tile_size = 1024
            if texture == 'low':
                rfne_kernel = [21, 21]
                corr_kernel = [35, 35]
                lv = 2
            else:
                rfne_kernel = [15, 15]
                corr_kernel = [21, 21]
                lv = 5
        else:
            cost_mode = 4 #Preffered MGM cost-mode
            spm = 2 # Bayes EM
            stereo_mode = 2 #MGM
            corr_tile_size = 6400
            if texture == 'low':
                rfne_kernel = [21, 21]
                corr_kernel = [9, 9]
                lv = 2
            else:
                rfne_kernel = [15, 15]
                corr_kernel = [7, 7]
                lv = 5
        stereo_opt = asp_utils.get_stereo_opts(session=t,threads=threads,ba_prefix=ba,align=align,lv=lv,corr_kernel=corr_kernel,rfne_kernel=rfne_kernel,stereo_mode=stereo_mode,spm=spm,cost_mode=cost_mode,corr_tile_size=corr_tile_size)
        print(stereo_opt + stereo_args)
        job_list.append(stereo_opt + stereo_args)
    return job_list

def triplet_stereo_job_list(overlap_list,t,img_list,threads=4,ba_prefix=None,cam_fol=None,dem=None,texture='high',crop_map=True,outfol=None,block=0):
    """
    Builds subprocess job list for triplet collection pairwise implementation

    Parameters
    ----------
    overlap_list: str
        path to pkl file containing the overlap list and overlap percentage
    t: str
        Session to use for stereo processing
    threads: int
        number of threads to use for 1 processing
    img_list: list
        List of paths of input images
    ba_prefix: str
        ba_prefix for locating the refined tsai camera models, or for locating the *.adjust files for RPC bundle adjusted cameras
    cam_fol: str
        Folder containing tsai camera models (None if using RPC models or using bundle adjusted tsai cameras
    dem: str
        Path to DEM used for mapprojection
    texture: str
        use option 'low' input image texture is low, 'normal' for normal textured images. This is used for determining the correlation and refinement kernel
    crop_map: bool
        crop images to map extent if True. Cropping to common resolution and extent should give best results in mapprojected images
    outfol: str
        Path to master output folder where the stereo results will be saved
    block: int
        Select 0 for the defualt MGM matching, 1 for block matching

    Returns
    ----------
    job_list: list
        list of stereo jobs build on the given parameters
    """

    job_list = []
    print(img_list)
    l_img_list = []
    r_img_list = []
    triplet_df = prep_trip_df(overlap_list)
    df_list = [x for _, x in triplet_df.groupby('identifier_text')]
    for df in df_list:
        outfolder = os.path.join(outfol, df.iloc[0]['identifier_text'])
        img1_list = df.img1.values
        img2_list = df.img2.values
        pbar = ProgressBar()
        print("preparing stereo jobs")
        for i, process in enumerate(pbar(img1_list)):
            img1 = img1_list[i]
            img2 = img2_list[i]
            IMG1 = os.path.splitext(os.path.basename(img1))[0]
            IMG2 = os.path.splitext(os.path.basename(img2))[0]
            out = outfolder + '/' + IMG1 + '__' + IMG2
            if 'rpc' in t:
                rpc = True
            else:
                rpc = False
            # https://www.geeksforgeeks.org/python-finding-strings-with-given-substring-in-list/
            try:
                img1 = [x for x in img_list if re.search(IMG1, x)][0]
                img2 = [x for x in img_list if re.search(IMG2, x)][0]

            except BaseException:
                continue
            if 'map' in t:
                out = out + '_map'
                try:
                    if crop_map:
                        in_img1, in_img2 = crop_sim_res_extent([img1, img2], out,rpc=rpc)
                    else:
                        in_img1, in_img2 = [img1,img2]
                except BaseException:
                    continue
            else:
                in_img1 = img1
                in_img2 = img2
            out = os.path.join(out, 'run')
            IMG1 = os.path.splitext(os.path.basename(in_img1))[0]
            IMG2 = os.path.splitext(os.path.basename(in_img2))[0]
            if 'map' in t:
                IMG1 = IMG1.split('_map',15)[0]
                IMG2 = IMG2.split('_map',15)[0]
            if 'pinhole' in t:
                if ba_prefix:
                    cam1 = glob.glob(
                        os.path.abspath(ba_prefix) + '-' + IMG1 + '*.tsai')[0]
                    cam2 = glob.glob(
                        os.path.abspath(ba_prefix) + '-' + IMG2 + '*.tsai')[0]
                else:
                    cam1 = glob.glob(os.path.join(os.path.abspath(cam_fol),'*'+IMG1 + '*.tsai'))[0]
                    cam2 = glob.glob(os.path.join(os.path.abspath(cam_fol),'*'+IMG2 + '*.tsai'))[0]
                stereo_args = [in_img1, in_img2, cam1, cam2, out]
                align = 'AffineEpipolar'
                ba = None
            elif 'rpc' in t:
                stereo_args = [in_img1, in_img2, out]
                align = 'AffineEpipolar'
                if ba_prefix:
                    ba = os.path.abspath(ba_prefix)
                else:
                    ba = None
            if 'map' in t:
                stereo_args.append(dem)
                align = 'None'
            if block == 1:
                print("Performing block matching")
                spm = 2
                stereo_mode = 0
                cost_mode = 2
                corr_tile_size = 1024
                if texture == 'low':
                    rfne_kernel = [21, 21]
                    corr_kernel = [35, 35]
                    lv = 5
                else:
                    rfne_kernel = [15, 15]
                    corr_kernel = [21, 21]
                    lv = 5
            else:
                cost_mode = 4
                spm = 2
                stereo_mode = 2
                corr_tile_size = 6400
                if texture == 'low':
                    rfne_kernel = [21, 21]
                    corr_kernel = [9, 9]
                    lv = 5
                else:
                    rfne_kernel = [15, 15]
                    corr_kernel = [7, 7]
                    lv = 5
            #write out file for dense matches logic
            # if mapprojected stereo, then need to update overlap list
            if 'map' in t:
            	l_img_list.append(os.path.basename(in_img1).split('_warp.tif',15)[0]+'.tif')
            	r_img_list.append(os.path.basename(in_img2).split('_warp.tif',15)[0]+'.tif')
            # Prepare stereo options
            stereo_opt = asp_utils.get_stereo_opts(session=t,threads=threads,ba_prefix=ba,align=align,corr_kernel=corr_kernel,lv=lv,rfne_kernel=rfne_kernel,stereo_mode=stereo_mode,spm=spm,cost_mode=cost_mode,corr_tile_size=corr_tile_size)
            job_list.append(stereo_opt + stereo_args)
    overlap_new = os.path.join(outfol,'overlap_list_as_per_dense_ba.pkl')
    df_out = pd.DataFrame({'img1':l_img_list,'img2':r_img_list})
    print("Saving modified overlap pkl as per dense match criteria at {}".format(overlap_new))
    if not os.path.exists(outfol):
        os.makedirs(outfol)
    #df.to_pickle(overlap_new)
    #return concatenated job list
    return job_list

def prep_trip_df(overlap_list, true_stereo=True):
    """
    Prepare dataframe from input plckle file containing overlapping images with percentages
    Parameters
    ----------
    overlap_list: str
        Path to pickle file containing overlapping images produced from skysat_overlap_parallel.py
    true_stereo: bool
        True means output dataframe has only pairs fromed by scenes from different views
    Returns
    ----------
    df: Pandas Dataframe
        dataframe cotianing list of plausible overlapping stereo pairs
    """
    # check date, if date not equal drop
    # then check time, if time equal drop
    # if satellite unequal, drop
    # then check overlap percent
    # then make different folders for different time period
    # to add timestamp/convergence angle filter, as list grows
    df = pd.read_pickle(overlap_list)
    sat = os.path.basename(df.iloc[0]['img1']).split('_',15)[2].split('d',15)[0]
    ccd = os.path.basename(df.iloc[0]['img1']).split('_',15)[2].split('d',15)[1]
    date = os.path.basename(df.iloc[0]['img1']).split('_', 15)[0]
    time = os.path.basename(df.iloc[0]['img1']).split('_', 15)[1]
    df['sat1'] = [os.path.basename(x).split('_', 15)[2].split('d', 15)[0] for x in df.img1.values]
    df['sat2'] = [os.path.basename(x).split('_', 15)[2].split('d', 15)[0] for x in df.img2.values]
    df['date1'] = [os.path.basename(x).split('_', 15)[0] for x in df.img1.values]
    df['date2'] = [os.path.basename(x).split('_', 15)[0] for x in df.img2.values]
    df['time1'] = [os.path.basename(x).split('_', 15)[1] for x in df.img1.values]
    df['time2'] = [os.path.basename(x).split('_', 15)[1] for x in df.img2.values]
    if true_stereo:
        # returned df has only those pairs which form true stereo
        df = df[df['date1'] == df['date2']]
        df = df[df['time1'] != df['time2']]
        df = df[df['sat1'] == df['sat2']]
    # filter to overlap percentage of around 5%
    df['overlap_perc'] = df['overlap_perc'] * 100
    df = df[(df['overlap_perc'] > 2)]
    df['identifier_text'] = df['date1'] + '_' + df['time1'] + '_' + df['date2'] + '_' + df['time2']
    print("Number of pairs over which stereo will be attempted are {}".format(len(df)))
    return df

def frame_intsec(img_list,proj,min_overlap):
    """
    Compute overlapping pairs with overlap percentage

    Parameters
    ----------
    img_list: list
        List containing paths to the two images
    proj: str
        proj4 string to transform the frames before computing overlap percentage
    min_overlap: float
        minimum overlap percentage to consider (between 0 to 1)

    Returns
    ----------
    valid: bool
       True if frame intersect as per user defined minimum overlap percentage
    perc_intsect: float
       Float value returning percentage of overlap ( ranges from 0: no overlap to 1: full overlap)
    """

    #shplist contains shp1,shp2
    img1 = img_list[0]
    img2 = img_list[1]
    shp1 = skysat_footprint(img1,proj)
    shp2 = skysat_footprint(img2,proj)
    if shp1.intersects(shp2)[0]:
        intsect = gpd.overlay(shp1,shp2, how='intersection')
        area_shp1 = shp1['geometry'].area.values[0]
        area_shp2 = shp2['geometry'].area.values[0]
        area_intsect = intsect['geometry'].area.values[0]
        perc_intsect = area_intsect/area_shp1 #we should be fine here with only 1 as skysat collects are mostly uniform ?
        if perc_intsect>=min_overlap:
            valid=True
        else:
            valid=False
    else:
        valid=False
        perc_intsect=0.0
    return valid,perc_intsect

def sort_img_list(img_list):
    """
    sort triplet stereo imagery into forward, nadir and aft (in the given order)
    The function is simple, just uses info in the filename string for now
    Parameters
    ----------
    img_list: list
        list of triplet stereo images
    Returns
    ----------
    for_img_list: list
        list containing filenames for forward images
    nadir_img_list: list
        list containing filenames for nadir images
    aft_img_list: list
        list containing filenames for aft images
    for_time: str
        the single time for all forward viewing images
    nadir_time: str
        the single time for all nadir viewing images
    aft_time: str
        the single time for all aft viewing images
    """
    #list of unique image acquisition time list
    time_list = sorted(list(np.unique(np.array([os.path.basename(img).split('_',15)[1] for img in img_list]))))
    for_time = time_list[0]
    nadir_time = time_list[1]
    aft_time = time_list[2]
    #make seperate image list
    #https://stackoverflow.com/questions/2152898/filtering-a-list-of-strings-based-on-contents
    for_img_list = [k for k in img_list if for_time in k]
    nadir_img_list = [k for k in img_list if nadir_time in k]
    aft_img_list = [k for k in img_list if aft_time in k]
    return (for_img_list,nadir_img_list,aft_img_list,for_time,nadir_time,aft_time)

def res_sort(img_list):
    """
    sort images based on resolution, finest resolution on top
    Parameters
    ----------
    img_list: list
        list of images to be sorted
    Returns
    ----------
    sorted_img_list: list
        list of sorted images with finest resolution on top
    """
    ds_list = [iolib.fn_getds(img) for img in img_list]
    res_list = [geolib.get_res(ds,square=True)[0] for ds in ds_list]
    #https://www.geeksforgeeks.org/python-sort-values-first-list-using-second-list
    zipped_pairs = zip(res_list, img_list)
    sorted_img_list = [x for _,x in sorted(zipped_pairs)]
    return sorted_img_list


def filter_video_dem_by_nmad(ds_list,min_count=2,max_nmad=5):
    """
    Filter Video DEM composites using NMAD and count stats
    This function will look for and eliminate pixels in median DEM where less than 
    <min_count> pairwise DEMs contributed and their vertical variability (NMAD) is higher than <max_nmad>
    Parameters
    -----------
    ds_list: list
        list of gdal datasets, containing median, count and nmad composites in order
    min_count: numeric
        minimum count to use in filtering
    max_nmad: numeric
        maximum NMAD variability to filter, if count is also <= min_count
    Returns
    -----------
    dem_filt: masked array
        filtered DEM
    nmad_filt_c: masked array
        filtered NMAD map
    count_filt_c: masked array
        filtered count map
    """

    dem = iolib.ds_getma(ds_list[0])
    count = iolib.ds_getma(ds_list[1])
    nmad = iolib.ds_getma(ds_list[2])
    
    nmad_filt = np.ma.masked_where(nmad>5,nmad)
    count_filt = np.ma.masked_where(count<2,count)
    valid_mask = malib.common_mask([nmad_filt,count_filt])
    nmad_filt_c = np.ma.array(nmad_filt,mask = valid_mask)
    count_filt_c = np.ma.array(count_filt,mask = valid_mask)
    dem_filt = np.ma.array(dem,mask = valid_mask)
    return dem_filt,nmad_filt_c,count_filt_c
