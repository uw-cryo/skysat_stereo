#! /usr/bin/env python
import numpy as np
from pygeotools.lib import geolib,iolib
import os,sys,glob,shutil
import pandas as pd
import geopandas as gpd
from pyproj import Proj, transform
from rpcm import rpc_from_geotiff

#TODO: 
# mapproject and dem_mosaic

def read_tsai_dict(tsai):
    """
    read tsai frame model from asp and return a python dictionary containing the parameters
    See ASP's frame camera implementation here: https://stereopipeline.readthedocs.io/en/latest/pinholemodels.html
    input: tsai: path to ASP frame camera model 
    output: dictionary containing camera model parameters 
    #TODO: support distortion model
    """
    camera = os.path.basename(tsai)
    with open(tsai,'r') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    fu = np.float(content[2].split(' = ',4)[1]) # focal length in x
    fv = np.float(content[3].split(' = ',4)[1]) # focal length in y
    cu = np.float(content[4].split(' = ',4)[1]) # optical center in x
    cv = np.float(content[5].split(' = ',4)[1]) # optical center in y
    cam = content[9].split(' = ',10)[1].split(' ')
    cam_cen = [np.float(x) for x in cam] # camera center coordinates in ECEF 
    rot = content[10].split(' = ',10)[1].split(' ')
    rot_mat = [np.float(x) for x in rot] # rotation matrix for camera to world coordinates transformation 
    pitch = np.float(content[11].split(' = ',10)[1]) # pixel pitch
    cam_cen_lat_lon = geolib.ecef2ll(cam_cen[0],cam_cen[1],cam_cen[2]) # camera center coordinates in geographic coordinates
    tsai_dict = {'camera':camera,'focal_length':(fu,fv),'optical_center':(cu,cv),'cam_cen_ecef':cam_cen,'cam_cen_wgs':cam_cen_lat_lon,'rotation_matrix':rot_mat,'pitch':pitch}
    return tsai_dict

def make_tsai(outfn,cu,cv,fu,fv,rot_mat,C,pitch):
    """
    write out pinhole model with given parameters
    See ASP's frame camera implementation here: https://stereopipeline.readthedocs.io/en/latest/pinholemodels.html
    inputs:
    # cu,cv: optical center (x,y)
    # fu,fv: focal length (x,y)
    # rot_mat: 3*3 numpy array of rotation matrix
    # camera_center: 1*3 camera center in ecef coordinates (x,y,z)
    # pitch: pixel_pitch
    # outfn: path where frame camera will be saved
    - NOTE:
        # Cameras with ASP's distortion model is currently not implemneted
    """
    out_str = f'VERSION_4\nPINHOLE\nfu = {fu}\nfv = {fv}\ncu = {cu}\ncv = {cv}\nu_direction = 1 0 0\nv_direction = 0 1 0\nw_direction = 0 0 1\nC = {C[0]} {C[1]} {C[2]}\nR = {rot_mat[0][0]} {rot_mat[0][1]} {rot_mat[0][2]} {rot_mat[1][0]} {rot_mat[1][1]} {rot_mat[1][2]} {rot_mat[2][0]} {rot_mat[2][1]} {rot_mat[2][2]}\npitch = {pitch}\nNULL'
    with open(outfn,'w') as f:
        f.write(out_str)

def cam_gen(img,fl=553846.153846,cx=1280,cy=540,pitch=1,ht_datum=None,gcp_std=1,out_fn=None,out_gcp=None,datum='WGS84',refdem=None,camera=None,frame_index=None):
        """
        function to build command for ASP's cam_gen module, to initiate frame camera models from input rpc model or frame_index (skysat video)
        Theory: Uses camera resection principle to refine camera extrinsic from given ground control point (for rpc cameras as input, also generates initial camera extrinsic, which is then refined from tandard resection principle)
        See ASP documentation: https://stereopipeline.readthedocs.io/en/latest/tools/cam_gen.html
        Also see simple python implementation: https://github.com/jeffwalton/photogrammetry-resection/blob/master/resection.py.
        inputs:
        img: path to image file for which camera is to be generated
        camera intrinsics:
        - fl: focal length (default at 553846.153846 px for skysat)
        - cx,cy: Optical center (default at 1280,540 px for skysat)
        - pitch: pixel pitch (default at 1, assuming l1a is input, for l1b (surper-resolution), use 0.8)
        gcp_related_vars
        - ht_datum: height values to use for getting ground control from corner coordinates, in case missing in DEM
        - gcp_std: standard deviation to be assigned to gcp (think of it as how accurate you think your gcps are, used as weights by ASP in bundle adjustment, default: 1)
        - datum: vertical reference datum (default to WGS84)
        - refdem: path to reference DEM to compute the ground control
        - camera: path to initial camera (e.g. RPC camera for L1B triplets)
        - frame_index: path to frame_index.csv containing attitude ephermis data for L1A skysat videos
        output filenames:
        - out_fn: path to store frame camera model at
        - out_gcp: path to store gcp file
        outputs: Nothing, writes out camera model and gcp at specied locations
        """
        cam_gen_opt = []
        cam_gen_opt.extend(['--focal-length',str(fl)])
        cam_gen_opt.extend(['--optical-center',str(cx),str(cy)])
        cam_gen_opt.extend(['--pixel-pitch',str(pitch)])
        if ht_datum:
            cam_gen_opt.extend(['--height-above-datum',str(ht_datum)])
        cam_gen_opt.extend(['--gcp-std',str(gcp_std)])
        cam_gen_opt.extend(['-o',out_fn])
        cam_gen_opt.extend(['--gcp-file',out_gcp])
        cam_gen_opt.extend(['--datum',datum])
        cam_gen_opt.extend(['--reference-dem',refdem])
        if camera:
            cam_gen_opt.extend(['--input-camera',camera])
        if frame_index:
            cam_gen_opt.extend(['--frame-index',frame_index])
        cam_gen_opt.extend(['--refine-camera'])
        cam_gen_args = [img]
        run_cmd('cam_gen',cam_gen_args+cam_gen_opt,msg='Running camgen command for image {}'.format(os.path.basename(img)))

def clean_img_in_gcp(row):
        """
        helper function to return basename of image path
        See clean_gcp function for main implementation 
        """
        return os.path.basename(row[7])

def clean_gcp(gcp_list,outdir):
    """
    ASP's cam_gen writes full path for images in the GCP files. This does not play well during bundle adjustment.
    The function returns a consolidated gcp file with all images paths only containing basenames so that bundle adjustment can roll along
    See ASP's gcp logic here: https://stereopipeline.readthedocs.io/en/latest/tools/bundle_adjust.html#bagcp
    inputs:
    - gcp_list: list of gcp paths
    - outdir: directory where clean consolidated gcp will be saved as clean_gcp.gcp
    """
    df_list = [pd.read_csv(x,header=None,delimiter=r"\s+") for x in gcp_list]
    gcp_df = pd.concat(df_list, ignore_index=True)
    gcp_df[7] = gcp_df.apply(clean_img_in_gcp,axis=1)
    gcp_df[0] = np.arange(len(gcp_df))
    gcp_df.to_csv(os.path.join(outdir,'clean_gcp.gcp'),sep = ' ',index=False,header=False)
    gcp_df.to_csv(os.path.join(outdir,'clean_gcp.csv'),sep = ' ',index=False)

def rpc2map (img,imgx,imgy,imgz=0):
    """
    generate 3D world coordinates from input image pixel coordinates using the RPC model
    See rpcm: https://github.com/cmla/rpcm/blob/master/rpcm/rpc_model.py for implementation
    inputs:
    img: image file containing RPC in in gdal tags
    imgx,imgy,imgz: Image x,y in pixel units, z: height in world coordinates
    output: mx,my: numpy arrays containing longitudes (mx) and latitudes (my) in geographic (EPSG:4326) coordinates
    """
    rpc = rpc_from_geotiff(img)
    mx,my = rpc.localization_iterative(imgx,imgy,imgz)
    return mx,my


def get_ba_opts(ba_prefix, camera_weight=0, overlap_list=None, overlap_limit=None, initial_transform=None, input_adjustments=None, flavor='general_ba', session='nadirpinhole', gcp_transform=False,num_iterations=2000,lon_lat_lim=None,elevation_limit=None):
    """
    prepares bundle adjustment cmd for ASP
    most of the parameters are tweaked to handle Planet SkySat data
    See ASP's bundle adjustment documentation: https://stereopipeline.readthedocs.io/en/latest/tools/bundle_adjust.html#
    inputs: 
    ba_prefix: prefix with which bundle adjustment results will be saved (can be a path, general convention for repo is some path with run prefix, eg., ba_pinhole1/run
    camera_weight: weight to be given to camera extrinsic to allow/prevent their movement during optimazation, default is 0, cameras are allowed to float as much the solver wants to
    overlap_list: path to a text file contianing 2 images per line, which are expected to be overlapping. This limits matching to the pairs in the list only. Very useful for SkySat triplet
    overlap_limit: if images are taken in sequence, the parameter(m) supplied here will only perform matching for an image with its (m) forward neighbours. Very useful for SkySat video
    initial_transform: apply an initial transform supplied as a 4*4 matrix in a text file (such as those output from ASP pc_align)
    input_adjustments: if handling RPC model, this will be adjustments from a previous invocation of the program
    flavor: flavors of bundle adjustment to chose from. 'general_ba' will prepare arguments for simple 1 round bundle_adjustment. `2_round_gcp_1` will prepare arguments for fully free camera optimazationwhile `2_round_gcp_2` prepares arguments for only shifting the optimized camera set a hole to the median transform from all gcps. This is genrally a part of 2 step process where `2_round_gcp_2` follows a `2_round_gcp_1` invocation.
    session: bundle adjustment session, default is nadirpinhole (prefered approach for skysat)
    gcp_transform: tranform using gcp argument, set to true during `2_round_gcp_2`.
    num_iterations: number of solver iterations, default at 2000.
    lon_lat_limit: Clip the match point/gcps to lie only within this limit after optimization
    elevation_limit: Clip the match point/gcps to lie only within this limit after optimization
    
    returns: a set of arguments as list to be run using subprocess command.
    """
     
    ba_opt = []
    ba_opt.extend(['-o', ba_prefix])
    ba_opt.extend(['--min-matches', '4'])
    ba_opt.extend(['--disable-tri-ip-filter'])
    ba_opt.extend(['--force-reuse-match-files'])
    ba_opt.extend(['--ip-per-tile', '4000'])
    ba_opt.extend(['--ip-inlier-factor', '0.2'])
    ba_opt.extend(['--ip-num-ransac-iterations', '1000'])
    ba_opt.extend(['--skip-rough-homography'])
    ba_opt.extend(['--min-triangulation-angle', '0.0001'])
    ba_opt.extend(['--save-cnet-as-csv'])
    ba_opt.extend(['--individually-normalize'])
    ba_opt.extend(['--camera-weight', str(camera_weight)])
    ba_opt.extend(['-t', session])
    ba_opt.extend(['--remove-outliers-params', '75 3 5 6'])
    # How about adding num random passes here ? Think about it, it might help if we are getting stuck in local minima :)
    if session == 'nadirpinhole':
        ba_opt.extend(['--inline-adjustments'])
    if flavor == '2round_gcp_1':
        ba_opt.extend(['--num-iterations', str(num_iterations)])
        ba_opt.extend(['--num-passes', '3'])
    elif flavor == '2round_gcp_2':
        ba_opt.extend(['--num-iterations', '0'])
        ba_opt.extend(['--num-passes', '1'])
        # gcp_transform=True
        if gcp_transform:
            ba_opt.extend(['--transform-cameras-using-gcp'])
        # maybe add gcp arg here, can be added when function is called as well
    if initial_transform:
        ba_opt.extend(['--initial-transform', initial_transform])
    if input_adjustments:
        ba_opt.extend(['--input-adjustments', input_adjustments])
    if overlap_list:
        ba_opt.extend(['--overlap-list', overlap_list])
    if lon_lat_limit:
        ba_opt.extend(['--lon-lat-limit',str(lon_lat_limit[0]),str(lon_lat_limit[1]),str(lon_lat_limit[2]),str(lon_lat_limit[3])])
    if elevation_limit:
        ba_opt.extend(['--elevation-limit',str(elevation_limit[0]),str(elevation_limit[1])])
    return ba_opt

def mapproject(img,outfn,session='rpc',dem='WGS84',tr=None,t_srs='EPSG:4326',cam=None,ba_prefix=None):
    """
    orthorectify input image over a given DEM using ASP's mapproject program.
    See mapproject documentation here: https://stereopipeline.readthedocs.io/en/latest/tools/mapproject.html
    inputs:
    img: Path to Raw image to be orthorectified
    outfn: Path to output orthorectified image
    session: type of input camera model (default: rpc)
    dem: input DEM over which images will be draped (default: WGS84, orthorectify just over datum)
    tr: target resolution of orthorectified output image
    t_srs: target projection of orthorectified output image (default: EPSG:4326)
    cam: if pinhole session, this will be the path to pinhole camera model
    ba_prefix: Bundle adjustment output for RPC camera.
    
    returns: Nothing, orthorectifies input image
    """
    map_opt = []
    map_opt.extend(['-t',session])
    map_opt.extend(['--t_srs',t_srs])
    if ba_prefix:
        map_opt.extend(['--bundle-adjust-prefix',ba_prefix])
    if tr:
        map_opt.extend(['--tr',tr])
    map_args = [dem,img,outfn]
        if cam:
    map_args = [dem,img,cam,outfn]
    run_cmd('mapproject',map_opt+map_args,msg='Running mapproject for {}'.format(img))

def dem_mosaic(img_list,outfn,tr=None,tsrs=None,stats=None):
    """
    mosaic  input image list using ASP's dem_mosaic program.
    See dem_mosaic documentation here: https://stereopipeline.readthedocs.io/en/latest/tools/dem_mosaic.html
    inputs:
    img_list: List of input images to be mosaiced
    outfn: Path to output mosaiced image
    tr: target resolution of orthorectified output image
    t_srs: target projection of orthorectified output image (default: EPSG:4326)
    stats: metric to use for mosaicing

    returns: Nothing, writes out a mosaiced image for the list of input images
    """

	dem_mosaic_opt = []
	dem_mosaic_opt.extend(['-o',outfn])
	if stats:
		dem_mosaic_opt.extend(['--{}'.format(stats)])
	if (tr is not None) & (ast.literal_eval(tr) is not None):
		dem_mosaic_opt.extend(['--tr', str(tr)])
	print(type(tr))
	if tsrs:
		dem_mosaic_opt.extend(['--t_srs', tsrs])
	dem_mosaic_args = img_list
	run_cmd('dem_mosaic',dem_mosaic_args+dem_mosaic_opt,msg='Generating compistes for {} with stats {}'.format(outfn,stats))

def get_stereo_opts(session='rpc',threads=4,ba_prefix=None,align='Affineepipolar',xcorr=2,std_mask=0.5,std_kernel=-1,lv=5,corr_kernel=[21,21],rfne_kernel=[35,35],stereo_mode=0,spm=1,cost_mode=2,corr_tile_size=1024,mvs=False):
    """
    prepares stereo cmd for ASP
    See ASP's stereo documentation here: https://stereopipeline.readthedocs.io/en/latest/correlation.html
    inputs:
    session: camera model with which stereo steps (preprocessing, triangulation will be performed (default: rpc)
    threads: number of threads to use for each stereo job (default: 4)
    ba_prefix: if rpc, read adjustment to rpc files from this path
    align: alignment method to be used befor correlation (default: Affineepipolar). Note will only be relevant if non-ortho images are used for correlation
    xcorr: Whether to perform cross-check (forward+backward search during stereo), default is 2, so check for disparity first from left to right and then from right to left
    std_mask: this does not perform what is expected, so omitted now
    std_kernel: omitted for now
    lv: number of pyramidal overview levels for stereo correlation, defualt is 5 levels
    corr_kernel: tempelate window size for stereo correlation (default is [21,21])
    rfne_kernel: tempelate window size for sub-pixel optimization (default is [35,35])
    stereo_mode: 0 for block matching, 1 for SGM, 2 for MGM (default is 0)
    spm: subpixel mode, 0 for parabolic localisation, 1 for adaptavie affine and 2 for simple affine (default is 1)
    cost_mode: Cost function to determine match scores, depends on stereo_mode, defualt is 2 (Normalised cross correlation) for block matching
    corr_tile_size: tile sizes for stereo correlation, default is ASP default size of 1024, for SGM/MGM this is changed to 6400 for skysat
    mvs: prepare arguments for experimental multiview video stereo

    returns: a set of stereo arguments as list to be run using subprocess command.
    """
    stereo_opt = []
    # session_args
    stereo_opt.extend(['-t', session])
    stereo_opt.extend(['--threads', str(threads)])
    if ba_prefix:
        stereo_opt.extend(['--bundle-adjust-prefix', ba_prefix])
    # stereo is a python wrapper for 3/4 stages
    # stereo_pprc args : This is for preprocessing (adjusting image dynamic
    # range, alignment using ip matches etc)
    stereo_opt.extend(['--individually-normalize'])
    stereo_opt.extend(['--alignment-method', align])
    stereo_opt.extend(['--ip-per-tile', '8000'])
    stereo_opt.extend(['--ip-num-ransac-iterations','2000'])
    #stereo_opt.extend(['--ip-detect-method', '1'])
    stereo_opt.extend(['--force-reuse-match-files'])
    stereo_opt.extend(['--skip-rough-homography'])
    # mask out completely feature less area using a std filter, to avoid gross MGM errors
    # this is experimental and needs more testing
    stereo_opt.extend(['--stddev-mask-thresh', str(std_mask)])
    stereo_opt.extend(['--stddev-mask-kernel', str(std_kernel)])
    # stereo_corr_args:
    # parallel stereo is generally not required with input SkySat imagery
    # So all the mgm/sgm calls are done without it.
    stereo_opt.extend(['--stereo-algorithm', str(stereo_mode)])
    # the kernel size would depend on the algorithm
    stereo_opt.extend(['--corr-kernel', str(corr_kernel[0]), str(corr_kernel[1])])
    stereo_opt.extend(['--corr-tile-size', str(corr_tile_size)])
    stereo_opt.extend(['--cost-mode', str(cost_mode)])
    stereo_opt.extend(['--corr-max-levels', str(lv)])
    # stereo_rfne_args:
    stereo_opt.extend(['--subpixel-mode', str(spm)])
    stereo_opt.extend(['--subpixel-kernel', str(rfne_kernel[0]), str(rfne_kernel[1])])
    stereo_opt.extend(['--xcorr-threshold', str(xcorr)])
    # stereo_fltr_args:
    """
    Nothing for now,going with default can include somethings like:
    - median-filter-size, --texture-smooth-size (I guess these are set to some defualts for sgm/mgm ?)
    """
    # stereo_tri_args:
    disp_trip = 10000
    if not mvs:
        stereo_opt.extend(['--num-matches-from-disp-triplets', str(disp_trip)])
        stereo_opt.extend(['--unalign-disparity'])
    return stereo_opt

def convergence_angle(az1, el1, az2, el2):
    """
    function to calculate convergence angle between two satellites 
    # Credits: from David's dgtools
    inputs:
    az1,el1: azimuth and elevation as arrays/list/single_number (in degrees for satellite 1)
    az2,el2: azimuth and elevation as arrays/list/single_number (in degrees for satellite 2)

    returns: convergence angle in degrees
    """
    conv_ang = np.rad2deg(np.arccos(np.sin(np.deg2rad(el1)) * np.sin(np.deg2rad(el2)) + np.cos(np.deg2rad(el1)) * np.cos(np.deg2rad(el2)) * np.cos(np.deg2rad(az1 - az2))))
    return conv_ang

def get_pc_align_opts(outprefix, max_displacement=100, align='point-to-plane', source=True, trans_only=False):
    """
    prepares ASP pc_align ICP cmd
    See pc_align documentation here: https://stereopipeline.readthedocs.io/en/latest/tools/pc_align.html
    inputs:
    outprefix: prefix with which pc_align results will be saved (can be a path, general convention for repo is some path with run prefix, eg., aligned_to/run)
    max_displacement: Maximum expected displacement between input DEMs, useful for culling outliers before solving for shifts, default: 100 m
    align: ICP's alignment algorithm to use. default: point-to-plane 
    source: if True, this tells the the algorithm to align the source to reference DEM/PC. If false, this tells the program to align reference to source and save inverse transformation. default: True
    trans_only: if True, this instructs the program to compute translation only when point cloud optimization. Default: False
    
    returns: list of pc_align parameteres
    """
   
    pc_align_opts = []
    pc_align_opts.extend(['--alignment-method', align])
    pc_align_opts.extend(['--max-displacement', str(max_displacement)])
    pc_align_opts.extend(['--highest-accuracy'])
    if source:
        pc_align_opts.extend(['--save-transformed-source-points'])
    else:
        pc_align_opts.extend(['--save-inv-transformed-reference-points'])
    if trans_only:
        pc_align_opts.extend(['--compute-translation-only'])
    pc_align_opts.extend(['-o', outprefix])
    return pc_align_opts
def get_point2dem_opts(tr, tsrs):
    """
    prepares argument for ASP's point cloud gridding algorithm (point2dem) cmd
    inputs:
    tr: target resolution of output DEM
    tsrs: projection of output DEM 
    
    returns: list of point2dem parameteres
    """

    point2dem_opts = []
    point2dem_opts.extend(['--tr', str(tr)])
    point2dem_opts.extend(['--t_srs', tsrs])
    point2dem_opts.extend(['--errorimage'])
    return point2dem_opts

def get_total_shift(pc_align_log):
    """
    returns total shift by pc_align 
    input: pc_align_log: log file written by ASP pc_align run
    returns: float value of applied displacement
    """
    with open(pc_align_log, 'r') as f:
        content = f.readlines()
    substring = 'Maximum displacement of points between the source cloud with any initial transform applied to it and the source cloud after alignment to the reference'
    max_alignment_string = [i for i in content if substring in i]
    total_shift = np.float(max_alignment_string[0].split(':',15)[-1].split('m')[0])
    return total_shift

def dem_align(ref_dem, source_dem, max_displacement, outprefix, align, trans_only=False):
    """
    This function implements the full DEM alignment workflow using ASP's pc_align and point2dem programs
    See relevent doumentation here:  https://stereopipeline.readthedocs.io/en/latest/tools/pc_align.html 
    inputs:
    ref_dem: reference DEM for alignment
    source_dem: source DEM to be aligned
    max_displacement: Maximum expected displacement between input DEMs, useful for culling outliers before solving for shifts, default: 100 m
    outprefix: prefix with which pc_align results will be saved (can be a path, general convention for repo is some path with run prefix, eg., aligned_to/run)
    max_displacement: Maximum expected displacement between input DEMs, useful for culling outliers before solving for shifts, default: 100 m
    align: ICP's alignment algorithm to use. default: point-to-plane
    trans_only: if True, this instructs the program to compute translation only when point cloud optimization. Default: False
    
    output:
    nothing, aligns and regrids aligned pointcloud
    """
    # this block checks wheter reference DEM is finer resolution or source DEM
    # if reference DEM is finer resolution, then source is aligned to reference
    # if source DEM is finer, then reference is aligned to source and source is corrected via the inverse transformation matrix of source to reference alignment.
    source_ds = iolib.fn_getds(source_dem)
    ref_ds = iolib.fn_getds(ref_dem)
    source_res = geolib.get_res(source_ds, square=True)[0]
    ref_res = geolib.get_res(ref_ds, square=True)[0]
    tr = source_res
    tsrs = source_ds.GetProjection()
    print(type(tsrs))
    if ref_res <= source_res:
        source = True
        pc_align_args = [ref_dem, source_dem]
        pc_id = 'trans_source.tif'
    else:
        source = False
        pc_align_args = [source_dem, ref_dem]
        pc_id = 'trans_reference.tif'
    print(f"Aligning clouds via the {align} method")
    
    pc_align_opts = get_pc_align_opts(outprefix,max_displacement,align=align,source=source,trans_only=trans_only)
    run_cmd('pc_align', pc_align_opts + pc_align_args)
    # this try, except block checks for 2 things.
    #- Did the transformed point-cloud got produced ?
    #- was the maximum displacement greater than twice the max_displacement specified by the user ? 
      # 2nd condition is implemented for tricky alignement of individual triplet DEMs to reference, as some small DEMs might be awkardly displaced to > 1000 m.
    # if the above conditions are not met, then gridding of the transformed point-cloud into final DEM will not occur.
    try:
        pc = glob.glob(outprefix + '*'+pc_id)[0]
        pc_log = sorted(glob.glob(outprefix+'*'+'log-pc_align*.txt'))[-1] # this will hopefully pull out latest transformation log
        max_disp = get_total_shift(pc_log)
        print(f"Maximum displacement is {max_disp}")
        if max_disp <= 2*max_displacement:
            grid = True
        else:
           grid = False
    except:
        grid = False
        pass
    if grid == True:
        point2dem_opts = get_point2dem_opts(tr, tsrs)
        point2dem_args = [pc]
        print(f"Saving aligned reference DEM at {os.path.splitext(pc)[0]}-DEM.tif")
        run_cmd('point2dem', point2dem_opts + point2dem_args)
    elif grid == False:
        print("aligned cloud not produced or the total shift applied to cloud is greater than 2 times the max_displacement specified, gridding abandoned")

def get_cam2rpc_opts(t='pinhole', dem=None, gsd=None, num_samples=50):
    """
    generates cmd for ASP cam2rpc
    This generates rpc camera models from the optimized frame camera models
    See documentation here: https://stereopipeline.readthedocs.io/en/latest/tools/cam2rpc.html
    inputs:
    t: session, or for here, type of input camera, default: pinhole 
    dem: DEM which will be used for calculating RPC polynomials
    gsd: Expected ground-samplind distance
    num_samples: Sampling for RPC approximation calculation (default=50)
    returns: generates a list of arguments for cam2rpc call.
    """
 
    cam2rpc_opts = []
    cam2rpc_opts.extend(['--dem-file', dem])
    dem_ds = iolib.fn_getds(dem)
    dem_proj = dem_ds.GetProjection()
    dem = iolib.ds_getma(dem_ds)
    min_height, max_height = np.percentile(dem.compressed(), (0.01, 0.99))
    tsrs = epsg2geolib(4326)
    xmin, ymin, xmax, ymax = geolib.ds_extent(ds, tsrs)
    cam2rpc_opts.extend(['--height-range', str(min_height), str(max_height)])
    cam2rpc_opts.extend(['--lon-lat-range', str(xmin),
                        str(ymin), str(xmax), str(ymax)])
    if gsd:
        cam2rpc_opts.extend(['--gsd', str(gsd)])
    cam2rpc_opts.extend(['--session', t])
    cam2rpc_opts.extend(['--num-samples', str(num_samples)])
    return cam2rpc_opts
    
