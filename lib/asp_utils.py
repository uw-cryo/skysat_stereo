#! /usr/bin/env python
import numpy as np
from pygeotools.lib import geolib
import os,sys,glob,shutil
import geopandas as gpd
from pyproj import Proj, transform

#TODO: 
# mapproject and dem_mosaic

def read_tsai_dict(tsai):
    """
    read tsai pinhole model from asp and return a dictionary containing the parameters
    TODO: support distortion model
    """
    camera = os.path.basename(tsai)
    with open(tsai,'r') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    fu = np.float(content[2].split(' = ',4)[1])
    fv = np.float(content[3].split(' = ',4)[1])
    cu = np.float(content[4].split(' = ',4)[1])
    cv = np.float(content[5].split(' = ',4)[1])
    cam = content[9].split(' = ',10)[1].split(' ')
    cam_cen = [np.float(x) for x in cam]
    rot = content[10].split(' = ',10)[1].split(' ')
    rot_mat = [np.float(x) for x in rot]
    pitch = np.float(content[11].split(' = ',10)[1])
    cam_cen_lat_lon = geolib.ecef2ll(cam_cen[0],cam_cen[1],cam_cen[2])
    tsai_dict = {'camera':camera,'focal_length':(fu,fv),'optical_center':(cu,cv),'cam_cen_ecef':cam_cen,'cam_cen_wgs':cam_cen_lat_lon,'rotation_matrix':rot_mat,'pitch':pitch}
    return tsai_dict

def make_tsai(outfn,cu,cv,fu,fv,rot_mat,C,pitch):
    """
    write out pinhole model with given parameters
    """
    out_str = f'VERSION_4\nPINHOLE\nfu = {fu}\nfv = {fv}\ncu = {cu}\ncv = {cv}\nu_direction = 1 0 0\nv_direction = 0 1 0\nw_direction = 0 0 1\nC = {C[0]} {C[1]} {C[2]}\nR = {rot_mat[0][0]} {rot_mat[0][1]} {rot_mat[0][2]} {rot_mat[1][0]} {rot_mat[1][1]} {rot_mat[1][2]} {rot_mat[2][0]} {rot_mat[2][1]} {rot_mat[2][2]}\npitch = {pitch}\nNULL'
    with open(outfn,'w') as f:
        f.write(out_str)

def cam_gen(img,fl,cx,cy,pitch,ht_datum,gcp_std,out_fn,out_gcp,datum,refdem,camera):
	cam_gen_opt = []
	cam_gen_opt.extend(['--focal-length',str(fl)])
	cam_gen_opt.extend(['--optical-center',str(cx),str(cy)])
	cam_gen_opt.extend(['--pixel-pitch',str(pitch)])
	cam_gen_opt.extend(['--height-above-datum',str(ht_datum)])
	cam_gen_opt.extend(['--gcp-std',str(gcp_std)])
	cam_gen_opt.extend(['-o',out_fn])
	cam_gen_opt.extend(['--gcp-file',out_gcp])
	cam_gen_opt.extend(['--datum',datum])
	cam_gen_opt.extend(['--reference-dem',refdem])
	cam_gen_opt.extend(['--input-camera',camera])
	cam_gen_opt.extend(['--refine-camera'])
	cam_gen_args = [img]
	run_cmd('cam_gen',cam_gen_args+cam_gen_opt,msg='Running camgen command for image {}'.format(os.path.basename(img)))

def get_ba_opts(ba_prefix, camera_weight=0, overlap_list=None, overlap_limit=None, initial_transform=None, input_adjustments=None, flavor='general_ba', session='nadirpinhole', gcp_transform=False,num_iterations=2000,lon_lat_lim=None,elevation_limit=None):
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

def mapproject(img,outfn,session='rpc',dem='WGS84',tr=None,t_srs='EPSG:4326',cam=None):
	map_opt = []
	map_opt.extend(['-t',session])
	map_opt.extend(['--t_srs',t_srs])
	if tr:
		map_opt.extend(['--tr',tr])
	map_args = [dem,img,outfn]
	if cam:
		map_args = [dem,img,cam,outfn]
	run_cmd('mapproject',map_opt+map_args,msg='Running mapproject for {}'.format(img))

def dem_mosaic(img_list,outfn,tr=None,tsrs=None,stats=None):
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

def get_stereo_opts(session='rpc',threads=4,ba_prefix=None,align='Affineepipolar',std_mask = 0.5, std_kernel = -1, lv = 5, corr_kernel = [21,21], rfne_kernel = [35,35], stereo_mode=0, spm = 1,cost_mode=2,corr_tile_size=1024):
    stereo_opt = []
    # session_args
    stereo_opt.extend(['-t',session])
    stereo_opt.extend(['--threads',str(threads)])
    if ba_prefix:
        stereo_opt.extend(['--bundle-adjust-prefix',ba_prefix])
    # stereo is a python wrapper for 3/4 stages
    # stereo_pprc args : This is for preprocessing (adjusting image dynamic range, alignment using ip matches etc)
    stereo_opt.extend(['--individually-normalize'])
    stereo_opt.extend(['--alignment-method',align])
    stereo_opt.extend(['--ip-per-tile','4000']) 
    stereo_opt.extend(['--ip-detect-method','1'])
    stereo_opt.extend(['--force-reuse-match-files'])
    stereo_opt.extend(['--skip-rough-homography'])
    # mask out completely feature less area using a std filter, to avoid gross MGM errors
    # this is experimental and needs more testing
    stereo_opt.extend(['--stddev-mask-thresh',str(std_mask)])
    stereo_opt.extend(['--stddev-mask-kernel',str(std_kernel)])
    # stereo_corr_args: 
    # parallel stereo is generally not required with input SkySat imagery
    # So all the mgm/sgm calls are done without it.
    stereo_opt.extend(['--stereo-algorithm',str(stereo_mode)])
    # the kernel size would depend on the algorithm
    stereo_opt.extend(['--corr-kernel',str(corr_kernel[0]),str(corr_kernel)[1]])
    stereo_opt.extend(['--corr-tile-size',str(corr_tile_size)])
    stereo_opt.extend(['--cost-mode',str(cost_mode)])
    stereo_opt.extend(['--corr-max-levels',str(lv)])
    # stereo_rfne_args:
    stereo_opt.extend(['--subpixel-mode',str(spm)])
    stereo_opt.extend(['--subpixel-kernel',str(rfne_kernel[0]),str(rfne_kernel[1])])
    # stereo_fltr_args:
    """
    Nothing for now, can include somethings like:
    - median-filter-size, --texture-smooth-size (I guess these are set to some defualts for sgm/mgm ?)
    """
    # stereo_tri_args:
    disp_trip = 10000
    stereo_opt.extend(['--num-matches-from-disp-triplets',str(disp_trip)])
    stereo_opt.extend(['--unalign-disparity'])
    return stereo_opt

def cam2rpc_opt(t='pinhole',gsd=None,num-samples=50):
    cam2rpc_opt = []
    cam2rpc_opt.extend(['--session',t])
    cam2rpc_opt.extend(['--num-samples',str(num_samples)])
    return cam2rpc_opt
