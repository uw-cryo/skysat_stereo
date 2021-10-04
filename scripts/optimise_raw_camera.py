#! /usr/bin/env python

import os,sys,glob
import pandas as pd
import geopandas as gpd
import numpy as np
from skysat_stereo import asp_utils as asp
from scipy.optimize import least_squares
from pyquaternion import Quaternion
import argparse
from pygeotools.lib import iolib,geolib
import logging
from pyproj import Transformer

def cam_solve(q1,q2,q3,q4,CX,CY,CZ,cu,cv,fu,fv,pitch,X,Y,Z):
    """
    Forward Solver for simple pinhole camera model
    Parameters
    -----------
    q1,q2,q3,q4: float
        quaternions
    CX,CY,CZ: float
        camera center position in ECEF system
    cx,cy: float
        position of optical center in pixel units
    fu,fv: float
        focal length in pixel units
    pitch: float
        camera pixel pitch in floats
    X,Y,Z: float
        3D points in ECEF coordinate system
    Returns
    ------------
    px,py: float
        points in image plane coordinates
    """
    #print(q1,q2,q3,q4)
    quaternion = Quaternion(q1,q2,q3,q4)
    rot_mat = quaternion.rotation_matrix
    rot_mat_inv = np.linalg.inv(rot_mat)
    world_points = np.stack((X,Y,Z),axis=0)
    cam_cen = np.array([CX,CY,CZ])
    cam_cord = np.matmul(rot_mat_inv,world_points) - np.reshape(
            np.matmul(rot_mat_inv,cam_cen),(3,1))
    px = (fu*cam_cord[0])/(pitch*cam_cord[2]) + (cu/pitch)
    py = (fv*cam_cord[1])/(pitch*cam_cord[2]) + (cv/pitch)
    return px,py

def reprojection_error(tpl,CX,CY,CZ,cu,cv,fu,fv,pitch,X,Y,Z,im_x,im_y):
    """
    tpl: tuple
        tuple containing four quaternion (this will be optimized)
    CX,CY,CZ: float
        camera center position in ECEF system
    cx,cy: float
        position of optical center in pixel units
    fu,fv: float
        focal length in pixel units
    pitch: float
        camera pixel pitch in floats
    X,Y,Z: float
        3D points in ECEF coordinate system from GCP
    im_x,im_y: float
        measured image pixel positions from GCP
    Returns
    ----------
    res: float
        residual between estimated and actual image coordinate
    """
    #print(tpl)
    q1,q2,q3,q4 = tpl
    px,py = cam_solve(q1,q2,q3,q4,CX,CY,CZ,cu,cv,fu,fv,pitch,X,Y,Z)
    #res = (im_x-px)**2 + (im_y-py)**2
    res = np.array(list(im_x-px) + list(im_y-py)).ravel()
    return res

def optimiser_quaternion(q1,q2,q3,q4,CX,CY,CZ,cu,cv,fu,fv,pitch,X,Y,Z,im_x,im_y):
    """
    q1,q2,q3,q4: float
        initial guess four quaternion (this will be optimized)
    CX,CY,CZ: float
        camera center position in ECEF system
    cx,cy: float
        position of optical center in pixel units
    fu,fv: float
        focal length in pixel units
    pitch: float
        camera pixel pitch in floats
    X,Y,Z: float
        3D points in ECEF coordinate system from GCP
    im_x,im_y: float
        measured image pixel positions from GCP
    Returns
    ----------
    q1,q2,q3,q4: float
        optimised_quaternion
    """
    print(q1)
    tpl_init = (q1,q2,q3,q4)
    error_func = lambda tpl: reprojection_error(tpl,CX,CY,CZ,cu,cv,fu,fv,pitch,X,Y,Z,im_x,im_y)
    print("Initial reprojection error {} RMSE px".format(np.sqrt(np.sum(error_func(tpl_init)**2))))
    result = least_squares(error_func,tpl_init[:],
                           bounds=((-1,-1,-1,-1),(1,1,1,1)),method='dogbox')
    #bounds are specified for the quaternions to normalise them
    Q1,Q2,Q3,Q4 = result.x
    print("Final reprojection error {} RMSE px".format(np.sqrt(np.sum(error_func((Q1,Q2,Q3,Q4))**2))))
    return(Q1,Q2,Q3,Q4)

def getparser():
    parser = argparse.ArgumentParser(
            description="optimise the raw camera model from cam_gen to confirm to satellite telemetry information")
    parser.add_argument('-camera_folder',required=True,
            help='Folder containing cam_gen derived frame camera model')
    parser.add_argument('-gcp_folder',required=False,default=None,
            help='Folder containing corner gcps; if none, program looks for gcps in the camera folder')
    parser.add_argument('-frame_index',required=True,
            help='path to frame_index.csv file')
    parser.add_argument('-outfol',required=True,
            help='path to folder to save optimised camera model')
    return parser

def main():
    parser = getparser()
    args = parser.parse_args()
    f_index = args.frame_index
    if os.path.splitext(f_index)[1] == '.csv':
        frame_index = pd.read_csv(f_index)
    else:
        frame_index = pd.read_pickle(f_index)
    logging.info("sample fn {}".format(glob.glob(os.path.join(args.camera_folder,
        '*{}*.tsai'.format(frame_index['name'].values[0])))))
    
    # cam_list = [glob.glob(os.path.join(args.camera_folder,'*{}*.tsai'.format(os.path.basename(frame))))[0] for frame in frame_index['name'].values]
    cam_list = []
    for frame in frame_index['name'].values:
        try:
            cam_list.append(glob.glob(os.path.join(args.camera_folder,'*{}*.tsai'.format(os.path.basename(frame))))[0])
        except:
            continue

    if not args.gcp_folder:
        gcp_folder = args.camera_folder
    else:
        gcp_folder = args.gcp_folder
    if not os.path.exists(args.outfol):
        os.makedirs(args.outfol)
    gcp_list = [glob.glob(os.path.join(gcp_folder,'*{}*.gcp'.format(os.path.basename(frame))))[0] for frame in frame_index['name'].values]
    CX,CY,CZ = [frame_index.x_sat_ecef_km.values*1000,frame_index.y_sat_ecef_km*1000,frame_index.z_sat_ecef_km*1000]
    rotation_matrices = [Quaternion(matrix=(np.reshape(asp.read_tsai_dict(x)['rotation_matrix'],(3,3)))) for x in cam_list]
    fu,fv = asp.read_tsai_dict(cam_list[0])['focal_length']
    cu,cv = asp.read_tsai_dict(cam_list[0])['optical_center']
    pitch = asp.read_tsai_dict(cam_list[0])['pitch']
    q1 = [x[0] for x in rotation_matrices]
    q2 = [x[1] for x in rotation_matrices]
    q3 = [x[2] for x in rotation_matrices]
    q4 = [x[3] for x in rotation_matrices]
    for idx,row in frame_index.iterrows():
        identifier = os.path.basename(row['name'])
        gcp = pd.read_csv(glob.glob(os.path.join(gcp_folder,'*{}*.gcp'.format(identifier)))[0],header=None,sep=' ')
        im_x,im_y = [gcp[8].values,gcp[9].values]
        lon,lat,ht = [gcp[2].values,gcp[1].values,gcp[3].values]
        ecef_proj = 'EPSG:4978'
        geo_proj = 'EPSG:4326'
        wgs2ecef = Transformer.from_crs(geo_proj,ecef_proj)
        X,Y,Z = wgs2ecef.transform(lat,lon,ht)
        CX_idx,CY_idx,CZ_idx = [CX[idx],CY[idx],CZ[idx]]
        q1_idx,q2_idx,q3_idx,q4_idx = [q1[idx],q2[idx],q3[idx],q4[idx]]
        #tpl_int = (q1_idx,q2_idx,q3_idx,q4_idx)
        print(idx)
        Q1,Q2,Q3,Q4 = optimiser_quaternion(q1_idx,q2_idx,q3_idx,q4_idx,CX_idx,
                CY_idx,CZ_idx,cu,cv,fu,fv,pitch,X,Y,Z,im_x,im_y)
        rot_mat = Quaternion([Q1,Q2,Q3,Q4]).rotation_matrix
        out_cam = os.path.join(args.outfol,'{}_scipy.tsai'.format(identifier))
        asp.make_tsai(out_cam,cu,cv,fu,fv,rot_mat,[CX_idx,CY_idx,CZ_idx],pitch)
    logging.info("Successfully created optimised camera models at {}".format(
        args.outfol))
if __name__=='__main__':
    main()
