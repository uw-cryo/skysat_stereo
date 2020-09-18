#! /usr/bin/env python
import os,sys,glob
import pandas as pd
import numpy as np
from pygeotools.lib import iolib,geolib,malib,warplib
import gdal
import matplotlib.pyplot as plt
import argparse
import geopandas as gpd
from shapely.geometry import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logging
import json
import subprocess

def gdaldem(ds,producttype='slope',returnma=True):
    """
    perform gdaldem operations such as slope, hillshade etc via the python api
    Parameters
    -----------
    ds: gdal dataset
        DEM dataset for which derived products are to be computed
    producttype: str
        operation to perform (e.g., slope, aspect, hillshade)
    returnma: bool
        return the product as masked array if true, or a dataset if false
    Returns
    -----------
    out: masked array or dataset
        output product in form of masked array or dataset (see params)
    """
    dem_p_ds = gdal.DEMProcessing('',ds,producttype,format='MEM')
    ma = iolib.ds_getma(dem_p_ds)
    if returnma:
        out = ma
    else:
        out = dem_p_ds
    return out

def cummulative_profile(xma,yma,xbin_width,limit_x_perc = (1,99)):
    """
    compute binned statistics for independent variable with respect to dependendent variable
    Parameters
    -----------
    xma: masked array
        independent variable (like slope)
    yma: masked array
        dependent variable (like elevation difference)
    xbin_width: int
        bin_width for independent variable
    limit_x_perc: tuple
        limit binning of independent variable to the given percentile (default: 1 to 99 %)
    Returns
    -----------
    x_bins: np.array
        bin locations
    y_mean: np.array
        binned mean value for dependent variable
    y_meadian: np.array
        binned median value for dependent variable
    y_std: np.array
        binned standard deviation value for dependent varuiable
    y_perc: np.array
        binned percentage of variables within the bin
    """
    # xclim get rids of outliers in the independent variable
    # we only look at the 1 to 99 percentile values by default
    xclim = malib.calcperc(xma,limit_x_perc)
    # this step computes common mask where pixels of both x and y variables are valid
    xma_lim = np.ma.masked_outside(xma,xclim[0],xclim[1])
    cmask = malib.common_mask([xma_lim,yma])
    # the common mask is used to flatten the required points in a 1-D array
    xma_c = np.ma.compressed(np.ma.array(xma_lim,mask=cmask))
    yma_c = np.ma.compressed(np.ma.array(yma,mask=cmask))
    # we then use pandas groupby to quickly compute binned statistics
    df = pd.DataFrame({'x':xma_c,'y':yma_c})
    df['x_rounded']=(df['x']+(xbin_width-1))//(xbin_width)*xbin_width
    grouped=df.groupby('x_rounded')
    df2=grouped['y'].agg([np.mean,np.count_nonzero,np.median,np.std])
    df2.reset_index(inplace=True)
    # variables are returned as numpy array
    x_bins = df2['x_rounded'].values
    y_mean = df2['mean'].values
    y_median = df2['median'].values
    y_std = df2['std'].values
    y_perc = (df2['count_nonzero'].values/np.sum(
        df2['count_nonzero'].values))*100
    return x_bins,y_mean,y_median,y_std,y_perc

def getparser():
    parser = argparse.ArgumentParser(description='Script for comparing 2 DEMs')
    parser.add_argument('-refdem',default=None,type=str,
            help='path to refernece DEM file')
    parser.add_argument('-srcdem',default=None,type=str,
            help='path to source DEM file')
    binary_choices = [1,0]
    parser.add_argument('-local_ortho',default=1,type=int,
            choices=binary_choices,
            help='perform comparison on local ortho grid if 1, else native grid if 0 (default: %(default)s)')
    res_choices = ['mean','min','max']
    parser.add_argument('-comparison_res',default='min',choices=res_choices,
            help='common resolution at which to perform comparison, (default: %(default)s)')
    parser.add_argument('-elev_bin_width',default=10,type=int,
            help='elevation bin width for computing binned elevation difference statistics (default: %(default)s m)')
    parser.add_argument('-slope_bin_width',default=2,type=int,
            help='slope bin width for computing binned elevation difference statistics (default: %(default)s degrees)')
    parser.add_argument('-coreg',default=1,type=int,choices=binary_choices,
            help='Attempt co-registeration and redo stats calculation (default:%(default)s)')
    parser.add_argument('-outfol',type=str,required=False,
            help='path to outfolder to store results')
    return parser

def main():
    parser = getparser()
    args = parser.parse_args()
    refdem = args.refdem
    srcdem = args.srcdem
    outfolder = '{}__{}_comparison_stats'.format(os.path.splitext(
        os.path.basename(refdem))[0],os.path.splitext(os.path.basename(
            srcdem))[0])
    header_str = '{}__{}'.format(os.path.splitext(os.path.basename(refdem))[0],
            os.path.splitext(os.path.basename(srcdem))[0])
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    if args.local_ortho == 1:
        temp_ds = warplib.memwarp_multi_fn([refdem,srcdem])[0]
        bbox = geolib.ds_extent(temp_ds)
        geo_crs = temp_ds.GetProjection()
        print ('Bounding box lon_lat is{}'.format(bbox))
        bound_poly = Polygon([[bbox[0],bbox[3]],[bbox[2],bbox[3]],[bbox[2],bbox[1]],[bbox[0],bbox[1]]])
        bound_shp = gpd.GeoDataFrame(index=[0],geometry=[bound_poly],crs=geo_crs)
        bound_centroid = bound_shp.centroid
        cx = bound_centroid.x.values[0]
        cy = bound_centroid.y.values[0]
        pad = np.ptp([bbox[3],bbox[1]])/6.0
        lat_1 = bbox[1]+pad
        lat_2 = bbox[3]-pad
        local_ortho = "+proj=ortho +lat_1={} +lat_2={} +lat_0={} +lon_0={} +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs".format(lat_1,lat_2,cy,cx)
        logging.info('Local Ortho projection is {}'.format(local_ortho))
        t_srs = local_ortho
    else:
        t_srs = 'first'
    # this step performs the desired warping operation
    ds_list = warplib.memwarp_multi_fn([refdem,srcdem],res=args.comparison_res,
            t_srs=t_srs)
    refma = iolib.ds_getma(ds_list[0])
    srcma = iolib.ds_getma(ds_list[1])
    init_diff = refma - srcma
    init_stats = malib.get_stats_dict(init_diff)
    print("Original descriptive statistics {}".format(init_stats))
    init_diff_json_fn = os.path.join(outfolder,
        '{}_precoreg_descriptive_stats.json'.format(header_str))
    init_diff_json = json.dumps(init_stats)

    with open(init_diff_json_fn,'w') as f:
        f.write(init_diff_json)
    logging.info("Saved initial stats at {}".format(init_diff_json))
    refslope = gdaldem(ds_list[0])
    # stats for elevation difference vs reference DEM elevation
    elev_bin,diff_mean,diff_median,diff_std,diff_perc = cummulative_profile(
            refma,init_diff,args.elev_bin_width)
    # stats for elevation difference vs reference DEM slope
    slope_bin,diff_mean_s,diff_median_s,diff_std_s,diff_perc_s = cummulative_profile(
            refslope,init_diff,args.slope_bin_width)
    f,ax = plt.subplots(1,2,figsize=(10,4))
    im = ax[0].scatter(elev_bin,diff_mean,c=diff_perc,cmap='inferno')
    ax[0].set_xlabel('Elevation (m)')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='2.5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical',label='pixel count percentage')
    im2 = ax[1].scatter(slope_bin,diff_mean_s,c=diff_perc_s,cmap='inferno')
    ax[1].set_xlabel('Slope (degrees)')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='2.5%', pad=0.05)
    plt.colorbar(im2, cax=cax, orientation='vertical',label='pixel count percentage')

    for axa in ax.ravel():
        axa.axhline(y=0,c='k')
        axa.set_ylabel('Elevation Difference (m)')
    plt.tight_layout()
    precoreg_plot = os.path.join(outfolder,header_str+'_precoreg_binned_plot.png')
    f.savefig(precoreg_plot,dpi=300,bbox_inches='tight', pad_inches=0.1)
    logging.info("Saved binned plot at {}".format(precoreg_plot))
    if args.coreg == 1:
        logging.info("will attempt coregisteration")
        if args.local_ortho == 1:
            ref_local_ortho = os.path.splitext(refdem)[0]+'_local_ortho.tif'
            src_local_ortho = os.path.splitext(srcdem)[0]+'_local_ortho.tif'
            # coregisteration works best at mean resolution
            # we will rewarp if the initial args.res was not mean
            if args.comparison_res != 'mean':
                ds_list = warplib.memwarp_multi_fn([refdem,srcdem],res='mean',
                        t_srs = t_srs)
                refma = iolib.ds_getma(ds_list[0])
                srcma = iolib.ds_getma(ds_list[1])
            iolib.writeGTiff(refma,ref_local_ortho,ds_list[0])
            iolib.writeGTiff(srcma,src_local_ortho,ds_list[1])
            coreg_ref = ref_local_ortho
            src_ref = src_local_ortho
        else:
            coreg_ref = refdem
            src_ref = srcdem
        demcoreg_dir = os.path.join(outfolder,'coreg_results')
        align_opts = ['-mode', 'nuth','-max_iter','12','-max_offset','400',
                '-outdir',demcoreg_dir]
        align_args = [coreg_ref,src_ref]
        align_cmd = ['dem_align.py']+align_opts+align_args
        subprocess.call(align_cmd)
        #ah final round of warping and stats calculation
        try:
            srcdem_align = glob.glob(os.path.join(demcoreg_dir,'*align.tif'))[0]
            logging.info("Attempting stats calculation for aligned DEM {}".format(
                srcdem_align))
            ds_list = warplib.memwarp_multi_fn([args.refdem,srcdem_align],
                    res=args.comparison_res,t_srs = t_srs)
            refma = iolib.ds_getma(ds_list[0])
            srcma = iolib.ds_getma(ds_list[1])
            # this is creepy, but I am recycling variable names to save on memory
            init_diff = refma - srcma
            init_stats = malib.get_stats_dict(init_diff)
            print("Final descriptive statistics {}".format(init_stats))
            init_diff_json_fn = os.path.join(outfolder,
                '{}_postcoreg_descriptive_stats.json'.format(header_str))
            init_diff_json = json.dumps(init_stats)

            with open(init_diff_json_fn,'w') as f:
                f.write(init_diff_json)
            logging.info("Saved final stats at {}".format(init_diff_json))
            refslope = gdaldem(ds_list[0])
            # stats for elevation difference vs reference DEM elevation
            elev_bin,diff_mean,diff_median,diff_std,diff_perc = cummulative_profile(
                refma,init_diff,args.elev_bin_width)
            # stats for elevation difference vs reference DEM slope
            slope_bin,diff_mean_s,diff_median_s,diff_std_s,diff_perc_s = cummulative_profile(
                refslope,init_diff,args.slope_bin_width)
            f,ax = plt.subplots(1,2,figsize=(10,4))
            im = ax[0].scatter(elev_bin,diff_mean,c=diff_perc,cmap='inferno')
            ax[0].set_xlabel('Elevation (m)')
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes('right', size='2.5%', pad=0.05)
            plt.colorbar(im, cax=cax, orientation='vertical',label='pixel count percentage')
            im2 = ax[1].scatter(slope_bin,diff_mean_s,c=diff_perc_s,cmap='inferno')
            ax[1].set_xlabel('Slope (degrees)')
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes('right', size='2.5%', pad=0.05)
            plt.colorbar(im2, cax=cax, orientation='vertical',label='pixel count percentage')

            for axa in ax.ravel():
                axa.axhline(y=0,c='k')
                axa.set_ylabel('Elevation Difference (m)')
            plt.tight_layout()
            precoreg_plot = os.path.join(outfolder,header_str+'_postcoreg_binned_plot.png')
            f.savefig(precoreg_plot,dpi=300,bbox_inches='tight', pad_inches=0.1)
        except:
            logging.info("Failed to compute post coreg stats, see corresponding job log")
        logging.info("Script is complete !")

if __name__=="__main__":
    main()


