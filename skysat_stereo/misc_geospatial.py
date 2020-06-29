#! /usr/bin/env python

import numpy as np
import pandas as pd
import geopandas as gpd

def shp_merger(shplist):
    """
    merge multiple geopandas shapefiles into 1 multi-row shapefile
    Parameters
    ----------
    shplist: list
        list of geopandas shapefiles
    Returns
    ----------
    gpd_merged: geopandas geodataframe
        merged multirow shapefile
    """
    #Taken from here: "https://stackoverflow.com/questions/48874113/concat-multiple-shapefiles-via-geopandas"
    gpd_merged = pd.concat([shp for shp in shplist]).pipe(gpd.GeoDataFrame)
    return gpd_merged

