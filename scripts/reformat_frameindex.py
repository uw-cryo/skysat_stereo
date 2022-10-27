#! /usr/bin/python

import argparse
import os,sys,glob,shutil
import pandas as pd
from shapely import wkt
from shapely.geometry.polygon import orient

def getparser():
    parser = argparse.ArgumentParser(description='Light-weight script to reformat new versions of Planet provided frame_index file')
    parser.add_argument('-in_frameindex',type=str,help='path to original frame_index.csv')
    return parser

def _correct_geom(row):
    return wkt.loads(row['geom'])

def main():
    parser = getparser()
    args = parser.parse_args()
    original_frame_fn = args.in_frameindex
    frame_index = pd.read_csv(original_frame_fn)
    frame_index['geom'] = frame_index.apply(_correct_geom,axis=1)

    #orient the polygon geometry
    updated_geomlist_asp_convention = [orient(test_geom,-1) for test_geom in frame_index['geom'].values]

    # remove the space between POLYGON and ((# 
    updated_geomlist_asp_convention = [f"POLYGON(({str(test_geom).split(' ((')[1]}" for test_geom in updated_geomlist_asp_convention]

    # remove the repeated last coordinate
    updated_geomlist_asp_convention = [','.join(test_geom.split(',')[:-1])+'))' for test_geom in updated_geomlist_asp_convention]

    # update geometry column
    frame_index['geom'] = updated_geomlist_asp_convention

    #writeout
    outfn = os.path.splitext(original_frame_fn)[0]+"_asp_convention.csv"
    frame_index.to_csv(outfn,index=False)

if __name__=="__main__":
    main()
