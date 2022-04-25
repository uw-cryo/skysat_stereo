#! /usr/bin/env python

import os,sys,glob,re
import argparse
import numpy
from skysat_stereo import skysat
from skysat_stereo import skysat_stereo_workflow as workflow
from tqdm import tqdm
import pandas as pd
import shutil

def getparser():
    parser = argparse.ArgumentParser(
        description='Script for copying/renaming dense match files, creating new overlap list if the dense matches were created on ortho imagery stereo jobs')
    parser.add_argument('-img', default=None, help='path to unmapped image folder',required=False)
    parser.add_argument('-orig_pickle', default=None, help='path to original overlap pickle written by skysat_overlap.py (default: %(default)s)',required=False)
    parser.add_argument('-dense_match_pickle', default=None, help = 'path to pickle file written by stereo file (default: %(default)s)',required=False)
    parser.add_argument('-stereo_dir', default=None, help = 'master triplet stereo directory (default: %(default)s)',required=True)
    parser.add_argument('-ba_dir', default=None, help = 'path to expected ba directory (default: %(default)s)',required=True)
    mode_opt = [1,0]
    parser.add_argument('-modify_overlap',choices = mode_opt, type = int, default = 0, help = 'by default, copy match files only (0), if (1), then modify overlap (default: %(default)s)')
    parser.add_argument('-out_overlap_fn', default = None, help='out overlap filename (default: %(default)s)',required=False)
    return parser

def main():
    parser = getparser()
    args = parser.parse_args()
    stereo_master_dir = os.path.abspath(args.stereo_dir)
    ba_dir = os.path.abspath(args.ba_dir)
    if args.modify_overlap == 1:
        img_fol = os.path.abspath(args.img)
        orig_pickle = os.path.abspath(args.orig_pickle)
        dense_match_pickle = os.path.abspath(args.dense_match_pickle)
    else:
        img_fol = None
        orig_pickle=None
        dense_match_pickle = None
        workflow.dense_match_wrapper(stereo_master_dir,ba_dir,modify_overlap=args.modify_overlap,
                                     img_fol=img_fol,orig_pickle=orig_pickle,dense_match_pickle=dense_match_pickle,
                                     out_overlap_fn=args.out_overlap_fn)
    print("Script is complete !")

if __name__=="__main__":
    main()