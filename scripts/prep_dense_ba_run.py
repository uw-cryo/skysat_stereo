#! /usr/bin/env python

import os,sys,glob,re
import argparse
import numpy
from skysat_stereo import skysat
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
    triplet_stereo_matches = sorted(glob.glob(os.path.join(stereo_master_dir,'20*/*/run*-*disp*.match')))
    print('Found {} dense matches'.format(len(triplet_stereo_matches)))
    ba_dir = os.path.abspath(args.ba_dir)
    if  not os.path.isdir(ba_dir):
        os.makedirs(ba_dir)
    out_dense_match_list = [os.path.join(ba_dir,'run-'+os.path.basename(match).split('run-disp-',15)[1]) for match in triplet_stereo_matches]
    for idx,match in tqdm(enumerate(triplet_stereo_matches)):
        shutil.copy2(match, out_dense_match_list[idx])
    print("Copied all files successfully")
    if args.modify_overlap == 1:
        img_fol = os.path.abspath(args.img)
        orig_df = pd.read_pickle(os.path.abspath(args.orig_pickle))
        dense_df = pd.read_pickle(os.path.abspath(args.dense_match_pickle))
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
        if not args.out_overlap_fn:
            out_overlap = os.path.join(ba_dir,'overlap_list_adapted_from_dense_matches.txt')
        else:
            out_overlap = os.path.join(ba_dir,args.out_overlap_fn)
        print("Saving adjusted overlap list at {}".format(out_overlap))
        with open(out_overlap,'w') as foo:
            for idx,img1 in enumerate(new_img1):
                out_str = '{} {}\n'.format(img1,new_img2[idx])
                f.write(out_str)
    print("Script is complete !")

if __name__=="__main__":
    main()
