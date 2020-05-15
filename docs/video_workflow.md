# workflow for generating DEMs from SkySat-C L1A videos

## Preprocessing:

In the preprocessing step, images are sampled (saved as truncated frame_index csv file) at a given interval and frame camera models, gcp files are written 
- Run `skysat_preprocess.py` as: 
