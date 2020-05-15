# workflow for generating DEMs from SkySat-C L1A videos
 
## Preprocessing:

In the preprocessing step, images are sampled (saved as truncated frame_index csv file) at a given interval and frame camera models, gcp files are written 
- Run `skysat_preprocess.py` as: 

```skysat_preprocess.py -mode video -t pinhole -img video_path/frames/ -video_sampling_mode: num_images -sampler 60 -outdir subsampled_frames_dir -frame_index video_path/frame_index.csv -dem path_to_reference_DEM -product l1a```
