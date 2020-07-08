# skysat_stereo
A collection of scripts and functions to process Planet SkySat imagery

## Overview
Planet operates a constellation of 13 SkySat-C SmallSats, which can acquire very high resolution (0.7 m to 0.9 m) triplet stereo and continuous video imagery with short revisit times. This provides an excellent opportunity to derive on-demand high-resolution Digital Elevation Model(DEM)s for any point on the Earth's surface, with immense applications for Earth science research. However, production of DEMs from SkySat imagery products is currently limited by the inherent accuracy of the default camera models. In this project. we developed an automated workflow to improve the accuracy of the SkySat camera models and a pipeline to produce accurate DEMs and orthoimage mosaics from the refined camera models.

### Repository Purpose
This Repository contains the scripts developed for automated stereo processing of SkySat products. Currently, the main purpose of the repository is to offer a supplement to the manuscript under review. But the repository is under active development and we will be updating the functions, documentation, make the pipeline more robust and potentially add unit tests. We will also add a dedicated contributing section, but till that is done, we welcome early feedback from people who had a chance to visit this page :) !


## Contents
- [skysat_stereo library](/skysat_stereo) contains several python functions for Ames Stereo Pipeline and specific SkySat processing.
- [scripts directory](/scripts/) contains the following command line scripts for various steps in SkySat imagery.
1. [skysat_overlap.py](/scripts/skysat_overlap.py) for finding overlapping scenes from the entire triplet stereo collection.
2. [skysat_preprocess.py](/scripts/skysat_preprocess.py) for sampling/subsetting video scenes and creating frame camera models from input metadata.
3. [ba_skysat.py](/scripts/ba_skysat.py). Bundle adjustment and camera refinement routine.
4. [skysat_stereo_cli.py](/scripts/skysat_stereo_cli.py) for performing stereo reconstruction for the input SkySat scenes.
5. [skysat_dem_mos.py](/scripts/skysat_dem_mos.py) perform mosaicing of triplet stereo and video DEMs and produce relative accuracy and count metrics.
6. [skysat_pc_cam.py](/scripts/skysat_pc_cam.py) utility to grid in parallel input point-clouds into DEMs, co-register to a reference DEM a single or multiple DEMs, align frame camera models using ICP transformation vectors, produce RPC camera models from aligned frame camera models.
7. [skysat_orthorectify.py](/scripts/skysat_orthorectify.py) orhtorectify images for browse or scientific purposes, produce orthoimage mosaics from the entire collection.
8. [plot_disparity.py](/scripts/plot_disparity.py) Plot x,y disparity, intersection error and DEMs from a stereo directory.

- [notebooks](/notebooks/) contains notebooks used in scientific analysis.

## Sample Outputs

### Triplet Stereo
![triplet_product](/docs/img/Figure3.jpg)

Figure 1: Sample orthoimage mosaic and DEM composite generated from a SkySat-C triplet stereo collection over Mt. Rainier, WA, USA. These final products were derived from L1B imagery that is \textsuperscript{\textcopyright}Planet, 2019.

![triplet_accuracy](/docs/img/Figure4.jpg)

Figure 2: Measure of relative and absolute accuracy before (using Planet RPC camera) and after (skysat_stereo correction workflow).

### Video

![video_samples](/docs/img/Figure5.jpg)

Figure 3: Sample products from video collection over Mt. St. Helen's crater (after skysat_stereo correction workflow). These final products were derived from L1A imagery that is \textsuperscript{\textcopyright}Planet, 2019.

![video_planet_rpc](/docs/img/SF2.jpg)

Figure 4: Same as in Figure 3 but produced from Planet provided RPC camera model.


## Major Software
- [NASA Ames Stereo Pipeline v 2.6.2](https://stereopipeline.readthedocs.io/en/latest/)
- [pygeotools](https://github.com/dshean/pygeotools)
- [demcoreg](https://github.com/dshean/demcoreg)
- Python stack ([numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/), [p_tqdm](https://github.com/swansonk14/p_tqdm))
- Python Geospatial stack ([gdal](https://gdal.org/), [rasterio](https://rasterio.readthedocs.io/en/latest/), [shapely](https://github.com/Toblerity/Shapely), [geopandas](https://geopandas.org/), [contextily](https://contextily.readthedocs.io/en/latest/))
- See also exact list with specific versions (if any) in the [yml file](/environment.yml).

## Installation
The scripts have been developed and tested on Linux operating system and should work for all nix platforms. No tests have been performed on Windows OS. Please see the [install instructions](/docs/install_instructions.md).
* Note: These scripts were developed to be run on a single [Broadwell node](https://www.nas.nasa.gov/hecc/resources/pleiades.html), using parallel processing technique. The number of concurrent parallel jobs etc., are fixed based on Broadwell's resources. These would be needed to be tweaked so as to run on other setups. This and many more knobs will be generalized as the project evolves.

## License
This project is licensed under the terms of the MIT License.


## Citation
Manuscript detailing the software utility is under review. Citation instructions for the manuscript will be updated. For now, if you find the contents of the repository useful for any purposes (scientific/commercial), please cite the package and the manuscript as:

* Insert Zenodo citation text here.....

* Bhushan, Shashank, Shean, David E., Alexandrov, Oleg, & Henderson, Scott. (2020). Automated digital elevation model (DEM) generation from very-high-resolution Planet SkySat triplet stereo and video imagery. ISPRS Journal of Photogrammetry and Remote Sensing, submitted.

## Acknowledgments

* Data was funded through the NASA Commercial Data Buy Pilot allocation to NASA Stereo2SWE (80NSSC18K1405) Project. Shashank Bhushan was funded through the NASA FINESST (80NSSC19K1338) and NASA HiMAT (NNX16AQ88G) awards. David Shean, Oleg Alexandrov and Scott Henderson were funded through the NASA Stereo2SWE (80NSSC18K1405) award. We acknowledge Friedrich Knuth and Michelle Hu for providing assistance and feedback on code development and documentation. 
