Update Notes for upcoming releases
### 0.3 
* Option to run from a particular stage in stereo (preprocessing, correlation, refinement, filter and triangulation) (added for triplet currently)
* Allow for controlling camera weight by the user during bundle_adjust/ adjust weight based on initial reprojection error (in progress)
* Extend logic of reading UTM zones from even unprojected Point Clouds (added)
* Virtual GCP support in bundle adjustment (in progress)
* Cross-track (multi-view triplet,mono support)
    * Bundle adjustment (in progress)
    * Stereo processing (done)
    * DEM mosaicking (done)
    * orthorectification (done)
* Ability to run processing for scenes within a bounding box 
    * Support added in skysat_overlap.py to write out overlapping pairs which are only intersect a bounding box (done)
    * This overlap list can be ingested by subsequent updated orthorectification and stereo processing and DEM processing programs (done)
    * Bundle adjustment (in progress)
* Update of wrapper scripts to implement the above changes (in progress)

