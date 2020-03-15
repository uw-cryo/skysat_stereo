# skysat_3d
A collection of scripts and functions to process Planet Skysat imagery 
- Planet Skysat is a 13 satellite LEO constellation, capable of acquiring videos, triplet stereo and mono images at resolutions of 0.7 to 0.9 m. 
- The videos (continous 30 fps video at varying angles)  and triplet stereo (3 image sets from 3 view angles) can be used for high-resolution DEM generation (2 to 4 m/px). 
- However,due to inherent inaccuracies in the camera models (RPC or rigorous frame camera), the Skysat constellations have
generally been unexplored for stereo reconstruction since the launch of their first constellation in 2013.
- We developed an automated workflow to process Skysat video and triplet stereo collections. 
- This workflow involves deriving frame camera models from the original RPC camera models provided by Planet, and using ASP’s iterative bundle adjustment capabilities to improve relative camera position and
orientation for all scenes, minimizing residuals of matched feature reprojection in all cameras. 
- We then compute pairwise DEMs from all possible stereo pair combinations of the refined images, which are
merged after outlier removal to obtain the final composite DEM.

Major Software 
- NASA Ames Stereo Pipeline
- pygeotools
- demcoreg
- Python stack 
- Python Geospatial stack
The scripts have been developed and tested on Linux operating system and should work for all nix platforms. No tests have been performed on Windows OS.

Development Team
- Shashank Bhushan (UW)
- David Shean (UW)
- Oleg Alexandrov (NASA Ames Research Center)
- Scott Henderson (UW)

Acknowledgement:
- Data was funded through the NASA Commercial Data Buy Pilot allocation to NASA Stereo2SWE (80NSSC18K1405) Project. Shashank Bhushan was funded through the NASA FINESST (80NSSC19K1338) and NASA HiMAT (NNX16AQ88G) awards. David Shean, Oleg Alexandrov and Scott Henderson were funded through the NASA Stereo2SWE (80NSSC18K1405) award.
