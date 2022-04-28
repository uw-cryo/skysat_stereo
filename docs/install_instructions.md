# Installation for skysat_stereo

- We use srtm4 package, which for now needs developer version of libtiff
- So first of all, install libtif on your machine using: `apt-get install libtiff-dev` for Ubuntu or `brew install libtiff` for (macOS)
- Download Ames Stereo Pipeline (v 3.0.1 alpha) executables from [here](https://github.com/NeoGeographyToolkit/StereoPipeline/releases/download/2022-04-22-daily-build/StereoPipeline-3.0.1-alpha-2022-04-22-x86_64-Linux.tar.bz2). Untar the downloaded file, and add the `bin` folder in the untarred folder to your `.bashrc` profile.  
- Clone the `skysat_stereo` repo to the location of your choice using `https://github.com/uw-cryo/skysat_stereo.git`
- We recommend using conda for managing packages. Powerusers can have a look at the `environment.yml` file in the github repository to make an environment using that.
- Otherwise, one can simply initiate a conda environment to avoid conflicts using the environment.yml file.
- Run the command `conda env create skysat_stereo/environment.yml`. This will create a new environemnt `skysat_stereo` containing all dependencies.
- Each time a new terminal is opened, activate the environment using `conda activate skysat_stereo`.
- Activate the skysat_stereo environment, and install the repository in editable command: `pip install -e skysat_stereo/`
- This will install all library files (the fancy apis), to run the command line calls, these scripts need to added path.
- To use the command line executables located in the `scripts` directory of `skysat_stereo` directory, add the `skysat_stereo/scripts/` path to your `.bashrc` as well. A guide on how to add paths to your .bashrc can be found [here](https://gist.github.com/nex3/c395b2f8fd4b02068be37c961301caa7).
- If any of this sounds confusing, please refer to this [guide](https://github.com/dshean/demcoreg/blob/master/docs/beginners_doc.md) which has tricks for installing packages/enivronment using conda for new users.
