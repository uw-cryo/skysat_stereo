#!/usr/bin/env python

from distutils.core import setup

setup(name='skysat_stereo',
      version='0.1',
      description='library for DEM generation workflows from Planet SkySat-C imagery ',
      author='Shashank Bhushan and Team 3D',
      author_email='sbaglapl@uw.edu',
      url='https://github.com/uw-cryo/skysat_stereo.git',
      packages=['skysat_stereo'],
      install_requires=['requests']
      )
