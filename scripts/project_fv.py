#!/usr/bin/env python3
#
# Project Jeremy's wavespace data unto a FV grid
#
#

# ========================================================================
#
# Imports
#
# ========================================================================
import argparse
import sys
import os
import time
from datetime import timedelta
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))
import ho_apriori.velocity.velocity as velocity
import ho_apriori.fv.fv as fv


# ========================================================================
#
# Parse arguments
#
# ========================================================================
parser = argparse.ArgumentParser(
    description='Convert velocity fields for Pele')
parser.add_argument(
    '-p', '--plot', help='Show plots', action='store_true')
args = parser.parse_args()


# ========================================================================
#
# Main
#
# ========================================================================
# Timer
start = time.time()

# Setup
resolution = 4
mu = 0.0028
Re = 1. / mu

# Load the velocity fields from the UT data
fname = os.path.abspath('../ho_apriori/data/hit_ut_wavespace_256.npz')
velocities = velocity.Velocity()
velocities.read(fname)

# Define FV solution space and project solution
fv = fv.FV(resolution, 2 * np.pi, velocities)
projection = time.time()
# fv.fast_projection(order=0)
fv.interpolation()
end = time.time() - start
print("Elapsed projection time " + str(timedelta(seconds=end)) +
      " (or {0:f} seconds)".format(end))
dat = fv.to_df()

# Rename columns to conform to Pele ordering
dat = dat.rename(columns={'x': 'y', 'y': 'x'})

# Sort coordinates to be read easily in Fortran
dat.sort_values(by=['z', 'y', 'x'], inplace=True)

# Calculate urms
urms = np.sqrt(np.mean(dat['u']**2 + dat['v']**2 + dat['w']**2) / 3)

# Calculate Taylor length scale (note Fortran ordering assumption for gradient)
u2 = np.mean(dat['u']**2)
dudx2 = np.mean(
    np.gradient(dat['u'].values.reshape((resolution, resolution, resolution),
                                        order='F'),
                fv.dx[0],
                axis=0)**2)
lambda0 = np.sqrt(u2 / dudx2)
k0 = 2. / lambda0
tau = lambda0 / urms
Re_lambda = urms * lambda0 / mu

# Normalize the data by urms
dat['u'] /= urms
dat['v'] /= urms
dat['w'] /= urms

# Print some information
print("Simulation information:")
print('\t resolution =', resolution)
print('\t urms =', urms)
print('\t lambda0 =', lambda0)
print('\t k0 = 2/lambda0 =', k0)
print('\t tau = lambda0/urms =', tau)
print('\t mu =', mu)
print('\t Re = 1/mu =', Re)
print('\t Re_lambda = urms*lambda0/mu = ', Re_lambda)

# ========================================================================
# Write out the data so we can read it in Pele IC function
oname = "hit_ic_ut_{0:d}.dat".format(resolution)
dat.to_csv(oname,
           columns=['x', 'y', 'z', 'u', 'v', 'w'],
           float_format='%.18e',
           index=False)

# output timer
end = time.time() - start
print("Elapsed time " + str(timedelta(seconds=end)) +
      " (or {0:f} seconds)".format(end))
