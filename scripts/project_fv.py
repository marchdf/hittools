#!/usr/bin/env python3
#
# Project wavespace data unto a FV grid
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
import subprocess as sp
import numpy as np
import pandas as pd
from mpi4py import MPI

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))
import hit_tools.velocity.velocity as velocity
import hit_tools.fv.fv as fv


# ========================================================================
#
# Function definitions
#
# ========================================================================
def run_cmd(cmd):
    log = open('logfile', "w")
    proc = sp.Popen(cmd,
                    shell=True,
                    stdout=log,
                    stderr=sp.PIPE)
    retcode = proc.wait()
    log.flush()

    return retcode


# ========================================================================
#
# Parse arguments
#
# ========================================================================
parser = argparse.ArgumentParser(
    description='Project velocity fields into a finite volume space')
parser.add_argument('-r',
                    '--resolution',
                    dest='res',
                    help='Number of element in one direction',
                    type=int,
                    default=6)
parser.add_argument('-o',
                    '--order',
                    dest='order',
                    help='Integration order',
                    type=int,
                    default=4)
args = parser.parse_args()


# ========================================================================
#
# Main
#
# ========================================================================

# Timers
timers = {'projection': 0,
          'merge': 0,
          'statistics': 0,
          'total': 0}
timers['total'] = time.time()

# Setup
mu = 0.0028
Re = 1. / mu
xmin = 0
xmax = 2 * np.pi
L = xmax - xmin

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
dimensions = MPI.Compute_dims(nprocs, 3)
periodicity = (False, False, False)
grid = comm.Create_cart(dimensions, periodicity, reorder=True)
coords = grid.Get_coords(rank)

if rank == 0:
    print("Running MPI job with {0:d} procs".format(nprocs))
    print("  Dim: " + str(grid.Get_dim()))
    print("  Topology: " + str(grid.Get_topo()))

# ========================================================================
# Perform the projection
timers['projection'] = time.time()

# Define FV solution space
pmap = grid.Get_topo()[0]
xloc = np.linspace(xmin, xmax, pmap[0] + 1)
yloc = np.linspace(xmin, xmax, pmap[1] + 1)
zloc = np.linspace(xmin, xmax, pmap[2] + 1)
fvs = fv.FV([args.res // pmap[0], args.res // pmap[1], args.res // pmap[2]],
            [xloc[coords[0]], yloc[coords[1]], zloc[coords[2]]],
            [xloc[coords[0] + 1], yloc[coords[1] + 1], zloc[coords[2] + 1]])

# Load the velocity fields
fname = os.path.abspath('../hit_tools/data/hit_ut_wavespace_256.npz')
#fname = os.path.abspath('../hit_tools/data/toy_data.npz')
velocities = velocity.Velocity()
velocities.read(fname)

# Project velocities on FV space
fvs.fast_projection_nufft(velocities, order=args.order)

# ========================================================================
# Write out the data files individually (we will merge later)
dat = fvs.to_df()

# Rename columns to conform to Pele ordering
dat = dat.rename(columns={'x': 'y', 'y': 'x'})

# Sort coordinates to be read easily in Fortran
dat.sort_values(by=['z', 'y', 'x'], inplace=True)

oname = "fv_{0:d}_{1:d}_{2:d}.dat".format(args.res, nprocs, rank)
dat.to_csv(oname,
           columns=['x', 'y', 'z', 'u', 'v', 'w'],
           float_format='%.18e',
           index=False)

timers['projection'] = time.time() - timers['projection']

# ========================================================================
# Merge the files together on rank 0, do statistics and normalize
comm.Barrier()
if rank == 0:

    # Merge all the data files
    timers['merge'] = time.time()
    fname = "hit_ic_ut_{0:d}.dat".format(args.res)
    tmpname = 'tmp.dat'
    retcode = run_cmd('head -1 ' + oname + ' > ' + tmpname)
    for n in range(nprocs):
        tname = "fv_{0:d}_{1:d}_{2:d}.dat".format(args.res, nprocs, n)
        retcode = run_cmd('tail -n +2 -q ' + tname + ' >> ' + tmpname)

    # Sort the merged file
    retcode = run_cmd('head -n 1 ' + tmpname + ' > ' + fname)
    retcode = run_cmd('tail -n +2 -q ' +
                      tmpname +
                      ' | sort -k3 -k2 -k1 -g -t, >> ' +
                      fname)
    timers['merge'] = time.time() - timers['merge']

    # Do some statistics
    timers['statistics'] = time.time()
    dat = pd.read_csv(fname)

    # Calculate urms
    umag = dat['u']**2 + dat['v']**2 + dat['w']**2
    urms = np.sqrt(np.mean(umag) / 3)

    # Calculate kinetic energy
    KE = 0.5 * np.sum(umag) * np.prod(fvs.dx)

    # Calculate Taylor length scale (note Fortran ordering assumption for
    # gradient)
    u2 = np.mean(dat['u']**2)
    dudx2 = np.mean(
        np.gradient(
            dat['u'].values.reshape(
                (args.res,
                 args.res,
                 args.res),
                order='F'),
            fvs.dx[0],
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
    print("  Simulation information:")
    print('    resolution =', args.res)
    print('    urms =', urms)
    print('    KE =', KE)
    print('    lambda0 =', lambda0)
    print('    k0 = 2/lambda0 =', k0)
    print('    tau = lambda0/urms =', tau)
    print('    mu =', mu)
    print('    Re = 1/mu =', Re)
    print('    Re_lambda = urms*lambda0/mu = ', Re_lambda)

    dat.to_csv(fname,
               columns=['x', 'y', 'z', 'u', 'v', 'w'],
               float_format='%.18e',
               index=False)

    # Clean up
    os.remove(tmpname)

    timers['statistics'] = time.time() - timers['statistics']


# output timer
timers['total'] = time.time() - timers['total']
if rank == 0:
    print("  Timers:")
    for key, value in timers.items():
        print("    " + key + " " + str(timedelta(seconds=value)) +
              " (or {0:.3f} seconds)".format(value))
