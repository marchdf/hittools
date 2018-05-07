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
import operator
import logging
from datetime import timedelta
import subprocess as sp
import numpy as np
import pandas as pd
from mpi4py import MPI

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))
import hittools.velocity.velocity as velocity
import hittools.fv.fv as fv


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
class FullPaths(argparse.Action):
    """Expand user- and relative-paths"""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace,
                self.dest,
                os.path.abspath(os.path.expanduser(values)))


parser = argparse.ArgumentParser(
    description='Project velocity fields into a finite volume space')
parser.add_argument('-r',
                    '--resolution',
                    dest='res',
                    help='Number of element in one direction',
                    type=int,
                    default=8)
parser.add_argument('-o',
                    '--order',
                    dest='order',
                    help='Integration order',
                    type=int,
                    default=4)
parser.add_argument('-f', '--file',
                    dest='iname',
                    help='File with wavespace velocity fields',
                    type=str,
                    required=True,
                    action=FullPaths)
parser.add_argument('-b', '--binfmt',
                    dest='binfmt',
                    help='Use a binary output format',
                    action='store_true')
args = parser.parse_args()


# ========================================================================
#
# Main
#
# ========================================================================

# Timers
timers = {'projection': 0,
          'loading': 0,
          'writing': 0,
          'total': 0}
timers['total'] = time.time()

# Setup
xmin = 0
xmax = 2 * np.pi
L = xmax - xmin

# MPI setup
minimize_communication = False
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
if minimize_communication:
    dimensions = MPI.Compute_dims(nprocs, 3)
else:  # you won't have to sort data on merge
    dimensions = [1, 1, nprocs]
periodicity = (False, False, False)
grid = comm.Create_cart(dimensions, periodicity, reorder=True)
coords = grid.Get_coords(rank)

# Logging information
if rank == 0:
    pfx = "fv_{0:d}".format(args.res)
    logname = pfx + '.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        filename=logname,
        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    logging.info("Running MPI job with {0:d} procs".format(nprocs))
    logging.info("  Dim: " + str(grid.Get_dim()))
    logging.info("  Topology: " + str(grid.Get_topo()))

# Exit with error if procs doesn't divide the space evenly
if (not minimize_communication) and (args.res % nprocs != 0):
    sys.exit(
        "Number of processors ({0:d}) does not divide z-direction nicely (N cells = {1:d})".format(
            nprocs,
            args.res))

# ========================================================================
# Perform the projection

# Define FV solution space
pmap = grid.Get_topo()[0]
xloc = np.linspace(xmin, xmax, pmap[0] + 1)
yloc = np.linspace(xmin, xmax, pmap[1] + 1)
zloc = np.linspace(xmin, xmax, pmap[2] + 1)
fvs = fv.FV([args.res // pmap[0], args.res // pmap[1], args.res // pmap[2]],
            [xloc[coords[0]], yloc[coords[1]], zloc[coords[2]]],
            [xloc[coords[0] + 1], yloc[coords[1] + 1], zloc[coords[2] + 1]])

# Load the velocity fields
if rank == 0:
    logging.info("  Loading file: {0:s}".format(args.iname))
timers['loading'] = time.time()
velocities = velocity.Velocity.fromSpectralFile(args.iname)
timers['loading'] = time.time() - timers['loading']

# Project velocities on FV space
timers['projection'] = time.time()
fvs.fast_projection_nufft(velocities, order=args.order)
timers['projection'] = time.time() - timers['projection']

# ========================================================================
# Write out the data files individually (we will merge later)
timers['writing'] = time.time()
dat = fvs.to_df()

# Sort coordinates to be read easily in Fortran
dat.sort_values(by=['z', 'y', 'x'], inplace=True)

# Rearrange the columns
dat = dat[['x', 'y', 'z', 'u', 'v', 'w']]

opfx = "fv_{0:d}_{1:d}_{2:d}".format(args.res, nprocs, rank)
if args.binfmt:
    oname = opfx + ".in"
    dat.values.tofile(oname)
else:
    oname = opfx + ".dat"
    dat.to_csv(oname,
               columns=['x', 'y', 'z', 'u', 'v', 'w'],
               float_format='%.18e',
               index=False)

timers['writing'] = time.time() - timers['writing']

# ========================================================================
# Output information
comm.Barrier()
timers['total'] = time.time() - timers['total']
if rank == 0:
    # Print some information
    logging.info("  FV solution information:")
    logging.info('    interpolation order = {0:d}'.format(args.order))
    logging.info('    resolution = {0:d}'.format(args.res))
    logging.info("  Timers:")
    for key, value in sorted(
            timers.items(), key=operator.itemgetter(1), reverse=True):
        logging.info("    {0:s} {1:s} (or {2:.3f} seconds)".format(
            key, str(timedelta(seconds=value)), value))
