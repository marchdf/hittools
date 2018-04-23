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

# Logging information
if rank == 0:
    pfx = "hit_ic_ut_{0:d}".format(args.res)
    logname = pfx + '.log'
    logging.basicConfig(level=logging.INFO,
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
if rank == 0:
    logging.info("  Loading file: {0:s}".format(args.iname))
velocities = velocity.Velocity.fromSpectralFile(args.iname)

# Project velocities on FV space
fvs.fast_projection_nufft(velocities, order=args.order)

# ========================================================================
# Write out the data files individually (we will merge later)
dat = fvs.to_df()

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
    fname = pfx + ".dat"
    tmpname = 'tmp.dat'
    retcode = run_cmd('head -1 ' + oname + ' > ' + tmpname)
    for n in range(nprocs):
        tname = "fv_{0:d}_{1:d}_{2:d}.dat".format(args.res, nprocs, n)
        retcode = run_cmd('tail -n +2 -q ' + tname + ' >> ' + tmpname)

    # Sort coordinates to be read easily in Fortran
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

    # Calculate kinetic energy and other integrals
    KE = 0.5 * np.mean(umag)
    KEu = 0.5 * np.mean(dat['u']**2)
    KEv = 0.5 * np.mean(dat['v']**2)
    KEw = 0.5 * np.mean(dat['w']**2)

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

    # Incompressible code so div u = 0
    div_u = np.gradient(dat['u'].values.reshape((args.res,
                                                 args.res,
                                                 args.res),
                                                order='F'),
                        fvs.dx[0],
                        axis=0) + \
        np.gradient(dat['v'].values.reshape((args.res,
                                             args.res,
                                             args.res),
                                            order='F'),
                    fvs.dx[1],
                    axis=1) + \
        np.gradient(dat['w'].values.reshape((args.res,
                                             args.res,
                                             args.res),
                                            order='F'),
                    fvs.dx[2],
                    axis=2)

    # Print some information
    logging.info("  FV solution information:")
    logging.info('    interpolation order = {0:d}'.format(args.order))
    logging.info('    resolution = {0:d}'.format(args.res))
    logging.info('    urms = {0:.16f}'.format(urms))
    logging.info('    KE = {0:f} (u:{1:f}, v:{2:f}, w:{3:f})'.format(KE,
                                                                     KEu,
                                                                     KEv,
                                                                     KEw))
    logging.info('    lambda0 = {0:.16f}'.format(lambda0))
    logging.info('    k0 = 2/lambda0 = {0:.16f}'.format(k0))
    logging.info('    tau = lambda0/urms = {0:.16f}'.format(tau))
    logging.info('    div u = {0:.16e}'.format(np.sum(div_u)))

    # Clean up
    os.remove(tmpname)

    timers['statistics'] = time.time() - timers['statistics']

# Clean up
os.remove(oname)

# output timer
timers['total'] = time.time() - timers['total']
if rank == 0:
    logging.info("  Timers:")
    for key, value in sorted(timers.items(), key=operator.itemgetter(1), reverse=True):
        logging.info("    {0:s} {1:s} (or {2:.3f} seconds)".format(
            key, str(timedelta(seconds=value)), value))
