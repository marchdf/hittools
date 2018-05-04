#!/usr/bin/env python3
#
# Calculate statistics on data file
#

# ========================================================================
#
# Imports
#
# ========================================================================
import argparse
import os
import time
import operator
import logging
from datetime import timedelta
import numpy as np
import pandas as pd


# ========================================================================
#
# Function definitions
#
# ========================================================================


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
    description='Calculate some statistics')
parser.add_argument('-f', '--file',
                    dest='fname',
                    help='File containing data',
                    type=str,
                    required=True,
                    action=FullPaths)
parser.add_argument('-b', '--binfmt',
                    dest='binfmt',
                    help='Use a binary input format',
                    action='store_true')
args = parser.parse_args()


# ========================================================================
#
# Main
#
# ========================================================================

# Timers
timers = {'statistics': 0,
          'loading': 0,
          'total': 0}
timers['total'] = time.time()

# Logging information
logname = os.path.splitext(args.fname)[0] + '.log'
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

logging.info("Getting statistics on file")
logging.info("  file: " + args.fname)

# Load file
timers['loading'] = time.time()
if args.binfmt:
    dat = pd.DataFrame(data=np.reshape(np.fromfile(args.fname), (-1, 6)),
                       columns=['x', 'y', 'z', 'u', 'v', 'w'])

else:
    dat = pd.read_csv(args.fname)
timers['loading'] = time.time() - timers['loading']

# Do some statistics
timers['statistics'] = time.time()

# Spacing
dx = dat.x[1] - dat.x[0]
res = int(np.cbrt(len(dat)))

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
            (res,
             res,
             res),
            order='F'),
        dx,
        axis=0)**2)
lambda0 = np.sqrt(u2 / dudx2)
k0 = 2. / lambda0
tau = lambda0 / urms

# Incompressible code so div u = 0
divu = np.gradient(dat['u'].values.reshape((res,
                                            res,
                                            res),
                                           order='F'),
                   dx,
                   axis=0) + \
    np.gradient(dat['v'].values.reshape((res,
                                         res,
                                         res),
                                        order='F'),
                dx,
                axis=1) + \
    np.gradient(dat['w'].values.reshape((res,
                                         res,
                                         res),
                                        order='F'),
                dx,
                axis=2)
dilatation = np.mean(divu**2)

timers['statistics'] = time.time() - timers['statistics']


# Print some information
logging.info("  Solution information:")
logging.info('    urms = {0:.16f}'.format(urms))
logging.info('    KE = {0:f} (u:{1:f}, v:{2:f}, w:{3:f})'.format(KE,
                                                                 KEu,
                                                                 KEv,
                                                                 KEw))
logging.info('    lambda0 = {0:.16f}'.format(lambda0))
logging.info('    k0 = 2/lambda0 = {0:.16f}'.format(k0))
logging.info('    tau = lambda0/urms = {0:.16f}'.format(tau))
logging.info('    dilatation (FD) = 0 = {0:.16e}'.format(dilatation))
logging.info("  Timers:")
timers['total'] = time.time() - timers['total']
for key, value in sorted(
        timers.items(), key=operator.itemgetter(1), reverse=True):
    logging.info("    {0:s} {1:s} (or {2:.3f} seconds)".format(
        key, str(timedelta(seconds=value)), value))
