#!/usr/bin/env python3

# Copyright 2017 National Renewable Energy Laboratory. This software
# is released under the license detailed in the file, LICENSE, which
# is located in the top-level directory structure.

# ========================================================================
#
# Imports
#
# ========================================================================
import sys
import os
import argparse
import subprocess
import time
from datetime import timedelta
import numpy as np
import ho_apriori.velocity.velocity as velocity
import ho_apriori.constants.constants as constants
import ho_apriori.sgs.sgs as sgs
import ho_apriori.fv.fv as fv


# ========================================================================
#
# Function definitions
#
# ========================================================================
def get_git_revision_hash():
    """Returns the git version of this project"""
    return subprocess.check_output(['git', 'describe', '--always'],
                                   universal_newlines=True)


def get_DuDx(velocities):

    dudx = velocities.get_filtered_velocity_derivative(0, 0)
    dudy = velocities.get_filtered_velocity_derivative(0, 1)
    dudz = velocities.get_filtered_velocity_derivative(0, 2)
    dvdx = velocities.get_filtered_velocity_derivative(1, 0)
    dvdy = velocities.get_filtered_velocity_derivative(1, 1)
    dvdz = velocities.get_filtered_velocity_derivative(1, 2)
    dwdx = velocities.get_filtered_velocity_derivative(2, 0)
    dwdy = velocities.get_filtered_velocity_derivative(2, 1)
    dwdz = velocities.get_filtered_velocity_derivative(2, 2)

    return [[dudx, dudy, dudz],
            [dvdx, dvdy, dvdz],
            [dwdx, dwdy, dwdz]]


def get_Sij(velocities):
    DuDx = get_DuDx(velocities)

    S00 = DuDx[0][0]
    S01 = 0.5 * (DuDx[0][1] + DuDx[1][0])
    S02 = 0.5 * (DuDx[0][2] + DuDx[2][0])
    S11 = DuDx[1][1]
    S12 = 0.5 * (DuDx[1][2] + DuDx[2][1])
    S22 = DuDx[2][2]

    Sijmag = 0.0
    for i in range(3):
        for j in range(3):
            rateofstrain = 0.5 * (DuDx[i][j] + DuDx[j][i])
            Sijmag += rateofstrain**2
    Sijmag = np.sqrt(2.0 * Sijmag)

    Skk = S00 + S11 + S22

    return [[S00, S01, S02],
            [S01, S11, S12],
            [S02, S12, S22]], Sijmag, Skk


def get_tau_sgs_spectral(width, velocities):

    Sij, Sijmag, Skk = get_Sij(velocities)
    deltabar = width * velocities.dx[0]
    const = constants.Constants()
    mut = const.rho * deltabar**2 * Sijmag

    tau_sgs_kk = 2.0 * const.CI * mut * Sijmag

    tau_sgs_00 = -2.0 * const.Cs**2 * mut * \
        (Sij[0][0] - Skk / 3.) - tau_sgs_kk / 3.
    tau_sgs_01 = -2.0 * const.Cs**2 * mut * Sij[0][1]
    tau_sgs_02 = -2.0 * const.Cs**2 * mut * Sij[0][2]

    tau_sgs_11 = -2.0 * const.Cs**2 * mut * \
        (Sij[1][1] - Skk / 3.) - tau_sgs_kk / 3.
    tau_sgs_12 = -2.0 * const.Cs**2 * mut * Sij[1][2]

    tau_sgs_22 = -2.0 * const.Cs**2 * mut * \
        (Sij[2][2] - Skk / 3.) - tau_sgs_kk / 3.

    return [[tau_sgs_00, tau_sgs_01, tau_sgs_02],
            [tau_sgs_01, tau_sgs_11, tau_sgs_12],
            [tau_sgs_02, tau_sgs_12, tau_sgs_22]], tau_sgs_kk


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Main script for the study')
    parser.add_argument(
        '-s', '--show', help='Show the plots', action='store_true')
    args = parser.parse_args()

    # Problem setup
    start = time.time()
    print('Code version: ', get_git_revision_hash())

    # Load the velocity data
    parent = os.path.abspath(os.path.join(__file__, '..'))
    fname = os.path.abspath(os.path.join(
        parent, 'ho_apriori', 'data', 'toy_data.npz'))
    velocities = velocity.Velocity()
    velocities.read(fname)

    # Get the reference SGS terms
    width = 2
    velocities.gaussian_filter(width)
    tau_spectral = get_tau_sgs_spectral(width, velocities)

    # Output timer
    end = time.time() - start
    print("Elapsed time " + str(timedelta(seconds=end)) +
          " (or {0:f} seconds)".format(end))
