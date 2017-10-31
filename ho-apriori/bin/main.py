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

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
import velocity


# ========================================================================
#
# Function definitions
#
# ========================================================================
def get_git_revision_hash():
    """Returns the git version of this project"""
    return subprocess.check_output(['git', 'describe', '--always'],
                                   universal_newlines=True)


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
    velocities = velocity.Velocity()
    velocities.read('../data/toy_data.npz')

    # Output timer
    end = time.time() - start
    print("Elapsed time " + str(timedelta(seconds=end)) +
          " (or {0:f} seconds)".format(end))
