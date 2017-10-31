# Copyright 2017 National Renewable Energy Laboratory. This software
# is released under the license detailed in the file, LICENSE, which
# is located in the top-level directory structure.

# ========================================================================
#
# Imports
#
# ========================================================================
import numpy as np


# ========================================================================
#
# Class definitions
#
# ========================================================================
class Velocity:
    'Velocity data'

    # ========================================================================
    def __init__(self):

        # Initialize variables
        self.fname = ''
        self.N = np.array([0, 0, 0], dtype=np.int64)
        self.L = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi], dtype=np.float64)
        self.dx = np.array([0, 0, 0], dtype=np.float64)
        self.x = np.empty((0, 0, 0), dtype=np.float64)
        self.y = np.empty((0, 0, 0), dtype=np.float64)
        self.z = np.empty((0, 0, 0), dtype=np.float64)
        self.U = np.empty((0, 0, 0), dtype=np.float64)
        self.V = np.empty((0, 0, 0), dtype=np.float64)
        self.W = np.empty((0, 0, 0), dtype=np.float64)
        self.Uf = np.empty((0, 0, 0), dtype=np.complex128)
        self.Vf = np.empty((0, 0, 0), dtype=np.complex128)
        self.Wf = np.empty((0, 0, 0), dtype=np.complex128)

    # ========================================================================
    def read(self, fname):
        """Read spectral data from numpy file."""
        self.fname = fname

        # load the data
        data = np.load(self.fname)
        self.Uf = data['uf']
        self.Vf = data['vf']
        self.Wf = data['wf']

        # Inverse fft to get spatial data
        self.U = np.fft.irfftn(self.Uf)
        self.V = np.fft.irfftn(self.Vf)
        self.W = np.fft.irfftn(self.Wf)

        # Save off some attributes
        self.N = np.array(self.U.shape, dtype=np.int64)
        self.dx = self.L / np.asarray(self.N, dtype=np.float64)
        self.x = np.arange(0, self.L[0], self.dx[0])
        self.y = np.arange(0, self.L[1], self.dx[1])
        self.z = np.arange(0, self.L[2], self.dx[2])
