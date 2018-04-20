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
# Function definitions
#
# ========================================================================
def velocity_x(x):
    """Helper function defining x-direction velocity."""
    return 1 * np.sin(2 * x[0]) * np.cos(4 * x[1]) * np.cos(8 * x[2])


def velocity_y(x):
    """Helper function defining y-direction velocity."""
    return 2 * np.cos(4 * x[0]) * np.cos(8 * x[1]) * np.sin(2 * x[2])


def velocity_z(x):
    """Helper function defining z-direction velocity."""
    return 3 * np.cos(8 * x[0]) * np.sin(2 * x[1]) * np.cos(4 * x[2])


# ========================================================================
#
# Class definitions
#
# ========================================================================
class Data:
    """Toy data used in testing"""

    # ========================================================================
    def __init__(self):

        self.N = np.array([32, 32, 32], dtype=np.int64)
        self.L = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi])
        self.dx = self.L / self.N
        x = np.arange(0, self.L[0], self.dx[0])
        y = np.arange(0, self.L[1], self.dx[1])
        z = np.arange(0, self.L[2], self.dx[2])

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        self.U = velocity_x([X, Y, Z])
        self.V = velocity_y([X, Y, Z])
        self.W = velocity_z([X, Y, Z])

        # Fourier data
        self.Uf = np.fft.rfftn(self.U)
        self.Vf = np.fft.rfftn(self.V)
        self.Wf = np.fft.rfftn(self.W)

    # ========================================================================
    def output_data(self, fname):
        """Output velocity data in Fourier space.

        :param fname: file name for the velocity data (numpy file)
        :type fname: string

        """
        np.savez_compressed(fname,
                            L=self.L,
                            uf=self.Uf,
                            vf=self.Vf,
                            wf=self.Wf)
