# Copyright 2017 National Renewable Energy Laboratory. This software
# is released under the license detailed in the file, LICENSE, which
# is located in the top-level directory structure.

import os
import unittest
import numpy as np
import numpy.testing as npt
from nufft import nufft3d2
import hit_tools.velocity.velocity as velocity
import hit_tools.data.data as data


class VelocityTestCase(unittest.TestCase):
    """Tests for `velocity.py`."""

    def setUp(self):
        parent = os.path.abspath(os.path.join(__file__, '../..'))
        self.fname = os.path.abspath(os.path.join(
            parent, 'hit_tools', 'data', 'toy_data.npz'))

        # Use the data class to create and output toy data
        self.data = data.Data()
        self.data.output_data(self.fname)

        # Load the velocity fields from the toy data
        self.velocities = velocity.Velocity()
        self.velocities.read(self.fname)

        # Set some parameters
        self.width = 4
        np.random.seed(1)

        # Random interpolation points in the domain
        self.xis = np.random.uniform(low=0, high=2 * np.pi, size=(100, 3))

    def test_read(self):
        """Is the velocity loading function correct?"""
        npt.assert_almost_equal(np.linalg.norm(self.velocities.U[0]),
                                64)
        npt.assert_almost_equal(np.linalg.norm(self.velocities.U[1]),
                                128)
        npt.assert_almost_equal(np.linalg.norm(self.velocities.U[2]),
                                192)
        npt.assert_almost_equal(np.linalg.norm(self.velocities.Uf[0]),
                                8192)
        npt.assert_almost_equal(np.linalg.norm(self.velocities.Uf[1]),
                                16384)
        npt.assert_almost_equal(np.linalg.norm(self.velocities.Uf[2]),
                                24576)

    def test_gaussian_filter(self):
        """Is the Gaussian filtering correct?"""
        self.velocities.gaussian_filter(self.width)

        npt.assert_almost_equal(np.linalg.norm(self.velocities.Uh[0]),
                                7.388369839861693)
        npt.assert_almost_equal(np.linalg.norm(self.velocities.Uh[1]),
                                14.776739679723407)
        npt.assert_almost_equal(np.linalg.norm(self.velocities.Uh[2]),
                                22.165109519585087)
        npt.assert_almost_equal(np.linalg.norm(self.velocities.Ufh[0]),
                                945.71133950229762)
        npt.assert_almost_equal(np.linalg.norm(self.velocities.Ufh[1]),
                                1891.422679004595)
        npt.assert_almost_equal(np.linalg.norm(self.velocities.Ufh[2]),
                                2837.134018506893)

    def test_filtered_velocity_derivative(self):
        """Is the velocity derivative correct?"""
        self.velocities.gaussian_filter(self.width)
        dudx = self.velocities.get_filtered_velocity_derivative(0, 0)
        dudy = self.velocities.get_filtered_velocity_derivative(0, 1)
        dudz = self.velocities.get_filtered_velocity_derivative(0, 2)
        dvdx = self.velocities.get_filtered_velocity_derivative(1, 0)
        dvdy = self.velocities.get_filtered_velocity_derivative(1, 1)
        dvdz = self.velocities.get_filtered_velocity_derivative(1, 2)
        dwdx = self.velocities.get_filtered_velocity_derivative(2, 0)
        dwdy = self.velocities.get_filtered_velocity_derivative(2, 1)
        dwdz = self.velocities.get_filtered_velocity_derivative(2, 2)

        npt.assert_almost_equal(np.linalg.norm(dudx),
                                14.776739679723388)
        npt.assert_almost_equal(np.linalg.norm(dudy),
                                29.553479359446769)
        npt.assert_almost_equal(np.linalg.norm(dudz),
                                59.106958718893658)
        npt.assert_almost_equal(np.linalg.norm(dvdx),
                                59.106958718893623)
        npt.assert_almost_equal(np.linalg.norm(dvdy),
                                118.21391743778726)
        npt.assert_almost_equal(np.linalg.norm(dvdz),
                                29.553479359446769)
        npt.assert_almost_equal(np.linalg.norm(dwdx),
                                177.3208761566807)
        npt.assert_almost_equal(np.linalg.norm(dwdy),
                                44.330219039170167)
        npt.assert_almost_equal(np.linalg.norm(dwdz),
                                88.660438078340349)

    def test_get_interpolated_velocity(self):
        """Is the interpolation  with DFT coefficients correct?"""

        # Probe random points in the domain
        Ui = np.zeros(self.xis.shape)
        for k, xi in enumerate(self.xis):

            # Get the interpolated velocity using DFT coefficients
            Ui[k, :] = self.velocities.get_interpolated_velocity(xi)

        # Tests
        npt.assert_allclose(Ui[:, 0],
                            data.velocity_x([self.xis[:, 0],
                                             self.xis[:, 1],
                                             self.xis[:, 2]]),
                            rtol=1e-11)
        npt.assert_allclose(Ui[:, 1],
                            data.velocity_y([self.xis[:, 0],
                                             self.xis[:, 1],
                                             self.xis[:, 2]]),
                            rtol=1e-11)
        npt.assert_allclose(Ui[:, 2],
                            data.velocity_z([self.xis[:, 0],
                                             self.xis[:, 1],
                                             self.xis[:, 2]]),
                            rtol=1e-11)

    def test_numba_get_interpolated_velocity(self):
        """Is the interpolation with DFT coefficients correct (Numba version)?"""

        # Probe random points in the domain
        Ui = np.zeros(self.xis.shape)
        for k, xi in enumerate(self.xis):

            # Get the interpolated velocity using DFT coefficients
            Ui[k, :] = self.velocities.numba_get_interpolated_velocity(xi)

        # Tests
        npt.assert_allclose(Ui[:, 0],
                            data.velocity_x([self.xis[:, 0],
                                             self.xis[:, 1],
                                             self.xis[:, 2]]),
                            rtol=1e-10)
        npt.assert_allclose(Ui[:, 1],
                            data.velocity_y([self.xis[:, 0],
                                             self.xis[:, 1],
                                             self.xis[:, 2]]),
                            rtol=1e-10)
        npt.assert_allclose(Ui[:, 2],
                            data.velocity_z([self.xis[:, 0],
                                             self.xis[:, 1],
                                             self.xis[:, 2]]),
                            rtol=1e-10)

    def test_nufft_get_interpolated_velocity(self):
        """Is the interpolation with DFT coefficients correct (NUFFT version)?"""

        # Our velocity data was generated with a real FFT. Fake the
        # data for the full FFT so we can use NUFFT.
        K = np.meshgrid(-self.velocities.k[0].astype(int),
                        -self.velocities.k[1].astype(int),
                        self.velocities.k[2][-2:0:-1].astype(int),
                        indexing='ij')
        Uf = [np.concatenate((self.velocities.Uf[c],
                              np.conj(self.velocities.Uf[c][K[0],
                                                            K[1],
                                                            K[2]])),
                             axis=2) for c in range(3)]

        # Probe random points in the domain
        Ui = np.zeros(self.xis.shape)
        for c in range(3):
            Ui[:, c] = np.real(nufft3d2(self.xis[:, 0],
                                        self.xis[:, 1],
                                        self.xis[:, 2],
                                        np.roll(np.roll(np.roll(Uf[c],
                                                                -int(self.velocities.N[0] / 2),
                                                                0),
                                                        -int(self.velocities.N[1] / 2),
                                                        1),
                                                -int(self.velocities.N[2] / 2),
                                                2),
                                        iflag=1,
                                        direct=False)) / self.velocities.N.prod()

        # Tests
        npt.assert_allclose(Ui[:, 0],
                            data.velocity_x([self.xis[:, 0],
                                             self.xis[:, 1],
                                             self.xis[:, 2]]),
                            rtol=1e-10)
        npt.assert_allclose(Ui[:, 1],
                            data.velocity_y([self.xis[:, 0],
                                             self.xis[:, 1],
                                             self.xis[:, 2]]),
                            rtol=1e-10)
        npt.assert_allclose(Ui[:, 2],
                            data.velocity_z([self.xis[:, 0],
                                             self.xis[:, 1],
                                             self.xis[:, 2]]),
                            rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
