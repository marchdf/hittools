# Copyright 2017 National Renewable Energy Laboratory. This software
# is released under the license detailed in the file, LICENSE, which
# is located in the top-level directory structure.

import os
import unittest
import numpy as np
import numpy.testing as npt
import ho_apriori.velocity.velocity as velocity
import ho_apriori.data.data as data


class VelocityTestCase(unittest.TestCase):
    """Tests for `velocity.py`."""

    def setUp(self):
        parent = os.path.abspath(os.path.join(__file__, '../..'))
        self.fname = os.path.abspath(os.path.join(
            parent, 'ho_apriori', 'data', 'toy_data.npz'))

        # Use the data class to create and output toy data
        self.data = data.Data()
        self.data.output_data(self.fname)

        # Load the velocity fields from the toy data
        self.velocities = velocity.Velocity()
        self.velocities.read(self.fname)

        # Set some parameters
        self.width = 4
        np.random.seed(1)

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
        """Is the interpolation of a function with DFT coefficients correct?"""

        # Loop over random points in the domain
        xis = np.random.uniform(low=0, high=2 * np.pi, size=(100, 3))
        for xi in xis:

            # Get the interpolated velocity using DFT coefficients
            Ui = self.velocities.get_interpolated_velocity(xi)

            npt.assert_allclose(Ui[0],
                                data.velocity_x(xi),
                                rtol=1e-11)
            npt.assert_allclose(Ui[1],
                                data.velocity_y(xi),
                                rtol=1e-11)
            npt.assert_allclose(Ui[2],
                                data.velocity_z(xi),
                                rtol=1e-11)


if __name__ == '__main__':
    unittest.main()
