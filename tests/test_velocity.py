# Copyright 2017 National Renewable Energy Laboratory. This software
# is released under the license detailed in the file, LICENSE, which
# is located in the top-level directory structure.

import os
import unittest
import numpy as np
import numpy.testing as npt
from nufft import nufft3d2
import hittools.velocity.velocity as velocity
import hittools.data.data as data


class VelocityTestCase(unittest.TestCase):
    """Tests for `velocity.py`."""

    def setUp(self):
        parent = os.path.abspath(os.path.join(__file__, "../.."))
        self.fname = os.path.abspath(
            os.path.join(parent, "hittools", "data", "toy_data.npz")
        )

        # Use the data class to create and output toy data
        self.data = data.Data()
        self.data.output_data(self.fname)

        # Load the velocity fields from the toy data
        self.velocities = velocity.Velocity.fromSpectralFile(self.fname)

        # Set some parameters
        self.width = 4
        np.random.seed(1)

        # Random interpolation points in the domain
        self.xis = np.random.uniform(low=0, high=2 * np.pi, size=(100, 3))

        # Velocities for spectra tests
        N = np.array([32, 32, 32], dtype=np.int64)
        L = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi])
        x = np.linspace(0, L[0], N[0])
        y = np.linspace(0, L[1], N[1])
        z = np.linspace(0, L[2], N[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        U = [
            np.cos(2 * X) + np.cos(4 * Y) + np.cos(8 * Z),
            np.cos(4 * X) + np.cos(8 * Y) + np.cos(2 * Z),
            np.cos(8 * X) + np.cos(2 * Y) + np.cos(4 * Z),
        ]
        self.spec_velocities = velocity.Velocity(L, U)

        # Velocities for divergence tests (TG vortex)
        N = np.array([32, 32, 32], dtype=np.int64)
        L = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi])
        dx = L / N
        x = np.arange(0, L[0], dx[0])
        y = np.arange(0, L[1], dx[1])
        z = np.arange(0, L[2], dx[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        a = [2, 4, -6]  # sum is zero for div u = 0
        U = [
            np.cos(a[0] * X) * np.sin(a[1] * Y) * np.sin(a[2] * Z),
            np.sin(a[0] * X) * np.cos(a[1] * Y) * np.sin(a[2] * Z),
            np.sin(a[0] * X) * np.sin(a[1] * Y) * np.cos(a[2] * Z),
        ]
        self.tg_velocities = velocity.Velocity(L, U)

    def test_fromSpectralFile(self):
        """Is the velocity file reading function correct?"""
        npt.assert_almost_equal(np.linalg.norm(self.velocities.U[0]), 64)
        npt.assert_almost_equal(np.linalg.norm(self.velocities.U[1]), 128)
        npt.assert_almost_equal(np.linalg.norm(self.velocities.U[2]), 192)
        npt.assert_almost_equal(np.linalg.norm(self.velocities.Uf[0]), 8192)
        npt.assert_almost_equal(np.linalg.norm(self.velocities.Uf[1]), 16384)
        npt.assert_almost_equal(np.linalg.norm(self.velocities.Uf[2]), 24576)

    def test_gaussian_filter(self):
        """Is the Gaussian filtering correct?"""
        filtered = self.velocities.gaussian_filter(self.width)

        npt.assert_almost_equal(np.linalg.norm(filtered.U[0]), 7.388369839861693)
        npt.assert_almost_equal(np.linalg.norm(filtered.U[1]), 14.776739679723407)
        npt.assert_almost_equal(np.linalg.norm(filtered.U[2]), 22.165109519585087)
        npt.assert_almost_equal(np.linalg.norm(filtered.Uf[0]), 945.71133950229762)
        npt.assert_almost_equal(np.linalg.norm(filtered.Uf[1]), 1891.422679004595)
        npt.assert_almost_equal(np.linalg.norm(filtered.Uf[2]), 2837.134018506893)

    def test_velocity_derivative(self):
        """Is the velocity derivative correct?"""
        filtered = self.velocities.gaussian_filter(self.width)
        dudx = filtered.get_velocity_derivative(0, 0)
        dudy = filtered.get_velocity_derivative(0, 1)
        dudz = filtered.get_velocity_derivative(0, 2)
        dvdx = filtered.get_velocity_derivative(1, 0)
        dvdy = filtered.get_velocity_derivative(1, 1)
        dvdz = filtered.get_velocity_derivative(1, 2)
        dwdx = filtered.get_velocity_derivative(2, 0)
        dwdy = filtered.get_velocity_derivative(2, 1)
        dwdz = filtered.get_velocity_derivative(2, 2)

        npt.assert_almost_equal(np.linalg.norm(dudx), 14.776739679723388)
        npt.assert_almost_equal(np.linalg.norm(dudy), 29.553479359446769)
        npt.assert_almost_equal(np.linalg.norm(dudz), 59.106958718893658)
        npt.assert_almost_equal(np.linalg.norm(dvdx), 59.106958718893623)
        npt.assert_almost_equal(np.linalg.norm(dvdy), 118.21391743778726)
        npt.assert_almost_equal(np.linalg.norm(dvdz), 29.553479359446769)
        npt.assert_almost_equal(np.linalg.norm(dwdx), 177.3208761566807)
        npt.assert_almost_equal(np.linalg.norm(dwdy), 44.330219039170167)
        npt.assert_almost_equal(np.linalg.norm(dwdz), 88.660438078340349)

    def test_get_velocity_divergence(self):
        """Is the velocity divergence correct?"""

        npt.assert_allclose(
            self.tg_velocities.get_velocity_divergence(), 0.0, atol=1e-13
        )

    def test_get_interpolated_velocity(self):
        """Is the interpolation with DFT coefficients correct?"""

        # Probe random points in the domain
        Ui = np.zeros(self.xis.shape)
        for k, xi in enumerate(self.xis):

            # Get the interpolated velocity using DFT coefficients
            Ui[k, :] = self.velocities.get_interpolated_velocity(xi)

        # Tests
        npt.assert_allclose(
            Ui[:, 0],
            data.velocity_x([self.xis[:, 0], self.xis[:, 1], self.xis[:, 2]]),
            rtol=1e-10,
        )
        npt.assert_allclose(
            Ui[:, 1],
            data.velocity_y([self.xis[:, 0], self.xis[:, 1], self.xis[:, 2]]),
            rtol=1e-10,
        )
        npt.assert_allclose(
            Ui[:, 2],
            data.velocity_z([self.xis[:, 0], self.xis[:, 1], self.xis[:, 2]]),
            rtol=1e-10,
        )

    def test_numba_get_interpolated_velocity(self):
        """Is the interpolation with DFT coefficients correct (Numba version)?"""

        # Probe random points in the domain
        Ui = np.zeros(self.xis.shape)
        for k, xi in enumerate(self.xis):

            # Get the interpolated velocity using DFT coefficients
            Ui[k, :] = self.velocities.numba_get_interpolated_velocity(xi)

        # Tests
        npt.assert_allclose(
            Ui[:, 0],
            data.velocity_x([self.xis[:, 0], self.xis[:, 1], self.xis[:, 2]]),
            rtol=1e-10,
        )
        npt.assert_allclose(
            Ui[:, 1],
            data.velocity_y([self.xis[:, 0], self.xis[:, 1], self.xis[:, 2]]),
            rtol=1e-10,
        )
        npt.assert_allclose(
            Ui[:, 2],
            data.velocity_z([self.xis[:, 0], self.xis[:, 1], self.xis[:, 2]]),
            rtol=1e-10,
        )

    def test_nufft_get_interpolated_velocity(self):
        """Is the interpolation with DFT coefficients correct (NUFFT version)?"""

        # Our velocity data was generated with a real FFT. Fake the
        # data for the full FFT so we can use NUFFT.
        K = np.meshgrid(
            -self.velocities.k[0].astype(int),
            -self.velocities.k[1].astype(int),
            self.velocities.k[2][-2:0:-1].astype(int),
            indexing="ij",
        )
        Uf = [
            np.concatenate(
                (
                    self.velocities.Uf[c],
                    np.conj(self.velocities.Uf[c][K[0], K[1], K[2]]),
                ),
                axis=2,
            )
            for c in range(3)
        ]

        # Probe random points in the domain
        Ui = np.zeros(self.xis.shape)
        for c in range(3):
            Ui[:, c] = (
                np.real(
                    nufft3d2(
                        self.xis[:, 0],
                        self.xis[:, 1],
                        self.xis[:, 2],
                        np.roll(
                            np.roll(
                                np.roll(Uf[c], -int(self.velocities.N[0] / 2), 0),
                                -int(self.velocities.N[1] / 2),
                                1,
                            ),
                            -int(self.velocities.N[2] / 2),
                            2,
                        ),
                        iflag=1,
                        direct=False,
                    )
                )
                / self.velocities.N.prod()
            )

        # Tests
        npt.assert_allclose(
            Ui[:, 0],
            data.velocity_x([self.xis[:, 0], self.xis[:, 1], self.xis[:, 2]]),
            rtol=1e-10,
        )
        npt.assert_allclose(
            Ui[:, 1],
            data.velocity_y([self.xis[:, 0], self.xis[:, 1], self.xis[:, 2]]),
            rtol=1e-10,
        )
        npt.assert_allclose(
            Ui[:, 2],
            data.velocity_z([self.xis[:, 0], self.xis[:, 1], self.xis[:, 2]]),
            rtol=1e-10,
        )

    def test_energy_spectra_1D(self):
        """Is the 1D energy spectra calculation correct?"""
        spectra = self.spec_velocities.energy_spectra()

        npt.assert_array_almost_equal(
            spectra[spectra["name"] == "E00(k0)"].E,
            np.array(
                [
                    1.5524253382618400e+00,
                    1.9043965960876359e-03,
                    1.2937655879710899e-01,
                    3.2396929410351194e-03,
                    1.2641071676809387e-01,
                    3.3946004729182920e-03,
                    2.5567076106081631e-03,
                    6.1644222548702448e-03,
                    1.2444774610601403e-01,
                    1.3364807492584877e-02,
                    2.0001132399453158e-03,
                    6.8824638042840861e-04,
                    3.1330150016246905e-04,
                    1.5202649021262232e-04,
                    7.2285173753924731e-05,
                ]
            ),
            decimal=10,
        )

    def test_energy_spectra_3D(self):
        """Is the 3D energy spectra calculation correct?"""
        spectra = self.spec_velocities.energy_spectra()

        npt.assert_array_almost_equal(
            spectra[spectra["name"] == "E3D"].E,
            np.array(
                [
                    0.0000000000000000e+00,
                    7.8834014523740028e-03,
                    6.2615307999715730e-01,
                    2.1798592592670751e-02,
                    7.0884791566813754e-01,
                    1.7364781642831974e-02,
                    1.4116103213105942e-02,
                    3.3464223219289504e-02,
                    6.6559562835682240e-01,
                    5.5738971213382879e-02,
                    9.1823516469926123e-03,
                    2.9841409265312598e-03,
                    1.1186279802934285e-03,
                    4.4642379383952465e-04,
                    1.6345777560316640e-04,
                    8.3291000621897498e-05,
                ]
            ),
            decimal=10,
        )

    def test_integral_length_scale_tensor(self):
        """Is the integral length scale tensor calculation correct?"""
        lengthscales = self.spec_velocities.integral_length_scale_tensor()

        npt.assert_array_almost_equal(
            lengthscales,
            np.array(
                [
                    [2.1003226357018558, 2.100322635701855, 2.1003226357018563],
                    [2.1003226357018563, 2.1003226357018558, 2.1003226357018554],
                    [2.100322635701855, 2.1003226357018563, 2.1003226357018558],
                ]
            ),
            decimal=10,
        )

    def test_structure_functions(self):
        """Is the structure function calculation correct?"""
        structure = self.spec_velocities.structure_functions()
        npt.assert_array_almost_equal(
            structure.SL,
            np.array(
                [
                    0.,
                    0.0785096199929016,
                    0.301723786914015,
                    0.6350383967300699,
                    1.0273238570414358,
                    1.4187430801470575,
                    1.7498429304380465,
                    1.9705439637367888,
                    2.0476761798337342,
                    1.9699335870800814,
                    1.7495128845335444,
                    1.4202020757179299,
                    1.0322182020004451,
                    0.6445799848193108,
                    0.3161686585952469,
                    0.0968242917758608,
                    0.0198170012870652,
                ]
            ),
            decimal=10,
        )

        npt.assert_array_almost_equal(
            structure.ST1,
            np.array(
                [
                    0.,
                    0.3013132971454286,
                    1.0238627309018322,
                    1.7385989137636211,
                    2.0290518004781566,
                    1.7328957100929185,
                    1.0315084708761031,
                    0.3389158307627682,
                    0.0581018731482762,
                    0.3476535108824844,
                    1.0331229227730068,
                    1.7120169980026754,
                    1.9892984566194272,
                    1.70615536360969,
                    1.030285607843781,
                    0.3568253757404142,
                    0.0782862747188336,
                ]
            ),
            decimal=10,
        )

        npt.assert_array_almost_equal(
            structure.ST2,
            np.array(
                [
                    0.,
                    1.0178163823125026,
                    2.0015211367847403,
                    1.0150513353640416,
                    0.1180048334623851,
                    1.0656938951979218,
                    1.893143904464391,
                    0.9886110282366569,
                    0.2135234550866583,
                    1.0735734216745583,
                    1.8132640956728139,
                    0.9960430563200141,
                    0.275482055930023,
                    1.0543843535145818,
                    1.7709728609211628,
                    1.0232015273797213,
                    0.2969253153556487,
                ]
            ),
            decimal=10,
        )

    def test_dissipation(self):
        """Is the dissipation calculation correct?"""
        dissipation = self.spec_velocities.dissipation(0.4)

        npt.assert_almost_equal(dissipation, 51.7389774006043481)


if __name__ == "__main__":
    unittest.main()
