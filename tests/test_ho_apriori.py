# Copyright 2017 National Renewable Energy Laboratory. This software
# is released under the license detailed in the file, LICENSE, which
# is located in the top-level directory structure.

import unittest
import numpy as np
import numpy.testing as npt


import matplotlib.pyplot as plt
import scipy.ndimage as spn
import scipy.ndimage.filters as filters


class HOAprioriTestCase(unittest.TestCase):
    """Tests for `ho_apriori.py`."""

    def setUp(self):

        self.N = np.array([32, 32, 32], dtype=np.int64)
        self.L = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi])
        self.dx = self.L / self.N
        x = np.arange(0, self.L[0], self.dx[0])
        y = np.arange(0, self.L[1], self.dx[1])
        z = np.arange(0, self.L[2], self.dx[2])

        X, Y, Z = np.meshgrid(x, y, z)
        self.U = 1 * np.sin(2 * X) * np.cos(4 * Y) * np.cos(8 * Z)
        self.V = 2 * np.cos(4 * X) * np.cos(8 * Y) * np.sin(2 * Z)
        self.W = 3 * np.cos(8 * X) * np.sin(2 * Y) * np.cos(4 * Z)

        # Fourier data
        self.Uf = np.fft.rfftn(self.U)
        self.Vf = np.fft.rfftn(self.V)
        self.Wf = np.fft.rfftn(self.W)

    def test_evaluate_fft_1d(self):
        """Evaluation of 1D FFT?"""

        x = np.arange(0, self.L[0], self.dx[0])

        # Function to approximate
        def f(x): return 4 * np.sin(2 * x) - 3.3 * np.cos(4 * x)

        # Interpolation points
        mis = np.random.randn(10) * 4 * np.pi

        # Test w/ real input assumption
        print("Test with real input assumption")
        uf = np.fft.rfft(f(x))
        k = np.fft.rfftfreq(self.N[0]) * self.N[0]
        A = np.exp(2 * np.pi * 1j * np.outer(mis, k) / self.N[0])
        ui = np.real(1. / self.N[0] * np.dot(A, 2 * uf))
        for m, mi in enumerate(mis):
            xi = mi * self.dx[0]
            print("Interpolation value at {0:.6f} is {1:.6f} and exact is {2:.6f} (error={3:.6e})".format(
                xi, ui[m], f(xi), np.fabs(ui[m] - f(xi))))

    # def test_gaussian_filter(self):
    #     """Is the Gaussian filtering in spectral space correct?"""

    #     width = 4
    #     sigma = width / np.sqrt(2 * 6.0)

    #     # This is the same as ...
    #     Ufilter = filters.gaussian_filter(self.U,
    #                                       sigma,
    #                                       mode='wrap',
    #                                       truncate=6)

    #     # ... these two operations
    #     Uffilter = spn.fourier_gaussian(self.Uf,
    #                                     sigma,
    #                                     n=32)
    #     U2 = np.real(np.fft.irfftn(Uffilter))

    #     plt.figure(0)
    #     plt.pcolor(self.U[:, :, 0])
    #     plt.colorbar()

    #     plt.figure(1)
    #     plt.pcolor(Ufilter[:, :, 0])
    #     plt.colorbar()

    #     plt.figure(2)
    #     plt.pcolor(U2[:, :, 0])
    #     plt.colorbar()

    #     plt.show()

    # def test_derivative(self):
    #     """Is the derivative spectral space correct?"""

    #     kx = np.fft.fftfreq(self.N[0]) * 32
    #     ky = np.fft.fftfreq(self.N[1]) * 32
    #     kz = np.fft.rfftfreq(self.N[2]) * 32
    #     print(kx)
    #     Kx, Ky, Kz = np.meshgrid(kx, ky, kz)

    #     dudx = np.fft.irfftn(1j * Kx * self.Uf)

    #     plt.figure(0)
    #     plt.pcolor(dudx[:, :, 0])
    #     plt.colorbar()

    #     plt.figure(1)
    #     plt.pcolor(self.dudx[:, :, 0])
    #     plt.colorbar()

    #     plt.show()

        # def test_energy_spectra_1D(self):
    #     """Is the 1D energy spectra calculation correct?"""
    #     spectra = tp.energy_spectra([self.U, self.V, self.W], self.N, self.L)

    #     npt.assert_array_almost_equal(spectra[spectra['name'] == 'E00(k0)'].E,
    #                                   np.array([1.5524253382618400e+00,
    #                                             1.9043965960876359e-03,
    #                                             1.2937655879710899e-01,
    #                                             3.2396929410351194e-03,
    #                                             1.2641071676809387e-01,
    #                                             3.3946004729182920e-03,
    #                                             2.5567076106081631e-03,
    #                                             6.1644222548702448e-03,
    #                                             1.2444774610601403e-01,
    #                                             1.3364807492584877e-02,
    #                                             2.0001132399453158e-03,
    #                                             6.8824638042840861e-04,
    #                                             3.1330150016246905e-04,
    #                                             1.5202649021262232e-04,
    #                                             7.2285173753924731e-05]),
    #                                   decimal=10)

    # def test_energy_spectra_3D(self):
    #     """Is the 3D energy spectra calculation correct?"""
    #     spectra = tp.energy_spectra([self.U, self.V, self.W], self.N, self.L)

    #     npt.assert_array_almost_equal(spectra[spectra['name'] == 'E3D'].E,
    #                                   np.array([0.0000000000000000e+00,
    #                                             7.8834014523740028e-03,
    #                                             6.2615307999715730e-01,
    #                                             2.1798592592670751e-02,
    #                                             7.0884791566813754e-01,
    #                                             1.7364781642831974e-02,
    #                                             1.4116103213105942e-02,
    #                                             3.3464223219289504e-02,
    #                                             6.6559562835682240e-01,
    #                                             5.5738971213382879e-02,
    #                                             9.1823516469926123e-03,
    #                                             2.9841409265312598e-03,
    #                                             1.1186279802934285e-03,
    #                                             4.4642379383952465e-04,
    #                                             1.6345777560316640e-04,
    #                                             8.3291000621897498e-05]),
    #                                   decimal=10)

    # def test_integral_length_scale_tensor(self):
    #     """Is the integral length scale tensor calculation correct?"""
    #     lengthscales = tp.integral_length_scale_tensor(
    #         [self.U, self.V, self.W], self.N, self.L)

    #     npt.assert_array_almost_equal(lengthscales,
    #                                   np.array([[2.1003226357018558,
    #                                              2.100322635701855,
    #                                              2.1003226357018563],
    #                                             [2.1003226357018563,
    #                                              2.1003226357018558,
    #                                              2.1003226357018554],
    #                                             [2.100322635701855,
    #                                              2.1003226357018563,
    #                                              2.1003226357018558]]),
    #                                   decimal=10)

    # def test_structure_functions(self):
    #     """Is the structure function calculation correct?"""
    #     structure = tp.structure_functions(
    #         [self.U, self.V, self.W], self.N, self.L)
    #     npt.assert_array_almost_equal(structure.SL,
    #                                   np.array([0.,
    #                                             0.3013132971454286,
    #                                             1.0238627309018322,
    #                                             1.7385989137636211,
    #                                             2.0290518004781566,
    #                                             1.7328957100929185,
    #                                             1.0315084708761031,
    #                                             0.3389158307627682,
    #                                             0.0581018731482762,
    #                                             0.3476535108824844,
    #                                             1.0331229227730068,
    #                                             1.7120169980026754,
    #                                             1.9892984566194272,
    #                                             1.70615536360969,
    #                                             1.030285607843781,
    #                                             0.3568253757404142,
    #                                             0.0782862747188336]),
    #                                   decimal=10)

    #     npt.assert_array_almost_equal(structure.ST1,
    #                                   np.array([0.,
    #                                             1.0178163823125026,
    #                                             2.0015211367847403,
    #                                             1.0150513353640416,
    #                                             0.1180048334623851,
    #                                             1.0656938951979218,
    #                                             1.893143904464391,
    #                                             0.9886110282366569,
    #                                             0.2135234550866583,
    #                                             1.0735734216745583,
    #                                             1.8132640956728139,
    #                                             0.9960430563200141,
    #                                             0.275482055930023,
    #                                             1.0543843535145818,
    #                                             1.7709728609211628,
    #                                             1.0232015273797213,
    #                                             0.2969253153556487]),
    #                                   decimal=10)

    #     npt.assert_array_almost_equal(structure.ST2,
    #                                   np.array([0.,
    #                                             0.0785096199929016,
    #                                             0.301723786914015,
    #                                             0.6350383967300699,
    #                                             1.0273238570414358,
    #                                             1.4187430801470575,
    #                                             1.7498429304380465,
    #                                             1.9705439637367888,
    #                                             2.0476761798337342,
    #                                             1.9699335870800814,
    #                                             1.7495128845335444,
    #                                             1.4202020757179299,
    #                                             1.0322182020004451,
    #                                             0.6445799848193108,
    #                                             0.3161686585952469,
    #                                             0.0968242917758608,
    #                                             0.0198170012870652]),
    #                                   decimal=10)

    # def test_dissipation(self):
    #     """Is the dissipation calculation correct?"""
    #     dissipation = tp.dissipation(
    #         [self.U, self.V, self.W], self.N, self.L, 0.4)

    #     npt.assert_almost_equal(dissipation, 51.7389774006043481)


if __name__ == '__main__':
    unittest.main()
