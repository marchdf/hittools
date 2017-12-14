# Copyright 2017 National Renewable Energy Laboratory. This software
# is released under the license detailed in the file, LICENSE, which
# is located in the top-level directory structure.

# ========================================================================
#
# Imports
#
# ========================================================================
import numpy as np
import scipy.ndimage as spn
from numba import jit, prange


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
        self.x = [np.empty((0, 0, 0), dtype=np.float64),
                  np.empty((0, 0, 0), dtype=np.float64),
                  np.empty((0, 0, 0), dtype=np.float64)]
        self.k = [np.empty((0, 0, 0), dtype=np.float64),
                  np.empty((0, 0, 0), dtype=np.float64),
                  np.empty((0, 0, 0), dtype=np.float64)]
        self.K = [np.empty((0, 0, 0), dtype=np.float64),
                  np.empty((0, 0, 0), dtype=np.float64),
                  np.empty((0, 0, 0), dtype=np.float64)]
        self.U = [np.empty((0, 0, 0), dtype=np.float64),
                  np.empty((0, 0, 0), dtype=np.float64),
                  np.empty((0, 0, 0), dtype=np.float64)]
        self.Uh = [np.empty((0, 0, 0), dtype=np.float64),
                   np.empty((0, 0, 0), dtype=np.float64),
                   np.empty((0, 0, 0), dtype=np.float64)]
        self.Uf = [np.empty((0, 0, 0), dtype=np.complex128),
                   np.empty((0, 0, 0), dtype=np.complex128),
                   np.empty((0, 0, 0), dtype=np.complex128)]
        self.Ufh = [np.empty((0, 0, 0), dtype=np.complex128),
                    np.empty((0, 0, 0), dtype=np.complex128),
                    np.empty((0, 0, 0), dtype=np.complex128)]

    # ========================================================================
    def read(self, fname):
        """Read spectral data from numpy file.

        :param fname: file name containing the velocity data (numpy file)
        :type fname: string

        """

        self.fname = fname

        # load the data
        data = np.load(self.fname)
        self.Uf = [data['uf'],
                   data['vf'],
                   data['wf']]

        # Inverse fft to get spatial data
        self.U = [np.fft.irfftn(self.Uf[0]),
                  np.fft.irfftn(self.Uf[1]),
                  np.fft.irfftn(self.Uf[2])]

        # Save off some attributes
        self.N = np.array(self.U[0].shape, dtype=np.int64)
        self.dx = self.L / np.asarray(self.N, dtype=np.float64)
        self.x = [np.arange(0, self.L[0], self.dx[0]),
                  np.arange(0, self.L[1], self.dx[1]),
                  np.arange(0, self.L[2], self.dx[2])]
        self.k = [np.fft.fftfreq(self.N[0]) * self.N[0],
                  np.fft.fftfreq(self.N[1]) * self.N[1],
                  np.fft.rfftfreq(self.N[2]) * self.N[2]]
        self.K = np.meshgrid(self.k[0], self.k[1], self.k[2])

    # ========================================================================
    def gaussian_filter(self, width):
        """Filter the velocity fields with Gaussian filter.

        Use a series of 1D Gaussian filtering filters for the
        velocity fields. The Gaussian kernel is defined as:

        - :math:`G(x-y) = \\sqrt{\\frac{\\gamma}{\\pi \\bar{\\Delta}^2}} \exp{\\left( \\frac{-\\gamma |x-y|^2}{\\bar{\\Delta}^2}\\right)}`

        where :math:`\\gamma=6`, :math:`\\bar{\\Delta} = \\omega
        \\Delta x`, and :math:`\\omega` is the filter width. Since we
        are using the SciPy filtering function, we have to adjust the
        variance parameter to ensure that we are getting the result we
        want. This filter is applied in Fourier and spatial domains.

        :param width: width of filter in :math:`\\Delta x` units, :math:`\\omega`
        :type width: double

        """

        gamma = 6.0
        sigma = width / np.sqrt(2 * gamma)
        self.Ufh = [spn.fourier_gaussian(self.Uf[0],
                                         sigma,
                                         n=self.N[0]),
                    spn.fourier_gaussian(self.Uf[1],
                                         sigma,
                                         n=self.N[0]),
                    spn.fourier_gaussian(self.Uf[2],
                                         sigma,
                                         n=self.N[0])]

        # Equivalent to spn.filters.gaussian_filter(U,sigma,mode='wrap',truncate=6)
        self.Uh = [np.fft.irfftn(self.Ufh[0]),
                   np.fft.irfftn(self.Ufh[1]),
                   np.fft.irfftn(self.Ufh[2])]

    # ========================================================================
    def get_filtered_velocity_derivative(self, velocity, direction):
        """
        Return the real space velocity derivative.

        :param velocity: velocity component
        :type velocity: int
        :param direction: derivative spatial direction
        :type direction: int
        :return: derivative of filtered velocity field in real space
        :rtype: array

        """

        return np.fft.irfftn(1j * self.K[direction] * self.Ufh[velocity])

    # ========================================================================
    def get_interpolated_velocity(self, xi):
        """
        Return the velocity interpolated using the Fourier coefficients.

        :param xi: coordinate of point
        :type xi: array
        :return: velocities interpolated at the point
        :rtype: list

        """

        A = np.exp(2 * np.pi * 1j * (xi[0] * self.K[0] / self.L[0] +
                                     xi[1] * self.K[1] / self.L[1] +
                                     xi[2] * self.K[2] / self.L[2]))

        ui = np.real(2. / np.prod(self.N)
                     * np.sum(A * self.Uf[0]))
        vi = np.real(2. / np.prod(self.N)
                     * np.sum(A * self.Uf[1]))
        wi = np.real(2. / np.prod(self.N)
                     * np.sum(A * self.Uf[2]))

        return [ui, vi, wi]

    # ========================================================================
    def numba_get_interpolated_velocity(self, xi):
        """
        Return the velocity interpolated using the Fourier coefficients.

        Calls a Numba loop

        :param xi: coordinate of point
        :type xi: array
        :return: velocities interpolated at the point
        :rtype: list

        """

        return numba_interpolation_loop(xi,
                                        self.L,
                                        self.N,
                                        self.K[0],
                                        self.K[1],
                                        self.K[2],
                                        self.Uf[0],
                                        self.Uf[1],
                                        self.Uf[2])


# ========================================================================
@jit(nopython=True, parallel=True)
def numba_interpolation_loop(xi, L, N, Kx, Ky, Kz, Uf, Vf, Wf):
    """
    Return the velocity interpolated using the Fourier coefficients.

    .. note::

       Meant to be used with Numba

    :param xi: coordinate of point
    :type xi: array
    :param L: domain length
    :type L: array
    :param N: number of points
    :type N: array
    :param Kx: x-direction wavenumbers
    :type Kx: array
    :param Ky: y-direction wavenumbers
    :type Ky: array
    :param Kz: z-direction wavenumbers
    :type Kz: array
    :param Uf: x-direction velocity
    :type Uf: array
    :param Vf: y-direction velocity
    :type Vf: array
    :param Wf: z-direction velocity
    :type Wf: array
    :return: velocities interpolated at the point
    :rtype: list

    """
    ui = 0.0
    vi = 0.0
    wi = 0.0
    N3 = N[0] * N[1] * N[2]
    for i in prange(Kx.shape[0]):
        for j in prange(Kx.shape[1]):
            for k in prange(Kx.shape[2]):

                a = np.exp(2 * np.pi * 1j * (xi[0] * Kx[i, j, k] / L[0] +
                                             xi[1] * Ky[i, j, k] / L[1] +
                                             xi[2] * Kz[i, j, k] / L[2]))

                ui += a * Uf[i, j, k]
                vi += a * Vf[i, j, k]
                wi += a * Wf[i, j, k]

    ui = 2. / N3 * ui
    vi = 2. / N3 * vi
    wi = 2. / N3 * wi

    return [ui.real, vi.real, wi.real]
