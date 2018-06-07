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
import pandas as pd
import scipy.integrate as spi


# ========================================================================
#
# Class definitions
#
# ========================================================================
class Velocity(object):
    "Velocity data"

    # ========================================================================
    def __init__(self, L, U, Uf=None):
        """
        :param L: domain length
        :type L: list of floats
        :param U: velocity fields
        :type U: list of 3D arrays
        :param Uf: real FFT of velocity fields
        :type Uf: list of 3D arrays
        """
        # Initialize variables
        self.U = U
        self.L = np.array([L[0], L[1], L[2]], dtype=np.float64)
        self.N = np.array(self.U[0].shape, dtype=np.int64)
        self.dx = self.L / np.asarray(self.N, dtype=np.float64)
        self.x = [np.arange(0, self.L[c], self.dx[c]) for c in range(3)]
        self.k = [
            np.fft.fftfreq(self.N[0]) * self.N[0],
            np.fft.fftfreq(self.N[1]) * self.N[1],
            np.fft.rfftfreq(self.N[2]) * self.N[2],
        ]
        self.K = np.meshgrid(self.k[0], self.k[1], self.k[2], indexing="ij")

        if Uf is None:
            self.Uf = [np.fft.rfftn(self.U[c]) for c in range(3)]
        else:
            self.Uf = Uf

    # ========================================================================
    @classmethod
    def fromSpectralFile(cls, fname):
        """Read spectral data from numpy file.

        :param fname: file name containing the velocity data (numpy file)
        :type fname: string

        """

        # load the data
        data = np.load(fname)
        Uf = [data["uf"], data["vf"], data["wf"]]

        # Inverse fft to get spatial data
        U = [np.fft.irfftn(Uf[c]) for c in range(3)]
        return cls(data["L"], U, Uf=Uf)

    # ========================================================================
    def gaussian_filter(self, width):
        """Return the filtered velocity fields with Gaussian filter.

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
        :return: filtered velocity
        :rtype: Velocity

        """

        gamma = 6.0
        sigma = width / np.sqrt(2 * gamma)
        Ufh = [
            spn.fourier_gaussian(self.Uf[0], sigma, n=self.N[0]),
            spn.fourier_gaussian(self.Uf[1], sigma, n=self.N[0]),
            spn.fourier_gaussian(self.Uf[2], sigma, n=self.N[0]),
        ]

        # Same as spn.filters.gaussian_filter(U,sigma,mode='wrap',truncate=6)
        Uh = [np.fft.irfftn(Ufh[0]), np.fft.irfftn(Ufh[1]), np.fft.irfftn(Ufh[2])]

        return Velocity(self.L, Uh, Uf=Ufh)

    # ========================================================================
    def get_velocity_derivative(self, velocity, direction):
        """
        Return the real space velocity derivative.

        :param velocity: velocity component
        :type velocity: int
        :param direction: derivative spatial direction
        :type direction: int
        :return: derivative of velocity field in real space
        :rtype: array

        """

        return np.fft.irfftn(1j * self.K[direction] * self.Uf[velocity])

    # ========================================================================
    def get_velocity_divergence(self):
        """
        Return the real space velocity divergence.

        :return: divergence of velocity field in real space
        :rtype: array

        """

        return (
            self.get_velocity_derivative(0, 0)
            + self.get_velocity_derivative(1, 1)
            + self.get_velocity_derivative(2, 2)
        )

    # ========================================================================
    def get_interpolated_velocity(self, xi):
        """
        Return the velocity interpolated using the Fourier coefficients.

        :param xi: coordinate of point
        :type xi: array
        :return: velocities interpolated at the point
        :rtype: list

        """

        A = np.exp(
            2
            * np.pi
            * 1j
            * (
                xi[0] * self.K[0] / self.L[0]
                + xi[1] * self.K[1] / self.L[1]
                + xi[2] * self.K[2] / self.L[2]
            )
        )

        ui = np.real(2. / np.prod(self.N) * np.sum(A * self.Uf[0]))
        vi = np.real(2. / np.prod(self.N) * np.sum(A * self.Uf[1]))
        wi = np.real(2. / np.prod(self.N) * np.sum(A * self.Uf[2]))

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

        return numba_interpolation_loop(
            xi,
            self.L,
            self.N,
            self.K[0],
            self.K[1],
            self.K[2],
            self.Uf[0],
            self.Uf[1],
            self.Uf[2],
        )

    # ========================================================================
    def energy_spectra(self):
        """Get the 1D and 3D energy spectra from 3D data.

        The 1D energy spectra are defined as:

        - :math:`E_{00}(k_0) = \\frac{1}{2} \\int \\int \\phi_{ii}(k_0,k_1,k_2) \\mathrm{d}k_1 \\mathrm{d}k_2`
        - :math:`E_{11}(k_1) = \\frac{1}{2} \\int \\int \\phi_{ii}(k_0,k_1,k_2) \\mathrm{d}k_0 \\mathrm{d}k_2`
        - :math:`E_{22}(k_2) = \\frac{1}{2} \\int \\int \\phi_{ii}(k_0,k_1,k_2) \\mathrm{d}k_0 \\mathrm{d}k_1`

        The 3D energy spectrum is defined as (see Eq. 6.188 in Pope):

        - :math:`E_{3D}(k) = \\frac{1}{2} \\int_S \\phi_{ii}(k_0,k_1,k_2) \\mathrm{d}S(k)`

        where :math:`k=\\sqrt{k_0^2 + k_1^2 + k_2^2}` and
        :math:`\\phi_{ii}(k_0,k_1,k_2) = u_i u_i` (velocities
        in Fourier space) is filtered so that only valid wavenumber
        combinations are counted.

        .. note::

           For the 3D spectrum, the integral is approximated by averaging
           :math:`\\phi_{ii}(k_0,k_1,k_2)` over a binned :math:`k` and
           multiplying by the surface area of the sphere at :math:`k`. The
           bins are defined by rounding the wavenumber :math:`k` to the
           closest integer. An average of every :math:`k` in each bin is
           then used as the return value for that bin.

        :return: Dataframe of 1D and 3D energy spectra
        :rtype: dataframe

        """

        # Setup

        # Initialize empty dataframe
        df = pd.DataFrame(columns=["name", "k", "E"])

        # Our velocity data was generated with a real FFT. Fake the
        # data for the full FFT so we can get the spectra.
        Ktmp = np.meshgrid(
            -self.k[0].astype(int),
            -self.k[1].astype(int),
            self.k[2][-2:0:-1].astype(int),
            indexing="ij",
        )
        Uf = [
            np.concatenate(
                (self.Uf[c], np.conj(self.Uf[c][Ktmp[0], Ktmp[1], Ktmp[2]])), axis=2
            )
            for c in range(3)
        ]

        # Wavenumbers
        k = [
            np.fft.fftfreq(Uf[0].shape[0]) * self.N[0],
            np.fft.fftfreq(Uf[0].shape[1]) * self.N[1],
            np.fft.fftfreq(Uf[0].shape[2]) * self.N[2],
        ]
        kmax = [k[0].max(), k[1].max(), k[2].max()]
        K = np.meshgrid(k[0], k[1], k[2], indexing="ij")
        kmag = np.sqrt(K[0] ** 2 + K[1] ** 2 + K[2] ** 2)
        halfN = np.array([int(n / 2) for n in self.N], dtype=np.int64)
        kbins = np.hstack((-1e-16, np.arange(0.5, halfN[0] - 1), halfN[0] - 1))

        # Energy in Fourier space
        Ef = (
            0.5
            / (np.prod(self.N) ** 2)
            * (
                np.absolute(Uf[0]) ** 2
                + np.absolute(Uf[1]) ** 2
                + np.absolute(Uf[2]) ** 2
            )
        )

        # Filter the data with ellipsoid filter
        ellipse = (K[0] / kmax[0]) ** 2 + (K[1] / kmax[1]) ** 2 + (
            K[2] / kmax[2]
        ) ** 2 > 1.0
        Ef[ellipse] = np.nan
        K[0][ellipse] = np.nan
        K[1][ellipse] = np.nan
        K[2][ellipse] = np.nan
        kmag[ellipse] = np.nan

        # 1D spectra Eii(kj)
        for j in range(3):
            jm = (j + 1) % 3
            jn = (j + 2) % 3

            # Binning
            whichbin = np.digitize(np.abs(K[j]).flat, kbins, right=True)
            ncount = np.bincount(whichbin)

            # Average in each wavenumber bin
            E = np.zeros(len(kbins) - 2)
            for k, n in enumerate(range(1, len(kbins) - 1)):
                area = np.pi * kmax[jm] * kmax[jn] * np.sqrt(1 - (k / kmax[j]) ** 2)
                E[k] = np.mean(Ef.flat[whichbin == n]) * area
            E[E < 1e-13] = 0.0

            # Store the data
            subdf = pd.DataFrame(columns=["name", "k", "E"])
            subdf["k"] = np.arange(0, kmax[j])
            subdf["E"] = E
            subdf["name"] = "E{0:d}{0:d}(k{1:d})".format(j, j)
            df = pd.concat([df, subdf], ignore_index=True)

        # 3D spectrum

        # Multiply spectra by the surface area of the sphere at kmag.
        E3D = 4.0 * np.pi * kmag ** 2 * Ef

        # Binning
        whichbin = np.digitize(kmag.flat, kbins, right=True)
        ncount = np.bincount(whichbin)

        # Average in each wavenumber bin
        E = np.zeros(len(kbins) - 1)
        kavg = np.zeros(len(kbins) - 1)
        for k, n in enumerate(range(1, len(kbins))):
            whichbin_idx = whichbin == n
            E[k] = np.mean(E3D.flat[whichbin_idx])
            kavg[k] = np.mean(kmag.flat[whichbin_idx])
        E[E < 1e-13] = 0.0

        # Store the data
        subdf = pd.DataFrame(columns=["name", "k", "E"])
        subdf["k"] = kavg
        subdf["E"] = E
        subdf["name"] = "E3D"
        df = pd.concat([df, subdf], ignore_index=True)

        return df

    # ========================================================================
    def dissipation(self, viscosity):
        """Calculate the dissipation.

        The dissipation is defined as (see Eq. 6.160 in Pope):

        - :math:`\\epsilon = 2 \\nu \\sum_k k^2 E({\\mathbf{k}})`

        where :math:`k=\\sqrt{k_0^2 + k_1^2 + k_2^2}`.

        :param viscosity: kinematic viscosity, :math:`\\nu`
        :type viscosity: double
        :return: dissipation, :math:`\\epsilon`
        :rtype: double

        """

        # Our velocity data was generated with a real FFT. Fake the
        # data for the full FFT so we can get the spectra.
        Ktmp = np.meshgrid(
            -self.k[0].astype(int),
            -self.k[1].astype(int),
            self.k[2][-2:0:-1].astype(int),
            indexing="ij",
        )
        Uf = [
            np.concatenate(
                (self.Uf[c], np.conj(self.Uf[c][Ktmp[0], Ktmp[1], Ktmp[2]])), axis=2
            )
            for c in range(3)
        ]

        # Wavenumbers
        k = [
            np.fft.fftfreq(Uf[0].shape[0]) * self.N[0],
            np.fft.fftfreq(Uf[0].shape[1]) * self.N[1],
            np.fft.fftfreq(Uf[0].shape[2]) * self.N[2],
        ]
        K = np.meshgrid(k[0], k[1], k[2], indexing="ij")
        kmag2 = K[0] ** 2 + K[1] ** 2 + K[2] ** 2

        # Energy in Fourier space
        Ef = (
            0.5
            / (np.prod(self.N) ** 2)
            * (
                np.absolute(Uf[0]) ** 2
                + np.absolute(Uf[1]) ** 2
                + np.absolute(Uf[2]) ** 2
            )
        )

        return 2.0 * viscosity * np.sum(kmag2 * Ef)

    # ========================================================================
    def integral_length_scale_tensor(self):
        """Calculate the integral lengthscale tensor.

        :math:`L_{ij} = \\frac{1}{R_{ii}(0)} \\int_0^\\infty R_{ii}(e_j r) \\mathrm{d} r`
        where :math:`R_{ij}(\\mathbf{r}) = \\langle u_i(\\mathbf{x}) u_j(\\mathbf{x}+\\mathbf{r}) \\rangle`

        :return: Array of the lengthscale tensor, :math:`L_{ij}`
        :rtype: array

        """

        dr = self.L / self.N
        Lij = np.zeros((3, 3))
        halfN = np.array([int(n / 2) for n in self.N], dtype=np.int64)

        for i in range(3):
            for j in range(3):

                idxm = (j + 1) % 3
                idxn = (j + 2) % 3

                Uf = np.fft.rfft(self.U[i], axis=j)

                if j == 0:
                    Rii = np.sum(
                        np.fft.irfft(Uf * np.conj(Uf), axis=j)[: halfN[i] + 1, :, :],
                        axis=(idxm, idxn),
                    )
                elif j == 1:
                    Rii = np.sum(
                        np.fft.irfft(Uf * np.conj(Uf), axis=j)[:, : halfN[i] + 1, :],
                        axis=(idxm, idxn),
                    )
                elif j == 2:
                    Rii = np.sum(
                        np.fft.irfft(Uf * np.conj(Uf), axis=j)[:, :, : halfN[i] + 1],
                        axis=(idxm, idxn),
                    )
                Rii = Rii / np.prod(self.N)
                Lij[i, j] = spi.simps(Rii, dx=dr[j]) / Rii[0]

        return Lij

    # ========================================================================
    def structure_functions(self):
        """Calculate the longitudinal and transverse structure functions.

        :math:`D_{ij}(r) = \\int_V (u_i(x+r,y,z)-u_i(x,y,z)) (u_j(x+r,y,z)-u_j(x,y,z)) \\mathrm{d} V`

        and :math:`S_{L} = D_{00}`, :math:`S_{T1} = D_{11}`, :math:`S_{T2} = D_{22}`.

        :return: Dataframe of structure functions (:math:`S_{L}`, :math:`S_{T1}`, and :math:`S_{T2}`)
        :rtype: dataframe

        """

        # Get the structure functions
        halfN = np.array([int(n / 2) for n in self.N], dtype=np.int64)
        SL = np.zeros(halfN[0] + 1)
        ST1 = np.zeros(halfN[0] + 1)
        ST2 = np.zeros(halfN[0] + 1)
        for i in range(self.N[0]):
            for r in range(halfN[0] + 1):
                SL[r] += np.sum(
                    (self.U[0][(i + r) % self.N[0], :, :] - self.U[0][i, :, :]) ** 2
                )
                ST1[r] += np.sum(
                    (self.U[1][(i + r) % self.N[0], :, :] - self.U[1][i, :, :]) ** 2
                )
                ST2[r] += np.sum(
                    (self.U[2][(i + r) % self.N[0], :, :] - self.U[2][i, :, :]) ** 2
                )

        # Store the data
        df = pd.DataFrame(columns=["r", "SL", "ST1", "ST2"])
        df["r"] = self.L[0] / self.N[0] * np.arange(halfN[0] + 1)
        df["SL"] = SL / np.prod(self.N)
        df["ST1"] = ST1 / np.prod(self.N)
        df["ST2"] = ST2 / np.prod(self.N)

        return df


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

                a = np.exp(
                    2
                    * np.pi
                    * 1j
                    * (
                        xi[0] * Kx[i, j, k] / L[0]
                        + xi[1] * Ky[i, j, k] / L[1]
                        + xi[2] * Kz[i, j, k] / L[2]
                    )
                )

                ui += a * Uf[i, j, k]
                vi += a * Vf[i, j, k]
                wi += a * Wf[i, j, k]

    ui = 2. / N3 * ui
    vi = 2. / N3 * vi
    wi = 2. / N3 * wi

    return [ui.real, vi.real, wi.real]
