# Copyright 2017 National Renewable Energy Laboratory. This software
# is released under the license detailed in the file, LICENSE, which
# is located in the top-level directory structure.

# ========================================================================
#
# Imports
#
# ========================================================================
import numpy as np
import pandas as pd
from scipy import integrate as spi
from numpy.polynomial import legendre as leg  # import the Legendre functions
from nufft import nufft3d2
import time
from datetime import timedelta


# ========================================================================
#
# Class definitions
#
# ========================================================================
class FV:
    'Finite volume projections.'

    # ========================================================================
    def __init__(self, N_E, xmin, xmax):
        """
        Initialize the FV space.

        :param N_E: number of elements per side of box
        :type N: array
        :param xmin: coordinates of minimum vertex
        :type xmin: array
        :param xmax: coordinates of maximum vertex
        :type xmax: array

        """
        self.N_E = np.array([N_E[0], N_E[1], N_E[2]], dtype=np.int64)
        self.xmin = np.array([xmin[0], xmin[1], xmin[2]], dtype=np.float64)
        self.xmax = np.array([xmax[0], xmax[1], xmax[2]], dtype=np.float64)
        self.L = self.xmax - self.xmin
        self.dx = self.L / np.asarray(self.N_E, dtype=np.float64)
        self.xc = [np.linspace(self.xmin[0] + 0.5 * self.dx[0],
                               self.xmax[0] - 0.5 * self.dx[0],
                               self.N_E[0]),
                   np.linspace(self.xmin[1] + 0.5 * self.dx[1],
                               self.xmax[1] - 0.5 * self.dx[1],
                               self.N_E[1]),
                   np.linspace(self.xmin[2] + 0.5 * self.dx[2],
                               self.xmax[2] - 0.5 * self.dx[2],
                               self.N_E[2])]

        self.XC = np.meshgrid(self.xc[0],
                              self.xc[1],
                              self.xc[2],
                              indexing='ij')

        self.U = [np.zeros(self.N_E),
                  np.zeros(self.N_E),
                  np.zeros(self.N_E)]

    # ========================================================================
    def projection(self, velocities):
        """
        Project the velocity fields on the FV solution space.

        In each element, calculate:
        :math:`\\bar{U} = \\frac{1}{\\Delta x \\Delta y \\Delta z} \\int_V U(x,y,z) \mathrm{d}x \mathrm{d}y \mathrm{d}z`

        .. note::

           This function is slow because it requires many function
           evaluations to approximate the integral. Use the
           interpolation or fast_projection functions as an alternative.

        :param velocities: velocity fields
        :type velocities: Velocity

        """

        def f(z, y, x, component):
            return velocities.numba_get_interpolated_velocity([x, y, z])[
                component]

        for i in range(self.N_E[0]):
            for j in range(self.N_E[1]):
                for k in range(self.N_E[2]):

                    xi = self.XC[0][i, j, k]
                    yi = self.XC[1][i, j, k]
                    zi = self.XC[2][i, j, k]

                    lbnd = [xi - 0.5 * self.dx[0],
                            yi - 0.5 * self.dx[1],
                            zi - 0.5 * self.dx[2]]
                    rbnd = [xi + 0.5 * self.dx[0],
                            yi + 0.5 * self.dx[1],
                            zi + 0.5 * self.dx[2]]

                    for component in range(3):
                        self.U[component][i, j, k] = 1. / np.prod(self.dx) * \
                            spi.tplquad(f,
                                        lbnd[0],
                                        rbnd[0],
                                        lambda x: lbnd[1],
                                        lambda x: rbnd[1],
                                        lambda x, y: lbnd[2],
                                        lambda x, y: rbnd[2],
                                        args=(component,))[0]

    # ========================================================================
    def fast_projection(self, velocities, order=4):
        """
        Project the velocity fields on the FV solution space.

        In each element, calculate (using Gauss-Legendre quadrature):
        :math:`\\bar{U} = \\frac{1}{\\Delta x \\Delta y \\Delta z} \\int_V U(x,y,z) \mathrm{d}x \mathrm{d}y \mathrm{d}z`

        :param velocities: velocity fields
        :type velocities: Velocity
        :param order: order to be used in Gauss-Legendre quadrature
        :type order: int

        """

        xg, wg = leg.leggauss(order + 1)
        XG = np.meshgrid(xg, xg, xg, indexing='ij')
        WG = np.meshgrid(wg, wg, wg, indexing='ij')
        W3G = WG[0] * WG[1] * WG[2]
        N_G = len(xg)

        u = [np.zeros(XG[0].shape),
             np.zeros(XG[1].shape),
             np.zeros(XG[2].shape)]

        for i in range(self.N_E[0]):
            for j in range(self.N_E[1]):
                for k in range(self.N_E[2]):

                    xi = self.XC[0][i, j, k]
                    yi = self.XC[1][i, j, k]
                    zi = self.XC[2][i, j, k]

                    lbnd = [xi - 0.5 * self.dx[0],
                            yi - 0.5 * self.dx[1],
                            zi - 0.5 * self.dx[2]]
                    rbnd = [xi + 0.5 * self.dx[0],
                            yi + 0.5 * self.dx[1],
                            zi + 0.5 * self.dx[2]]

                    xloc = lbnd[0] + 0.5 * self.dx[0] * (1 + XG[0])
                    yloc = lbnd[1] + 0.5 * self.dx[1] * (1 + XG[1])
                    zloc = lbnd[2] + 0.5 * self.dx[2] * (1 + XG[2])

                    for ig in range(N_G):
                        for jg in range(N_G):
                            for kg in range(N_G):
                                u[0][ig, jg, kg], u[1][ig, jg, kg], u[2][ig, jg, kg] = velocities.numba_get_interpolated_velocity(
                                    [xloc[ig, jg, kg], yloc[ig, jg, kg], zloc[ig, jg, kg]])

                    for component in range(3):
                        self.U[component][i, j, k] = 0.5**3 * \
                            np.sum(W3G * u[component])

    # ========================================================================
    def fast_projection_nufft(self, velocities, order=4, eps=1e-13):
        """
        Project the velocity fields on the FV solution space using NUFFT library.

        In each element, calculate (using Gauss-Legendre quadrature):
        :math:`\\bar{U} = \\frac{1}{\\Delta x \\Delta y \\Delta z} \\int_V U(x,y,z) \mathrm{d}x \mathrm{d}y \mathrm{d}z`

        :param velocities: velocity fields
        :type velocities: Velocity
        :param order: order to be used in Gauss-Legendre quadrature
        :type order: int
        :param eps: tolerance for NUFFT
        :type eps: double

        """

        xg, wg = leg.leggauss(order + 1)
        XG = np.meshgrid(xg, xg, xg, indexing='ij')
        WG = np.meshgrid(wg, wg, wg, indexing='ij')
        N_G = len(xg)

        # Our velocity data was generated with a real FFT. Fake the
        # data for the full FFT so we can use NUFFT.
        K = np.meshgrid(-velocities.k[0].astype(int),
                        -velocities.k[1].astype(int),
                        velocities.k[2][-2:0:-1].astype(int),
                        indexing='ij')
        Uf = [np.concatenate((velocities.Uf[c],
                              np.conj(velocities.Uf[c][K[0],
                                                       K[1],
                                                       K[2]])),
                             axis=2) for c in range(3)]

        # Get all the quadrature node coordinates.
        # They are ordered in (x,y,z,xg,yg,zg) and flattened
        xis = np.zeros((np.prod(self.N_E) * N_G**3, 3))
        xis[:, 0] = np.add.outer(self.XC[0],
                                 0.5 * self.dx[0] * XG[0]).reshape(-1)
        xis[:, 1] = np.add.outer(self.XC[1],
                                 0.5 * self.dx[1] * XG[1]).reshape(-1)
        xis[:, 2] = np.add.outer(self.XC[2],
                                 0.5 * self.dx[2] * XG[2]).reshape(-1)

        # Use NUFFT to get the velocities
        w3g = np.reshape(WG[0] * WG[1] * WG[2], -1)
        for c in range(3):
            u = np.reshape(np.real(nufft3d2(xis[:, 0],
                                            xis[:, 1],
                                            xis[:, 2],
                                            np.roll(np.roll(np.roll(Uf[c],
                                                                    -int(velocities.N[0] / 2),
                                                                    0),
                                                            -int(velocities.N[1] / 2),
                                                            1),
                                                    -int(velocities.N[2] / 2),
                                                    2),
                                            iflag=1,
                                            eps=eps)) / velocities.N.prod(),
                           (self.N_E[0], self.N_E[1], self.N_E[2], -1))

            self.U[c] = 0.5**3 * np.sum(w3g * u, axis=-1)

    # ========================================================================
    def interpolation(self, velocities):
        """
        Interpolate the velocity fields on the FV solution space

        :param velocities: velocity fields
        :type velocities: Velocity

        """

        for i in range(self.N_E[0]):
            for j in range(self.N_E[1]):
                for k in range(self.N_E[2]):

                    xi = self.XC[0][i, j, k]
                    yi = self.XC[1][i, j, k]
                    zi = self.XC[2][i, j, k]

                    self.U[0][i, j, k], self.U[1][i, j, k], self.U[2][i, j,
                                                                      k] = velocities.numba_get_interpolated_velocity([xi, yi, zi])

    # ========================================================================
    def to_df(self):
        """
        Return the interpolated fields as a dataframe

        :return: interpolated fields
        :rtype: DataFrame

        """

        return pd.DataFrame({'x': self.XC[0].flatten(),
                             'y': self.XC[1].flatten(),
                             'z': self.XC[2].flatten(),
                             'u': self.U[0].flatten(),
                             'v': self.U[1].flatten(),
                             'w': self.U[2].flatten()})
