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

# ========================================================================
#
# Class definitions
#
# ========================================================================


class FV:
    'Finite volume projections.'

    # ========================================================================
    def __init__(self, N_E, L, velocities):
        """
        Initialize the FV space.

        :param N_E: number of elements per side of box
        :type N: int
        :param L: size of box
        :type L: double
        :param velocities: velocity fields
        :type velocities: Velocity

        """

        self.N_E = np.array([N_E, N_E, N_E], dtype=np.int64)
        self.L = np.array([L, L, L], dtype=np.float64)
        self.dx = L / np.asarray(self.N_E, dtype=np.float64)
        self.xc = [np.linspace(0.5 * self.dx[0],
                               self.L[0] - 0.5 * self.dx[0],
                               self.N_E[0]),
                   np.linspace(0.5 * self.dx[1],
                               self.L[1] - 0.5 * self.dx[1],
                               self.N_E[1]),
                   np.linspace(0.5 * self.dx[2],
                               self.L[2] - 0.5 * self.dx[2],
                               self.N_E[2])]

        self.XC = np.meshgrid(self.xc[0],
                              self.xc[1],
                              self.xc[2],)

        self.U = [np.zeros(self.N_E),
                  np.zeros(self.N_E),
                  np.zeros(self.N_E)]

        self.velocities = velocities

    # ========================================================================
    def projection(self):
        """
        Project the velocity fields on the FV solution space.

        In each element, calculate:
        :math:`\\bar{U} = \\frac{1}{\\Delta x \\Delta y \\Delta z} \\int_V U(x,y,z) \mathrm{d}x \mathrm{d}y \mathrm{d}z`

        .. note::

           This function is slow because it requires many function
           evaluations to approximate the integral. Use the
           interpolation or fast_projection functions as an alternative.

        """

        def f(z, y, x, component):
            return self.velocities.get_interpolated_velocity([x, y, z])[component]

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
    def fast_projection(self, order=4):
        """
        Project the velocity fields on the FV solution space.

        In each element, calculate (using Gauss-Legendre quadrature):
        :math:`\\bar{U} = \\frac{1}{\\Delta x \\Delta y \\Delta z} \\int_V U(x,y,z) \mathrm{d}x \mathrm{d}y \mathrm{d}z`

        :param order: order to be used in Gauss-Legendre quadrature
        :type order: int

        """

        def f(z, y, x, component):
            return self.velocities.get_interpolated_velocity([x, y, z])[component]

        xg, wg = leg.leggauss(order + 1)
        XG = np.meshgrid(xg, xg, xg)
        WG = np.meshgrid(wg, wg, wg)
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
                                u[0][ig, jg, kg], u[1][ig, jg, kg], u[2][ig, jg, kg] = self.velocities.get_interpolated_velocity([
                                    xloc[ig, jg, kg],
                                    yloc[ig, jg, kg],
                                    zloc[ig, jg, kg]])

                    for component in range(3):
                        self.U[component][i, j, k] = 0.5**3 * \
                            np.sum(W3G * u[component])

    # ========================================================================
    def interpolation(self):
        """
        Interpolate the velocity fields on the FV solution space.

        """

        for i in range(self.N_E[0]):
            for j in range(self.N_E[1]):
                for k in range(self.N_E[2]):

                    xi = self.XC[0][i, j, k]
                    yi = self.XC[1][i, j, k]
                    zi = self.XC[2][i, j, k]

                    self.U[0][i, j, k], self.U[1][i, j, k], self.U[2][i, j,
                                                                      k] = self.velocities.get_interpolated_velocity([xi, yi, zi])

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
