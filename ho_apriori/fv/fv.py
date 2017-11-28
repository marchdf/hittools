# Copyright 2017 National Renewable Energy Laboratory. This software
# is released under the license detailed in the file, LICENSE, which
# is located in the top-level directory structure.

# ========================================================================
#
# Imports
#
# ========================================================================
import numpy as np
from scipy import integrate as spi


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
           interpolation function as an alternative.

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

                    for component in range(3):
                        self.U[component][i, j, k] = self.velocities.get_interpolated_velocity([xi, yi, zi])[
                            component]
