# Copyright 2017 National Renewable Energy Laboratory. This software
# is released under the license detailed in the file, LICENSE, which
# is located in the top-level directory structure.

# ========================================================================
#
# Imports
#
# ========================================================================
import numpy as np
from ..constants import constants

# ========================================================================
#
# Class definitions
#
# ========================================================================


class SGS:
    "Subgrid scale data using spectral information."

    # ========================================================================
    def __init__(self):

        self.width = None
        self.velocities = None
        self.mut = None
        self.tau_sgs = None
        self.tau_sgs_kk = None
        self.constants = constants.Constants()

    # ========================================================================
    def set_filter_width(self, width):
        """
        Set the filter width.

        :param width: width of filter in :math:`\\Delta x` units, :math:`\\omega`
        :type width: double

        """
        self.width = width

    # ========================================================================
    def set_velocities(self, velocities):
        """
        Set the velocity fields and filter them.

        :param velocities: velocity fields
        :type velocities: Velocity

        """
        self.velocities = velocities
        self.filtered = self.velocities.gaussian_filter(self.width)

    # ========================================================================
    def calculate_tau_sgs(self):
        """
        Calculate the SGS shear stress using the constant Smagorinsky model.

        - :math:`\\tau_{ij} - \\frac{\\delta_{ij}}{3} \\tau_{kk} = -2 C_s^2 \\mu_t \\left(\\tilde{S}_{ij} - \\frac{\\delta_{ij}}{3} \\tilde{S}_{kk} \\right)`

        where :math:`\\mu_t = \\bar{\\rho}\\bar{\\Delta}^2 |\\tilde{S}|`, :math:`|\\tilde{S}| = \\sqrt{2 \\tilde{S}_{ij}\\tilde{S}_{ij}}`, and :math:`\\tilde{S}_{ij} = 0.5 \\left( \\frac{\\partial u_i}{\\partial x_j} +  \\frac{\\partial u_j}{\\partial x_i} \\right)`.

        """

        # Filtered velocity derivatives
        dudx = self.filtered.get_velocity_derivative(0, 0)
        dudy = self.filtered.get_velocity_derivative(0, 1)
        dudz = self.filtered.get_velocity_derivative(0, 2)
        dvdx = self.filtered.get_velocity_derivative(1, 0)
        dvdy = self.filtered.get_velocity_derivative(1, 1)
        dvdz = self.filtered.get_velocity_derivative(1, 2)
        dwdx = self.filtered.get_velocity_derivative(2, 0)
        dwdy = self.filtered.get_velocity_derivative(2, 1)
        dwdz = self.filtered.get_velocity_derivative(2, 2)
        DuDx = [[dudx, dudy, dudz], [dvdx, dvdy, dvdz], [dwdx, dwdy, dwdz]]

        # Rate of strain
        S00 = DuDx[0][0]
        S01 = 0.5 * (DuDx[0][1] + DuDx[1][0])
        S02 = 0.5 * (DuDx[0][2] + DuDx[2][0])
        S11 = DuDx[1][1]
        S12 = 0.5 * (DuDx[1][2] + DuDx[2][1])
        S22 = DuDx[2][2]
        Sij = [[S00, S01, S02], [S01, S11, S12], [S02, S12, S22]]

        Sijmag = 0.0
        for i in range(3):
            for j in range(3):
                Sijmag += Sij[i][j] ** 2
        Sijmag = np.sqrt(2.0 * Sijmag)

        Skk = S00 + S11 + S22

        # SGS shear stress
        deltabar = self.width * self.filtered.dx[0]
        self.mut = self.constants.rho * deltabar ** 2 * Sijmag

        self.tau_sgs_kk = 2.0 * self.constants.CI * self.mut * Sijmag

        tau_sgs_00 = (
            -2.0 * self.constants.Cs ** 2 * self.mut * (Sij[0][0] - Skk / 3.)
            + self.tau_sgs_kk / 3.
        )
        tau_sgs_01 = -2.0 * self.constants.Cs ** 2 * self.mut * Sij[0][1]
        tau_sgs_02 = -2.0 * self.constants.Cs ** 2 * self.mut * Sij[0][2]

        tau_sgs_11 = (
            -2.0 * self.constants.Cs ** 2 * self.mut * (Sij[1][1] - Skk / 3.)
            + self.tau_sgs_kk / 3.
        )
        tau_sgs_12 = -2.0 * self.constants.Cs ** 2 * self.mut * Sij[1][2]

        tau_sgs_22 = (
            -2.0 * self.constants.Cs ** 2 * self.mut * (Sij[2][2] - Skk / 3.)
            + self.tau_sgs_kk / 3.
        )

        self.tau_sgs = [
            [tau_sgs_00, tau_sgs_01, tau_sgs_02],
            [tau_sgs_01, tau_sgs_11, tau_sgs_12],
            [tau_sgs_02, tau_sgs_12, tau_sgs_22],
        ]
