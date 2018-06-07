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
class DG:
    "Discontinuous Galerkin projections."

    # ========================================================================
    def __init__(self):

        self.width = None
        self.velocities = None
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
        self.velocities.gaussian_filter(self.width)
