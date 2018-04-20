# Copyright 2017 National Renewable Energy Laboratory. This software
# is released under the license detailed in the file, LICENSE, which
# is located in the top-level directory structure.

import os
import unittest
import numpy as np
import numpy.testing as npt
import hit_tools.velocity.velocity as velocity
import hit_tools.sgs.sgs as sgs


class SGSTestCase(unittest.TestCase):
    """Tests for `sgs.py`."""

    def setUp(self):
        parent = os.path.abspath(os.path.join(__file__, '../..'))
        self.fname = os.path.abspath(os.path.join(
            parent, 'hit_tools', 'data', 'toy_data.npz'))
        self.velocities = velocity.Velocity.fromSpectralFile(self.fname)
        self.width = 4
        self.sgs = sgs.SGS()
        self.sgs.set_filter_width(self.width)
        self.sgs.set_velocities(self.velocities)

    def test_calculate_tau_sgs(self):
        """Is the SGS shear stress calculation correct?"""

        self.sgs.calculate_tau_sgs()

        npt.assert_almost_equal(np.linalg.norm(self.sgs.tau_sgs_kk),
                                71.693882439894921)
        npt.assert_almost_equal(np.linalg.norm(self.sgs.tau_sgs[0][0]),
                                24.140172881732443)
        npt.assert_almost_equal(np.linalg.norm(self.sgs.tau_sgs[0][1]),
                                1.5891197503366674)
        npt.assert_almost_equal(np.linalg.norm(self.sgs.tau_sgs[0][2]),
                                6.9760844586474011)
        npt.assert_almost_equal(np.linalg.norm(self.sgs.tau_sgs[1][0]),
                                np.linalg.norm(self.sgs.tau_sgs[0][1]))
        npt.assert_almost_equal(np.linalg.norm(self.sgs.tau_sgs[1][1]),
                                24.658538778746841)
        npt.assert_almost_equal(np.linalg.norm(self.sgs.tau_sgs[1][2]),
                                1.1195528607166192)
        npt.assert_almost_equal(np.linalg.norm(self.sgs.tau_sgs[0][2]),
                                np.linalg.norm(self.sgs.tau_sgs[2][0]))
        npt.assert_almost_equal(np.linalg.norm(self.sgs.tau_sgs[1][2]),
                                np.linalg.norm(self.sgs.tau_sgs[2][1]))
        npt.assert_almost_equal(np.linalg.norm(self.sgs.tau_sgs[2][2]),
                                24.315509250778007)
