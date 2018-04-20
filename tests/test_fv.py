# Copyright 2017 National Renewable Energy Laboratory. This software
# is released under the license detailed in the file, LICENSE, which
# is located in the top-level directory structure.

import os
import unittest
import numpy as np
import numpy.testing as npt
import hittools.velocity.velocity as velocity
import hittools.data.data as data
import hittools.fv.fv as fv


class FVTestCase(unittest.TestCase):
    """Tests for `fv.py`."""

    def setUp(self):
        parent = os.path.abspath(os.path.join(__file__, '../..'))
        self.fname = os.path.abspath(os.path.join(
            parent, 'hittools', 'data', 'toy_data.npz'))

        # Use the data class to create and output toy data
        self.data = data.Data()
        self.data.output_data(self.fname)

        # Load the velocity fields from the toy data
        self.velocities = velocity.Velocity.fromSpectralFile(self.fname)

        # Define FV solution space
        self.fv = fv.FV([2, 2, 2],
                        [0, 0, 0],
                        [np.pi / 8., np.pi / 8., np.pi / 8.])

    def test_projection(self):
        """Is the FV projection correct?"""

        self.fv.projection(self.velocities)
        npt.assert_array_almost_equal(self.fv.U[0],
                                      np.array([[[0.1111007024614374, -0.1111007024614374],
                                                 [0.0460194177487053, -0.0460194177487052]],
                                                [[0.3163880325649341, -0.3163880325649344],
                                                 [0.1310522140609364, -0.1310522140609361]]]),
                                      decimal=13)

    def test_fast_projection(self):
        """Is the FV fast projection correct?"""

        self.fv.fast_projection(self.velocities)
        npt.assert_array_almost_equal(self.fv.U[0],
                                      np.array([[[0.1111007024658372, -0.1111007024658371],
                                                 [0.0460194177505278, -0.0460194177505277]],
                                                [[0.3163880325774635, -0.3163880325774637],
                                                 [0.1310522140661262, -0.131052214066126]]]),
                                      decimal=13)

    def test_fast_projection_nufft(self):
        """Is the FV fast projection with NUFFT correct?"""

        self.fv.fast_projection_nufft(self.velocities)
        npt.assert_array_almost_equal(self.fv.U[0],
                                      np.array([[[0.1111007024658372, -0.1111007024658371],
                                                 [0.0460194177505278, -0.0460194177505277]],
                                                [[0.3163880325774635, -0.3163880325774637],
                                                 [0.1310522140661262, -0.131052214066126]]]),
                                      decimal=13)

    def test_interpolation(self):
        """Is the FV interpolation correct?"""

        self.fv.interpolation(self.velocities)
        npt.assert_array_almost_equal(self.fv.U[0],
                                      np.array([[[0.1274488947760401, -0.1274488947760397],
                                                 [0.0527910607256974, -0.052791060725697]],
                                                [[0.3629437454255757, -0.3629437454255763],
                                                 [0.1503362217337619, -0.1503362217337616]]]),
                                      decimal=13)


if __name__ == '__main__':
    unittest.main()
