# Copyright 2017 National Renewable Energy Laboratory. This software
# is released under the license detailed in the file, LICENSE, which
# is located in the top-level directory structure.

import unittest
import numpy as np
import numpy.testing as npt


class HITToolsTestCase(unittest.TestCase):
    """Tests for `hittools.py`."""

    def setUp(self):

        self.N = np.array([32, 32, 32], dtype=np.int64)


if __name__ == "__main__":
    unittest.main()
