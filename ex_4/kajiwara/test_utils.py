import unittest

import numpy as np

from modules.utils import get_framing_data

class TestUtils(unittest.TestCase):
    def test_get_framing_data(self):
        test_list = [i for i in range(10)]
        want = np.array([[0,1,2,3], [2,3,4,5], [4,5,6,7], [6,7,8,9]])
        res = get_framing_data(test_list, 4, 0.5)
        self.assertEqual(res.tolist(), want.tolist())