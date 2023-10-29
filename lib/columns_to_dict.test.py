# Filename: test_mapping.py

import unittest
from columns_to_dict import columns_to_dict

class TestCreateMapping(unittest.TestCase):

    def test_simple(self):
        dict = columns_to_dict([1, 2, 3], ['x', 'y', 'z'])
        self.assertEqual(dict, {1: 'x', 2: 'y', 3: 'z'})

    def test_different_lengths_col_1(self):
        dict = columns_to_dict([1, 2], ['x', 'y', 'z'])
        self.assertEqual(dict, {1: 'x', 2: 'y'})

    def test_different_lengths_col_2(self):
        dict = columns_to_dict([1, 2, 3], ['a', 'b'])
        self.assertEqual(dict, {1: 'a', 2: 'b'})

if __name__ == '__main__':
    unittest.main()
