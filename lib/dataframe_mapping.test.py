# Filename: test_mapping.py

import unittest
import pandas as pd
from dataframe_mapping import create_mapping

class TestCreateMapping(unittest.TestCase):

    def test_create_mapping(self):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        mapping = create_mapping(df, 'A', 'B')
        self.assertEqual(mapping, {1: 'x', 2: 'y', 3: 'z'})

    def test_non_existent_columns(self):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        with self.assertRaises(ValueError) as context:
            create_mapping(df, 'C', 'D')
        self.assertTrue("Columns C or D not found in DataFrame" in str(context.exception))

    def test_duplicate_keys(self):
        df = pd.DataFrame({'A': [1, 2, 2], 'B': ['x', 'y', 'z']})
        mapping = create_mapping(df, 'A', 'B')
        self.assertEqual(mapping, {1: 'x', 2: 'z'})  # Note: only the last occurrence is kept

if __name__ == '__main__':
    unittest.main()
