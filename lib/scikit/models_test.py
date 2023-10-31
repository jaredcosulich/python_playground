import unittest
import numpy as np
from .models import train_linear_regression

class TestModels(unittest.TestCase):

    def test_train_linear_regression(self):
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2, 3])
        model = train_linear_regression(X, y)
        self.assertAlmostEqual(model.coef_[0], 1, places=3)

if __name__ == '__main__':
    unittest.main()