import unittest
import torch
from .nn_blocks import SimpleLinear

class TestNNBlocks(unittest.TestCase):

    def test_simple_linear(self):
        layer = SimpleLinear(5, 3)  # 5 input features, 3 output features
        x = torch.randn(10, 5)  # 10 samples, 5 features per sample
        y = layer(x)
        self.assertEqual(y.size(), (10, 3))  # Check output size

if __name__ == '__main__':
    unittest.main()
