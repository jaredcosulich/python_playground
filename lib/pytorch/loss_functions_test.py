import unittest
import torch
from .loss_functions import mse_loss, cross_entropy_loss

class TestLossFunctions(unittest.TestCase):

    def test_mse_loss(self):
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([1.0, 2.0, 4.0])
        expected_output = torch.tensor(0.3333, dtype=torch.float32)
        self.assertTrue(torch.allclose(mse_loss(y_true, y_pred), expected_output, atol=1e-4))

    def test_cross_entropy_loss(self):
        y_true = torch.tensor([0, 1], dtype=torch.long)
        y_pred = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float32)
        expected_output = torch.tensor(0.1269, dtype=torch.float32)
        self.assertTrue(torch.allclose(cross_entropy_loss(y_true, y_pred), expected_output, atol=1e-4))

    def test_cross_entropy_loss_variation(self):
      y_true = torch.tensor([0, 1, 2], dtype=torch.long)
      y_pred = torch.tensor([[3.0, 1.0, 0.2], [1.0, 2.0, 0.1], [0.1, 1.0, 3.0]], dtype=torch.float32)
      expected_output = torch.tensor(0.2568, dtype=torch.float32)  
      print(cross_entropy_loss(y_true, y_pred))
      self.assertTrue(torch.allclose(cross_entropy_loss(y_true, y_pred), expected_output, atol=1e-4))


if __name__ == '__main__':
    unittest.main()
