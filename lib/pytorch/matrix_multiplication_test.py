import unittest
import torch
from .matrix_multiplication import matrix_multiplication

class TestMatrixMultiplication(unittest.TestCase):
    def test_matrix_multiplication_simple(self):
        tensor1 = torch.tensor([[1., 2.], [3., 4.]])
        tensor2 = torch.tensor([[5., 6.], [7., 8.]])
        expected_output = torch.tensor([[19., 22.], [43., 50.]])
        self.assertTrue(torch.equal(matrix_multiplication(tensor1, tensor2), expected_output))

    def test_matrix_multiplication_mismatched_dimensions(self):
        tensor1 = torch.tensor([[1., 2.], [3., 4.]])
        tensor2 = torch.tensor([[1., 2., 3.]])
        with self.assertRaises(RuntimeError):
            matrix_multiplication(tensor1, tensor2)

if __name__ == '__main__':
    unittest.main()
