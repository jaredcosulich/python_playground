import unittest
import torch
from tensor_operations import create_tensor

class TestTensorOperations(unittest.TestCase):

    def test_create_tensor(self):
        data = [[1, 2], [3, 4]]
        tensor = create_tensor(data)
        expected_tensor = torch.tensor([[1, 2], [3, 4]])
        self.assertTrue(torch.equal(tensor, expected_tensor))
        self.assertFalse(tensor.requires_grad)

    def test_create_tensor__require_grad(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        tensor = create_tensor(data, requires_grad=True)
        self.assertTrue(tensor.requires_grad)

    def test_create_tensor__specified_type(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        tensor = create_tensor(data, dtype=torch.float32)
        expected_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        self.assertTrue(torch.equal(tensor, expected_tensor))

if __name__ == '__main__':
    unittest.main()
