import unittest
import torch
from tensor_operations import create_tensor, perform_operations

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

    def test_perform_operations(self):
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        result = perform_operations(tensor)
        expected_result = {
            'indexed': torch.tensor([1, 2, 3]),
            'sliced': torch.tensor([[1, 2], [4, 5]]),
            'joined': torch.tensor([[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]]),
        }
        for key in expected_result:
            self.assertTrue(torch.equal(result[key], expected_result[key]))

if __name__ == '__main__':
    unittest.main()
