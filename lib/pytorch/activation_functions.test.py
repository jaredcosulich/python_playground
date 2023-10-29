import unittest
import torch
from activation_functions import relu_activation, sigmoid_activation, tanh_activation

class TestActivationFunctions(unittest.TestCase):

    def test_relu_activation(self):
        tensor = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        expected_output = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 2.0])
        self.assertTrue(torch.equal(relu_activation(tensor), expected_output))

    def test_sigmoid_activation(self):
        tensor = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        print(sigmoid_activation(tensor))
        expected_output = torch.tensor([0.0, 0.2689, 0.5, 0.7311, 1.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(sigmoid_activation(tensor), expected_output, atol=1e-4))

    def test_tanh_activation(self):
        tensor = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        expected_output = torch.tensor([-1.0, -0.7616, 0.0, 0.7616, 1.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(tanh_activation(tensor), expected_output, atol=1e-4))

if __name__ == '__main__':
    unittest.main()
