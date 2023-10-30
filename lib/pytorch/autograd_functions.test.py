import unittest
import torch
from autograd_functions import compute_gradient

class TestAutogradFunctions(unittest.TestCase):

    def test_compute_gradient(self):
        # Define a simple function f(x) = x^2
        def f(x):
            return x ** 2

        x = torch.tensor(3.0)
        gradient = compute_gradient(f, x)
        expected_gradient = torch.tensor(6.0)
        self.assertTrue(torch.equal(gradient, expected_gradient))

    def test_compute_gradient__more_complex(self):
        # Define a more_complex function f(x) = 
        def f(x):
            return (2 * x) ** 3

        x = torch.tensor(3.0)
        gradient = compute_gradient(f, x)
        expected_gradient = torch.tensor(216.0)
        self.assertTrue(torch.equal(gradient, expected_gradient))



if __name__ == '__main__':
    unittest.main()
