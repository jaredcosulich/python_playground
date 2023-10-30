import unittest
import torch
from .optimization import SimpleSGD

class TestOptimization(unittest.TestCase):

    def test_simple_sgd(self):
        # Define a simple quadratic loss function f(x) = (x - 3)^2
        def loss_fn(x):
            return (x - 3) ** 2

        x = torch.tensor(0.0, requires_grad=True)
        optimizer = SimpleSGD([x], lr=0.1)

        # Perform 50 optimization steps
        for _ in range(50):
            loss = loss_fn(x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # x should be close to 3 after optimization
        self.assertAlmostEqual(x.item(), 3, places=1)

    def test_multi_dimensional_parameters(self):
        # Define a simple quadratic loss function f(W) = ||W - target||^2
        target = torch.tensor([[3, 3], [3, 3]], dtype=torch.float32)
        def loss_fn(W):
            return torch.sum((W - target)**2)

        W = torch.zeros(2, 2, requires_grad=True)
        optimizer = SimpleSGD([W], lr=0.1)

        for _ in range(50):
            loss = loss_fn(W)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        self.assertTrue(torch.allclose(W, target, atol=1e-1))

    def test_multiple_parameters(self):
        # Defining a simple loss function f(x, y) = (x - 3)^2 + (y - 4)^2
        def loss_fn(x, y):
            return (x - 3)**2 + (y - 4)**2

        x = torch.tensor(0.0, requires_grad=True)
        y = torch.tensor(0.0, requires_grad=True)
        optimizer = SimpleSGD([x, y], lr=0.1)

        for _ in range(50):
            loss = loss_fn(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        self.assertAlmostEqual(x.item(), 3, places=1)
        self.assertAlmostEqual(y.item(), 4, places=1)

    def test_zero_gradient(self):
        x = torch.tensor(3.0, requires_grad=True)  # At this point, gradient is zero
        optimizer = SimpleSGD([x], lr=0.1)

        loss = (x - 3)**2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(x.grad)
        self.assertEqual(x.grad.item(), 0)
        self.assertAlmostEqual(x.item(), 3, places=1)  # x should remain unchanged


if __name__ == '__main__':
    unittest.main()
