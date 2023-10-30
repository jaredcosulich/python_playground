import torch

class SimpleSGD:
    def __init__(self, params, lr=0.01):
        """
        Initializes the SimpleSGD optimizer.

        Parameters:
        params (iterable): Iterable of parameters to optimize.
        lr (float): Learning rate.
        """
        self.params = list(params)
        self.lr = lr

    def step(self):
        """
        Performs a single optimization step (parameter update).
        """
        with torch.no_grad():
            for param in self.params:
                if param.grad is not None:
                    param -= self.lr * param.grad

    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.
        """
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

