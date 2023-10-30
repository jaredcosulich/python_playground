import torch

def compute_gradient(f, x):
    """
    Computes the gradient of the function f at the point x.

    Parameters:
    f (function): The function whose gradient is to be computed.
    x (torch.Tensor): The point at which the gradient is to be computed.

    Returns:
    torch.Tensor: The gradient of f at the point x.
    """
    x.requires_grad_(True)
    y = f(x)
    y.backward()
    return x.grad

def analyze_graph(f, x):
    """
    Analyzes the computation graph of the function f at the point x,
    and returns the gradients at each operation.

    Parameters:
    f (function): The function whose computation graph is to be analyzed.
    x (torch.Tensor): The point at which the computation graph is to be analyzed.

    Returns:
    dict: A dictionary containing gradients at each operation in the computation graph.
    """
    x.requires_grad_(True)
    y = f(x)

    gradients = {}

    def save_grad(name):
        def hook(grad):
            gradients[name] = grad.clone()
        return hook

    # Register hooks to capture gradients
    x.register_hook(save_grad('x'))
    y.register_hook(save_grad('y'))

    y.backward()
    
    return {name: grad.item() for name, grad in gradients.items() if grad is not None}
