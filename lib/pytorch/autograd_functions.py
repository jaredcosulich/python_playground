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
    Analyzes the computation graph of the function f at the point x.

    Parameters:
    f (function): The function whose computation graph is to be analyzed.
    x (torch.Tensor): The point at which the computation graph is to be analyzed.

    Returns:
    torch.autograd.Function: The backward function of the last operation in the computation graph.
    """
    x.requires_grad_(True)
    y = f(x)
    return y.grad_fn
