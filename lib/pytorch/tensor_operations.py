import torch

def create_tensor(data, dtype=None, device=None, requires_grad=False):
    """
    Creates a tensor from the provided data.

    Parameters:
    data (array-like): The data to create the tensor from.
    dtype (torch.dtype, optional): The desired data type of the tensor.
    device (torch.device, optional): The desired device of the tensor.
    requires_grad (bool, optional): Whether the tensor requires gradients.

    Returns:
    torch.Tensor: The created tensor.
    """
    return torch.tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
