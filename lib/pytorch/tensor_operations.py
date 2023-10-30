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


def perform_operations(tensor):
    """
    Performs various tensor operations: indexing, slicing, joining, and mutating.

    Parameters:
    tensor (torch.Tensor): The input tensor.

    Returns:
    dict: A dictionary containing the resulting tensors from various operations.
    """
    result = {
        'indexed': tensor[0, :],        # Indexing [[1, 2, 3], [4, 5, 6]] -> [1, 2, 3]
        'sliced': tensor[:, :2],        # Slicing [[1, 2, 3], [4, 5, 6]] -> [[1, 2], [4, 5]]
        'joined': torch.cat(
            (tensor, tensor), dim=0
        ),  # Joining [[1, 2, 3], [4, 5, 6]] -> [[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]]
    }
    return result


def transfer_tensor(tensor, device):
    """
    Transfers a tensor to the specified device.

    Parameters:
    tensor (torch.Tensor): The input tensor.
    device (torch.device): The target device.

    Returns:
    torch.Tensor: The tensor on the target device.
    """
    return tensor.to(device)
