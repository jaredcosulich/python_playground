import torch
import torch.nn.functional as F

def relu_activation(tensor):
    """
    Applies the ReLU activation function to a tensor.

    Parameters:
    tensor (torch.Tensor): The input tensor.

    Returns:
    torch.Tensor: The tensor after applying ReLU.
    """
    return F.relu(tensor)

def sigmoid_activation(tensor):
    """
    Applies the sigmoid activation function to a tensor.

    Parameters:
    tensor (torch.Tensor): The input tensor.

    Returns:
    torch.Tensor: The tensor after applying sigmoid.
    """
    return torch.sigmoid(tensor)

def tanh_activation(tensor):
    """
    Applies the tanh activation function to a tensor.

    Parameters:
    tensor (torch.Tensor): The input tensor.

    Returns:
    torch.Tensor: The tensor after applying tanh.
    """
    return torch.tanh(tensor)
