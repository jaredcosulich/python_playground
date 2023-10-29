import torch

def matrix_multiplication(tensor1, tensor2):
    """
    Performs matrix multiplication of two tensors.

    Parameters:
    tensor1 (torch.Tensor): The first tensor.
    tensor2 (torch.Tensor): The second tensor.

    Returns:
    torch.Tensor: The product of tensor1 and tensor2.
    """
    return torch.mm(tensor1, tensor2)