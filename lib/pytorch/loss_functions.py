import torch
import torch.nn.functional as F

def mse_loss(y_true, y_pred):
    """
    Computes the Mean Squared Error (MSE) loss.

    The MSE loss is appropriate for regression tasks. 
    It measures the average squared differences between 
    the predicted and true values, aiming to minimize 
    the error in a quadratic manner.

    Example:
    - Predicting house prices, where the task is to 
      minimize the error between the predicted price 
      and the actual price.

    Parameters:
    y_true (torch.Tensor): The ground truth values.
    y_pred (torch.Tensor): The predicted values.

    Returns:
    torch.Tensor: The MSE loss.
    """
    return F.mse_loss(y_true, y_pred, reduction='mean')

def cross_entropy_loss(y_true, y_pred):
    """
    Computes the Cross Entropy loss.

    Cross Entropy loss is suitable for classification tasks, 
    especially when the classes are mutually exclusive. 
    It measures the performance of a classification model 
    whose output is a probability value between 0 and 1.

    Example:
    - Classifying images of cats and dogs, where the task 
      is to maximize the likelihood of the correct class.

    Parameters:
    y_true (torch.Tensor): The ground truth values.
    y_pred (torch.Tensor): The predicted values.

    Returns:
    torch.Tensor: The Cross Entropy loss.
    """
    return F.cross_entropy(y_pred, y_true)
