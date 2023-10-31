import torch
import torch.nn as nn

class SimpleLinear(nn.Module):
    def __init__(self, in_features, out_features):
        """
        Initializes the SimpleLinear layer.

        Parameters:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        """
        super(SimpleLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=0)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / torch.sqrt(torch.tensor(fan_in).float())
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Forward pass through the layer.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor.
        """
        return torch.nn.functional.linear(x, self.weight, self.bias)
