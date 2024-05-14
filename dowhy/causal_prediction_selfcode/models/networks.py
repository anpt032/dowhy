import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    

