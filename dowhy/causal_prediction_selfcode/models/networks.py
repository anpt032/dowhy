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
    

class MNIST_MLP(nn.Module):
    
    def __init__(self, input_shape):
        super(MNIST_MLP, self).__init__()
        self.hdim = hdim = 390
        self.encoder = nn.Sequential(
            nn.Linear(input_shape[0] * input_shape[1] * input_shape[2], hdim),
            nn.ReLU(True),
            nn.Linear(hdim, hdim),
            nn.ReLU(True),
        )

        self.n_outputs = hdim

        for m in self.encoder:
            if isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain("relu")
                nn.init.xavier_uniform_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.encoder(x)
    

class ResNet(nn.Module):

    def __init__(self, input_shape, resnet18=True, resnet_dropout=0.0):
        super(ResNet, self).__init__()
        if resnet18:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048

        # Adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(nc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i%3, :, :]
            
        # Save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.dropout = nn.Dropout(resnet_dropout)

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs"""
        return self.dropout(self.network(x))
    
    def train(self, mode=True):
        """
            Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, out_features),
        )
    else:
        return nn.Linear(in_features, out_features)