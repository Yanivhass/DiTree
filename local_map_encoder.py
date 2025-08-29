import math
from typing import Dict, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models

from model.diffusion.conditional_unet1d import ConditionalUnet1D

"""
This module defines various encoder classes and utility functions for processing
local maps and generating embeddings. It includes implementations for ResNet-based,
MLP-based, and other custom encoders, as well as utilities for replacing submodules
in PyTorch models.
"""

def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

class ConditionalUnet1DWithLocalMap(nn.Module):
    def __init__(self, input_dim, encoder_name,embedding_dim,additional_global_cond_dim=14,
                 state_embedding_dim=128,local_map_size=None, **kwargs):
        super().__init__()
        if encoder_name == 'grid':
            self.encoder = GridEncoder()
        elif encoder_name == 'cnn':
            self.encoder = CNNEncoder()
        elif encoder_name == 'max':
            self.encoder = MaxEncoder()
        elif encoder_name == 'identity':
            self.encoder = IdentityEncoder()
        elif encoder_name == 'mlp':
            self.encoder = MLPEncoder(local_map_size ** 2, embedding_dim)
        elif encoder_name == 'resnet':
            self.encoder = ResNet18Encoder(embedding_dim)
            self.encoder = replace_bn_with_gn(self.encoder)   ## TEMP, please uncomment
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")
        self.unet = ConditionalUnet1D(input_dim, global_cond_dim=embedding_dim + additional_global_cond_dim, **kwargs)

    def forward(self, sample, local_map, timestep, global_cond=None, **kwargs):
        local_map_embedding = self.encoder(local_map)  # Process the image to get conditioning features
        if global_cond is not None:
            global_cond = torch.cat([local_map_embedding, global_cond], dim=1)
        else:
            global_cond = local_map_embedding
        output = self.unet(sample, timestep, global_cond=global_cond, **kwargs)
        return output


class ResNet18Encoder(nn.Module):
    def __init__(self, embedding_dim=9, pretrained=False):
        super().__init__()
        self.resnet18 = models.resnet18()
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, embedding_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = x.repeat(1, 3, 1, 1)  # Repeat channel to match ResNet input
        x = self.resnet18(x)
        return x



class MLPEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=9):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class GridEncoder(nn.Module):
    def __init__(self, embedding_dim=9):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.conv2 = nn.Conv2d(3, 6, 3)
        self.conv3 = nn.Conv2d(6, 4, 3)
        self.pool = nn.AdaptiveMaxPool2d((6, 6))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        return x

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        num_groups = 8
        self.conv1 = nn.Conv2d(1, 2, 3)
        # self.gn1 = nn.GroupNorm(num_groups, 2)
        self.conv2 = nn.Conv2d(2, 4, 3)
        # self.gn2 = nn.GroupNorm(num_groups, 4)
        self.conv3 = nn.Conv2d(4, 4, 3)
        # self.gn3 = nn.GroupNorm(num_groups, 8)
        self.conv4 = nn.Conv2d(4, 4, 3)
        # self.gn4 = nn.GroupNorm(num_groups, 16)

    def forward(self, x):
        x = x.unsqueeze(1)
        # x = F.mish(self.gn1(self.conv1(x)))
        # x = F.mish(self.gn2(self.conv2(x)))
        # x = F.mish(self.gn3(self.conv3(x)))
        x = F.mish(self.conv1(x))
        x = F.mish(self.conv2(x))
        x = F.mish(self.conv3(x))
        x = F.mish(self.conv4(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        return x


class MaxEncoder(nn.Module):
    def __init__(self, embedding_dim=9):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(math.floor(math.sqrt(embedding_dim)))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # x = torch.flatten(x, 1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x


class IdentityEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.flatten(x, 1)
        return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # create a random binary tensor
    x = torch.randint(0, 2, (1, 10, 10)).float()
    plt.imshow(x.squeeze(0).numpy())
    plt.show()

    # Test the MLPEncoder
    encoder = MLPEncoder(input_dim=100, embedding_dim=9)
    y = encoder(x)
    print(y.shape)
    plt.imshow(y.squeeze(0).detach().numpy().reshape(3, 3))
    plt.show()

    # Test the GridEncoder
    encoder = GridEncoder()
    y = encoder(x)
    print(y.shape)
    plt.imshow(y.squeeze(0).detach().numpy().reshape(3, 3))
    plt.show()

    # Test the MaxEncoder
    encoder = MaxEncoder()
    y = encoder(x)
    print(y.shape)
    plt.imshow(y.squeeze(0).detach().numpy().reshape(3, 3))
    plt.show()

    # Test the IdentityEncoder
    encoder = IdentityEncoder()
    y = encoder(x)
    print(y.shape)
    plt.imshow(y.squeeze(0).detach().numpy().reshape(10, 10))
    plt.show()
