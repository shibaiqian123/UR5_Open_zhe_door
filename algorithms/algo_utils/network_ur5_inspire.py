import torch
import torch.nn as nn
import torch.nn.functional as F

import math 

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, net_cfg, proprio_shape):
        super().__init__()
        hidden_dim = net_cfg['hid_dim']
        activation = get_activation(net_cfg['activation'])
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim[0]))
        layers.append(activation)
        for l in range(len(hidden_dim)):
            if l == len(hidden_dim) - 1:
                layers.append(nn.Linear(hidden_dim[l], output_dim))
            else:
                layers.append(nn.Linear(hidden_dim[l], hidden_dim[l + 1]))
                layers.append(activation)
        self.model = nn.Sequential(*layers)

        # Initialize the weights like in stable baselines
        init_weights = [math.sqrt(2)] * len(hidden_dim)
        self.output_dim = output_dim
        if output_dim == 1:
            init_weights.append(1)
        else:
            init_weights.append(0.01)
        [torch.nn.init.orthogonal_(module.weight, gain=init_weights[idx]) for idx, module in
         enumerate(mod for mod in self.model if isinstance(mod, nn.Linear))]

    def forward(self, x):
        return self.model(x)


def conv(in_channels, out_channels, kernel_size):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)


def conv_stride(in_channels, out_channels, kernel_size, stride):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2
    )


