import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import math

from modules.layers import PV, SST




class PV_SST_Pyr(nn.Module):
    # Network recaptitulating the PV autoencoder modalities + SST top-down modulation

    def __init__(self, thal_input_dim, SST_input_dim, latent_dim, output_dim=10):
        super().__init__()
        self.thal_input_dim = thal_input_dim
        self.SST_input_dim = SST_input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.PV = PV(thal_input_dim, latent_dim)
        self.SST = SST(SST_input_dim, output_dim)

        self.initialise_weights()

    def initialise_weights(self):
        for layer in [self.PV, self.SST]:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, thal_input, SST_input):
        encoded, decoded = self.PV(thal_input)
        output = self.SST(SST_input)

        return encoded, decoded, output














