import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import math

from modules.layers import encoder, decoder, classifier




class PV_SST_Pyr(nn.Module):
    # Network recaptitulating the PV autoencoder modalities + SST top-down modulation

    def __init__(self, thal_input_dim, latent_dim, output_dim=10):
        super().__init__()
        self.thal_input_dim = thal_input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.encoder = encoder(thal_input_dim, latent_dim)
        self.decoder = decoder(latent_dim, thal_input_dim)
        self.classifier = classifier(latent_dim, output_dim)

        self.initialise_weights()

    def initialise_weights(self):
        for layer in [self.encoder, self.decoder, self.classifier]:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, thal_input):
        encoded = self.encoder(thal_input)
        recon = self.decoder(encoded)
        output = self.classifier(encoded)
        return encoded, recon, output















