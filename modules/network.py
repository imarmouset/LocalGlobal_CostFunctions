import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import math

from modules.layers import encoder, decoder, classifier




class PV_SST_Pyr(nn.Module):
    # Network recaptitulating the PV autoencoder modalities 

    def __init__(self, thal_input_dim, latent_dim, output_dim=10, 
                 freeze=None, defreeze=None, swap_digits=None):
        super().__init__()
        self.thal_input_dim = thal_input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.swap_digits = swap_digits

        self.encoder = encoder(thal_input_dim, latent_dim)
        self.decoder = decoder(latent_dim, thal_input_dim)
        self.classifier = classifier(latent_dim, output_dim)

        if freeze is not None:
            self.freeze_module(freeze)
        if defreeze is not None:
            self.defreeze_module(defreeze)

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
    
    def freeze_module(self, module_name):
        if module_name in ['encoder', 'decoder', 'classifier']:
            for param in getattr(self, module_name).parameters():
                param.requires_grad = False
            print(f"{module_name} has been frozen.")
        else:
            raise ValueError(f"Invalid module name: {module_name}. "
                             f"Must be 'encoder', 'decoder', or 'classifier'.")

    def unfreeze_module(self, module_name):
        if module_name in ['encoder', 'decoder', 'classifier']:
            for param in getattr(self, module_name).parameters():
                param.requires_grad = True
            print(f"{module_name} has been unfrozen.")
        else:
            raise ValueError(f"Invalid module name: {module_name}. "
                             f"Must be 'encoder', 'decoder', or 'classifier'.")














