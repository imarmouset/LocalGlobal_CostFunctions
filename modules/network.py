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

    def __init__(self, thal_input_dim, latent_dim, output_dim=10, 
                 freeze_encoder=False, freeze_decoder=False, freeze_classifier=False):
        super().__init__()
        self.thal_input_dim = thal_input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.encoder = encoder(thal_input_dim, latent_dim)
        self.decoder = decoder(latent_dim, thal_input_dim)
        self.classifier = classifier(latent_dim, output_dim)

        if freeze_encoder:
            self.freeze_module('encoder')
        if freeze_decoder:
            self.freeze_module('decoder')
        if freeze_classifier:
            self.freeze_module('classifier')

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
        if module_name.lower() == 'encoder':
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif module_name.lower() == 'decoder':
            for param in self.decoder.parameters():
                param.requires_grad = False
        elif module_name.lower() == 'classifier':
            for param in self.classifier.parameters():
                param.requires_grad = False

    def unfreeze_module(self, module_name):
        if module_name.lower() == 'encoder':
            for param in self.encoder.parameters():
                param.requires_grad = True
        elif module_name.lower() == 'decoder':
            for param in self.decoder.parameters():
                param.requires_grad = True
        elif module_name.lower() == 'classifier':
            for param in self.classifier.parameters():
                param.requires_grad = True















