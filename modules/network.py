import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from modules.layers import HiddenLayerGlobal, OutputLayerGlobal, Flatten
from modules.layers import Decoder, Pyr, PV


class NeuralNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_hidden_layers, n_hidden_units):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units

        self.feature_layers = []
        self.feature_layers.append(Flatten())

        self.classification_layers = []
        if n_hidden_layers == 0:
            self.classification_layers.append(OutputLayerGlobal(n_inputs, n_outputs))
        elif n_hidden_layers == 1:
            self.classification_layers.append(HiddenLayerGlobal(n_inputs, n_hidden_units))
            self.classification_layers.append(OutputLayerGlobal(n_hidden_units, n_outputs))
        elif n_hidden_layers > 1:
            self.classification_layers.append(HiddenLayerGlobal(n_inputs, n_hidden_units))
            for i in range(1, n_hidden_layers):
                self.classification_layers.append(HiddenLayerGlobal(n_hidden_units, n_hidden_units))
            self.classification_layers.append(OutputLayerGlobal(n_hidden_units, n_outputs))

        self.layers = nn.Sequential(*(self.feature_layers + self.classification_layers))
        self._initialise_weights() 

    def forward(self, x):
        return self.layers(x)
    
    def _initialise_weights(self):
        for layer in self.classification_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def loss(self, output, target):
        return F.mse_loss(output, target) 
    
    def MSE_loss(self, output, target): # same result than the F.mse_loss(output, target)
        l = 0
        for i in range(len(output)):
            diff = (output[i] - target[i])**2
            l += diff
        l = float((sum(l)/len(output))/len(l))
        return l
    
    def backward(self, target):
        # Call the backward method of the output layer with target as the argument
        burst_rate, event_rate, feedback_bp, feedback_fa = self.classification_layers[-1].backward(target)
        # In each iteration, it calls the backward method of the current layer with the values 
        # obtained from the previous layer's backward pass. These values are updated with 
        # the new outputs of the current layer's backward method.
        for i in range(len(self.classification_layers) - 2, -1, -1):
            burst_rate, event_rate, feedback_bp, feedback_fa = self.classification_layers[i].backward(burst_rate,
                                                                                                      event_rate,
                                                                                                      feedback_bp,
                                                                                                      feedback_fa)





class Net(nn.Module): # works 
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    

    


class LocalNetwork(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, l23_modulation_factor=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim

        self.PV = PV(input_dim, output_dim)
        self.Pyr = Pyr(input_dim, latent_dim, output_dim, l23_modulation_factor)
        self.decoder = Decoder(latent_dim, input_dim)
        
