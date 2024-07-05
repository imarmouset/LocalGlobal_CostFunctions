import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import math




class PV(nn.Module):
    # Autoencoder

    def __init__(self, input_dim, encoded_dim):
        super().__init__()
        self.input_dim = input_dim
        self.encoded_dim = encoded_dim

        self.Thal2PV = Thal2PV(input_dim, encoded_dim) # encoder
        self.PV2Pyr = PV2Pyr(encoded_dim, input_dim) # decoder

        self.reset_parameters()

    def reset_parameters(self):
        for layer in [self.Thal2PV, self.PV2Pyr]:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, input):
        encoded = self.Thal2PV(input)
        decoded = self.PV2Pyr(encoded)
        # decoded.view(-1, 1, 28, 28)

        return encoded, decoded


class SST(nn.Module):
    # Simple classifier for a global loss computation

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.SST_input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.flatten = Flatten()
        self.activation = nn.Sigmoid()

        self.hook = {'fc1': [], 'fc2': [], 'fc3': []}
        self.register_hook = False

        self.reset_parameters()


    def forward(self, input):
        input = self.flatten(input)
        fc1_out = self.activation(self.fc1(input))
        fc2_out = self.activation(self.fc2(fc1_out))
        output = self.fc3(fc2_out)

        if self.register_hook:
            fc1_out.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc1'))
            fc2_out.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc2'))
            output.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc3'))

        return output

    def reset_parameters(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            if type(layer) == nn.Linear:
                stdv = 1. / math.sqrt(layer.weight.size(1))
                layer.weight.data.uniform_(-stdv, stdv)
                layer.bias.data.uniform_(-stdv, stdv)

    def hook_fn(self, grad, name):
        self.hook[name].append(grad)

    def reset_hook(self):
        self.hook = {'fc1': [], 'fc2': [], 'fc3': []} 







class Thal2PV(nn.Module):
    # Encoder part of the autoencoder

    def __init__(self, input_dim, encoded_dim):
        super().__init__()
        self.input_dim = input_dim
        self.encoded_dim = encoded_dim

        self.flatten = Flatten()
        self.activation = nn.Sigmoid()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, encoded_dim)

        self.hook = {'fc1': [], 'fc2': [], 'fc3': []}
        self.register_hook = False

        self.reset_parameters()


    def forward(self, input):
        input = self.flatten(input)
        fc1_out = self.activation(self.fc1(input))
        fc2_out = self.activation(self.fc2(fc1_out))
        encoded = self.activation(self.fc3(fc2_out))

        if self.register_hook:
            fc1_out.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc1'))
            fc2_out.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc2'))
            encoded.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc3'))

        return encoded

    def reset_parameters(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            if type(layer) == nn.Linear:
                stdv = 1. / math.sqrt(layer.weight.size(1))
                layer.weight.data.uniform_(-stdv, stdv)
                layer.bias.data.uniform_(-stdv, stdv)

    def hook_fn(self, grad, name):
        self.hook[name].append(grad)

    def reset_hook(self):
        self.hook = {'fc1': [], 'fc2': [], 'fc3': []} 




class PV2Pyr(nn.Module):
    # Decoder part of the autoencoder

    def __init__(self, encoded_dim, decoded_dim):
        super().__init__()
        self.encoded_dim = encoded_dim
        self.decoded_dim = decoded_dim

        self.activation = nn.Sigmoid()
        self.fc1 = nn.Linear(encoded_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, decoded_dim)

        self.hook = {'fc1': [], 'fc2': [], 'fc3': []}
        self.register_hook = False

        self.reset_parameters()


    def forward(self, input):
        fc1_out = self.activation(self.fc1(input))
        fc2_out = self.activation(self.fc2(fc1_out))
        decoded = self.activation(self.fc3(fc2_out))

        if self.register_hook:
            fc1_out.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc1'))
            fc2_out.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc2'))
            decoded.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc3'))

        return decoded

    def reset_parameters(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            if type(layer) == nn.Linear:
                stdv = 1. / math.sqrt(layer.weight.size(1))
                layer.weight.data.uniform_(-stdv, stdv)
                layer.bias.data.uniform_(-stdv, stdv)

    def hook_fn(self, grad, name):
        self.hook[name].append(grad)

    def reset_hook(self):
        self.hook = {'fc1': [], 'fc2': [], 'fc3': []} 




class Flatten(nn.Module):
    def forward(self, input):
        self.input_size = input.size()
        return input.view(self.input_size[0], -1)
    
    def backward(self, input):
        pass







    













