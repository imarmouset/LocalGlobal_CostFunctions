import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import math




class classifier(nn.Module):
    # Simple classifier for a global loss computation

    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(latent_dim, output_dim)
        self.activation = nn.Sigmoid()

        self.hook = {'fc1': []}
        self.register_hook = False

        self.reset_parameters()

    def forward(self, input):
        output = self.activation(self.fc1(input)) 

        if self.register_hook:
            output.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc1'))

        return output

    def reset_parameters(self):
        for layer in [self.fc1]:
            if type(layer) == nn.Linear:
                stdv = 1. / math.sqrt(layer.weight.size(1))
                layer.weight.data.uniform_(-stdv, stdv)
                layer.bias.data.uniform_(-stdv, stdv)

    def hook_fn(self, grad, name):
        self.hook[name].append(grad)

    def reset_hook(self):
        self.hook = {'fc1': []} 







class encoder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.flatten = Flatten()
        self.activation = nn.Sigmoid()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, latent_dim)

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



class decoder(nn.Module):

    def __init__(self, latent_dim, decoded_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoded_dim = decoded_dim

        self.activation = nn.Sigmoid()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, decoded_dim)

        self.hook = {'fc1': [], 'fc2': []}
        self.register_hook = False

        self.reset_parameters()


    def forward(self, input):
        fc1_out = self.activation(self.fc1(input))
        decoded = self.activation(self.fc2(fc1_out))

        if self.register_hook:
            fc1_out.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc1'))
            decoded.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc2'))

        return decoded

    def reset_parameters(self):
        for layer in [self.fc1, self.fc2]:
            if type(layer) == nn.Linear:
                stdv = 1. / math.sqrt(layer.weight.size(1))
                layer.weight.data.uniform_(-stdv, stdv)
                layer.bias.data.uniform_(-stdv, stdv)

    def hook_fn(self, grad, name):
        self.hook[name].append(grad)

    def reset_hook(self):
        self.hook = {'fc1': [], 'fc2': []} 




class Flatten(nn.Module):
    def forward(self, input):
        self.input_size = input.size()
        return input.view(self.input_size[0], -1)
    
    def backward(self, input):
        pass







    













