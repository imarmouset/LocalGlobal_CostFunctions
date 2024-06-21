import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import math


class OutputLayerGlobal(nn.Module):
     def __init__(self, input_dim, output_dim):
         super(OutputLayerGlobal, self).__init__()
         self.input_dim = input_dim
         self.output_dim = output_dim
         self.weight = Parameter(torch.Tensor(output_dim, input_dim))
         self.bias = Parameter(torch.Tensor(output_dim))
         self.reset_parameters()
                   
     def forward(self, input):
         self.e = F.linear(input, self.weight, self.bias)
         self.e = torch.sigmoid(self.e)
         #self.e = F.softmax(self.e, dim=1)
         return self.e
     
     def backward(self, input):
         pass

     def update_weight(self, weight):
         pass
   
     def extra_repr(self):
         return 'input_dim={}, output_dim={}'.format(
             self.input_dim, self.output_dim)
    
     def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)



class HiddenLayerGlobal(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HiddenLayerGlobal, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = Parameter(torch.Tensor(output_dim))
        self.reset_parameters()

    def forward(self, input):
        self.e = F.linear(input, self.weight, self.bias)
        #return torch.relu(self.e)
        return torch.sigmoid(self.e) 

    def backward(self, input):
        pass

    def update_weight(self, weight):
        pass

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
            self.input_dim, self.output_dim)
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)



class Flatten(nn.Module):
    def forward(self, input):
        self.input_size = input.size()
        return input.view(self.input_size[0], -1)
    
    def backward(self, input):
        pass



class PV(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim # thalamic input
        self.output_dim = output_dim # Pyramidal output

        self.fc1 = nn.Linear(input_dim, input_dim)
        self.activation = nn.Sigmoid()
        self.PV_Pyr = nn.Linear(input_dim, output_dim)
            
        self.hook = {'fc1': [], 'PV_Pyr': []}
        self.register_hook = False

        # self.PV_opto_mask = None
        # self.PV_scale = 0
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        PV_out  = self.activation(self.fc1(input))
        Pyr_pred = self.PV_Pyr(PV_out)

        if self.register_hook:
            PV_out.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc1'))
            Pyr_pred.register_hook(lambda grad: self.hook_fn(grad=grad,name='PV_Pyr'))
        
        return PV_out, Pyr_pred

    def hook_fn(self, grad, name):
        self.hook[name].append(grad)

    def reset_hook(self):
        self.hook = {'fc1': [], 'l23_l5': []}       



class Pyr(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, PV_modulation_factor=0.3):
        super().__init__()
        self.input_dim = input_dim # thalamus input
        self.latent_dim=latent_dim
        self.output_dim = output_dim
        self.PV_modulation_factor = PV_modulation_factor

        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.activation = nn.Sigmoid()

        self.hook = {'fc1': []}
        self.register_hook = False

    def forward(self, input, PV_pred):
        self.e = self.activation(self.fc1(input) + self.PV_modulation_factor*PV_pred.detach())
        # detach: no gradients will be computed on PV_pred during backporp 
        if self.register_hook:
            self.e.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc1'))
        return self.e
    
    def hook_fn(self, grad, name):
        self.hook[name].append(grad)

    def reset_hook(self):
        self.hook = {'fc1': []}
        
        
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim,  input_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, input_dim)
    def forward(self, x):
        self.e = self.fc1(x)
        return self.e



