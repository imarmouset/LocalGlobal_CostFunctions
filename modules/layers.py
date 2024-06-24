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
    def __init__(self, input_dim, latent_dim, feedback_alignment):
        super().__init__()
        self.input_dim = input_dim # thalamic input
        self.latent_dim = latent_dim # PV prediction of Pyr cells 

        self.flatten = Flatten()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.activation = nn.Sigmoid()
        if feedback_alignment:
            self.fc2 = LinearFA(input_dim, latent_dim)
        else: 
            self.fc2 = nn.Linear(input_dim, latent_dim)
            
        self.hook = {'fc1': [], 'fc2': []}
        self.register_hook = False
        self.reset_parameters()

        # self.PV_opto_mask = None
        # self.PV_scale = 0

    def forward(self, input):
        input = self.flatten(input)
        PV_out  = self.activation(self.fc1(input))
        PV_pred = self.fc2(PV_out)

        if self.register_hook:
            PV_out.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc1'))
            PV_pred.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc2'))

        return PV_out, PV_pred
    
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


class Pyr(nn.Module):
    def __init__(self, input_dim, latent_dim, PV_modulation_factor=0.3):
        super().__init__()
        self.input_dim = input_dim # Thalamus input
        self.latent_dim=latent_dim # PV prediction of Pyr cells, output of fc1, and Pyramidal output
        self.PV_modulation_factor = PV_modulation_factor # How much the PV prediction modulates the Pyr output

        self.flatten = Flatten()
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.activation = nn.Sigmoid()

        self.reset_parameters()

        self.hook = {'fc1': []}
        self.register_hook = False

    def forward(self, input, PV_pred):
        input = self.flatten(input)
        Pyr_out = self.activation(self.fc1(input) + self.PV_modulation_factor*PV_pred.detach())
        # detach: no gradients will be computed on PV_pred during backprop, feedback alignment to do done on PV class 

        if self.register_hook:
            Pyr_out.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc1'))

        return Pyr_out

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.fc1.weight.size(1))
        self.fc1.weight.data.uniform_(-stdv, stdv)
        self.fc1.bias.data.uniform_(-stdv, stdv)

    def hook_fn(self, grad, name):
        self.hook[name].append(grad)

    def reset_hook(self):
        self.hook = {'fc1': []}


        
class Decoder(nn.Module):
    def __init__(self, latent_dim,  input_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, input_dim)
        self.reset_parameters()

    def forward(self, x):
        self.e = self.fc1(x)
        return self.e

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.fc1.weight.size(1))
        self.fc1.weight.data.uniform_(-stdv, stdv)
        self.fc1.bias.data.uniform_(-stdv, stdv)




class LinearFunctionFA(torch.autograd.Function):
    # Taken from neoSSL shallow_mlp.py
    @staticmethod
    def forward(ctx, input, weight, bias=None, backward_weight=None, weight_masks=None):
        ctx.save_for_backward(input, weight, bias, backward_weight, weight_masks)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, backward_weight, weight_masks = ctx.saved_tensors
        backward_weight = (backward_weight * weight_masks).to(grad_output.device)
        input = input.to(grad_output.device)
        grad_input = grad_output.mm(backward_weight) # use backward_weight here instead of weight.t()
        grad_weight = grad_output.t().mm(input)
        grad_bias = grad_output.sum(0) if bias is not None else None
        return grad_input, grad_weight, grad_bias, None, None


class LinearFA(nn.Module):
    # Taken from neoSSL shallow_mlp.py
    def __init__(self, input_features, output_features, bias=False, fa_sparsity=0.):
        super(LinearFA, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # This registers the parameter with the module, making it appear in methods like `.parameters()` and `.to()`
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        # Initialize random fixed backward weights
        self.backward_weight = torch.randn(output_features, input_features)
        stdv = 1. / math.sqrt(self.backward_weight.size(1))
        self.backward_weight.data.uniform_(-stdv, stdv)

        
        # Initialize forward weights and biases with the usual initialization method
        stdv_weight = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv_weight, stdv_weight)
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        self.weight_masks = (torch.rand_like(self.backward_weight) > fa_sparsity).float()
        self.backward_weight = self.backward_weight * self.weight_masks


    def forward(self, input):
        return LinearFunctionFA.apply(input, self.weight, self.bias, self.backward_weight, self.weight_masks)
