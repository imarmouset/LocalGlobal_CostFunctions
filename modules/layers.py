import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import math


class PyrGlobal(nn.Module):
     def __init__(self, hidden_dim, output_dim):
        super(PyrGlobal, self).__init__()
        self.hidden_dim = hidden_dim # SST top_down input
        self.output_dim=output_dim # Pyr classification of the image

        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid()

        self.reset_parameters()

        self.hook = {'fc1': []}
        self.register_hook = False
                   
     def forward(self, input):
        output = self.activation(self.fc1(input))

        if self.register_hook:
            output.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc1'))

        return output

     def reset_parameters(self):
        stdv = 1. / math.sqrt(self.fc1.weight.size(1))
        self.fc1.weight.data.uniform_(-stdv, stdv)
        self.fc1.bias.data.uniform_(-stdv, stdv)

     def hook_fn(self, grad, name):
        self.hook[name].append(grad)

     def reset_hook(self):
        self.hook = {'fc1': []}
    




class SST(nn.Module):
    def __init__(self, input_dim, topdown_dim):
        super(SST, self).__init__()
        self.input_dim = input_dim # (?) input
        self.hidden_dim = topdown_dim # sent to Pyr neurons as top-down input

        self.flatten = Flatten()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, topdown_dim)
        self.activation = nn.Sigmoid()
            
        self.hook = {'fc1': [], 'fc2': [], 'fc3': []}
        self.register_hook = False

        self.reset_parameters()

    def forward(self, input):
        input = self.flatten(input)
        fc1_out = self.activation(self.fc1(input))
        fc2_out = self.activation(self.fc2(fc1_out))
        top_down = self.activation(self.fc3(fc2_out))

        if self.register_hook:
            fc1_out.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc1'))
            fc2_out.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc2'))
            top_down.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc3'))

        return top_down

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



class PV(nn.Module):
    def __init__(self, input_dim, latent_dim, feedback_alignment = False):
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
        PV_pred = self.activation(self.fc2(PV_out))

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


class PyrLocal(nn.Module):
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
        Pyr_out = self.activation(self.fc1(input) + self.PV_modulation_factor*PV_pred)
        #Pyr_out = self.activation(self.fc1(input) + self.PV_modulation_factor*PV_pred.detach())
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



class PyrCombined(nn.Module):
    def __init__(self, thal_input_dim, latent_dim, topdown_dim, output_dim, PV_modulation_factor=0.3):
        super().__init__()
        self.thal_input_dim = thal_input_dim # Thalamus input to PV and Pyr
        self.latent_dim=latent_dim # Dim of PV prediction of Pyr cells; of output of fc1; of Pyramidal output
        self.topdown_dim = topdown_dim # SST output (top_down) input to Pyr neurons
        self.output_dim=output_dim # Pyr classification of the image      
        self.PV_modulation_factor = PV_modulation_factor # How much the PV prediction modulates the Pyr output

        self.activation = nn.Sigmoid()
        self.flatten = Flatten()
        self.fc1 = nn.Linear(topdown_dim, thal_input_dim)
        self.fc2 = nn.Linear(thal_input_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, output_dim)

        self.reset_parameters()

        self.hook = {'fc1': [], 'fc2': [], 'fc3': []}
        self.register_hook = False

    def forward(self, input_thal, PV_pred, top_down):
        # 1 - Top-down inputs into apical dendrites
        td_processed = self.activation(self.fc1(top_down))
        # 2 - Thalamic inputs into basal dendrites, sumed to the top-down processed
        input_thal = self.flatten(input_thal)
        td_thal_processed = self.activation(self.fc2(input_thal + td_processed))
        # 3 - PV_pred input into basal dendrites, sumed to thalamic and top-down inputs
        PV_input = self.PV_modulation_factor * PV_pred
        Pyr_out = self.activation(self.fc3(td_thal_processed + PV_input))

        if self.register_hook:
            td_processed.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc1'))
            td_thal_processed.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc2'))
            Pyr_out.register_hook(lambda grad: self.hook_fn(grad=grad,name='fc3'))

        # returns a classification of the image 
        return Pyr_out

    def reset_parameters(self):
        pass 

    def hook_fn(self, grad, name):
        self.hook[name].append(grad)

    def reset_hook(self):
        self.hook = {'fc1': [], 'fc2': [], 'fc3': []}



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






