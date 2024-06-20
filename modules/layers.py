import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import math


class OutputLayer(nn.Module):
     def __init__(self, input_dim, output_dim):
         super(OutputLayer, self).__init__()
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

     #def reset_parameters(self):
       # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
       # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
       # bound = 1 / math.sqrt(fan_in)
       # nn.init.uniform_(self.bias, -bound, bound)
    
     def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)



class HiddenLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HiddenLayer, self).__init__()
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

    #def reset_parameters(self):
       # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
       # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
       # bound = 1 / math.sqrt(fan_in)
       # nn.init_.uniform_(self.bias, -bound, bound)
    
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


