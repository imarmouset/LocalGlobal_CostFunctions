import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from modules.layers import Decoder, PyrLocal, PV, SST, PyrGlobal, Flatten


class GlobalNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.SST = SST(input_dim, hidden_dim)
        self.PyrGlobal = PyrGlobal(hidden_dim, output_dim) 


        self._initialise_weights() 
    
    def _initialise_weights(self):
        for layer in [self.SST, self.PyrGlobal]:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, input):

        top_down = self.SST(input) # some input processed as a top-down input sent to pyramidal neurons
        output = self.PyrGlobal(top_down) # pryamidal neurons processing of the top-down input

        return top_down, output


    def reset_hook(self):
        self.SST.reset_hook()
        self.PyrGlobal.reset_hook()
    
    def register_hook(self, register=True):
        self.SST.register_hook = register
        self.PyrGlobal.register_hook = register
    
    




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
    def __init__(self, input_dim, latent_dim, PV_modulation_factor=0.3, feedback_alignment = False):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.PV = PV(input_dim, latent_dim, feedback_alignment)
        self.Pyr = PyrLocal(input_dim, latent_dim, PV_modulation_factor) 
        self.Decoder = Decoder(latent_dim, input_dim)

        self._initialise_weights() 
    
    def _initialise_weights(self):
        for layer in [self.PV, self.Pyr, self.Decoder]:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


    def forward(self, input1, input2):

        PV_out, PV_pred = self.PV(input1) # thalamic input onto PV cells: PV output (1 layer activated) and PV predictions (2nd layer)
        Pyr_out = self.Pyr(input2, PV_pred) # input is thalamic, and Pyr_pred is the last output of the PV layer
        #Pyr_out = self.Pyr(input2, PV_pred.detach())
        recon = self.Decoder(Pyr_out) # reconstruct the input from the Pyr_out --> used for teaching signal 

        return PV_out, PV_pred, Pyr_out, recon


    def reset_hook(self):
        self.PV.reset_hook()
        self.Pyr.reset_hook()
    
    def register_hook(self, register=True):
        self.PV.register_hook = register
        self.Pyr.register_hook = register
        

