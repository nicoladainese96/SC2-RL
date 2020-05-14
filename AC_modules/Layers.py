import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

debug = False

class Convolution(nn.Module):
    """Double convolutional layer"""
    def __init__(self, k_out, k_in=3, kernel_size=2, stride=1, padding=0):
        super(Convolution, self).__init__()
        assert k_out%2 == 0, "Please provide an even number of output kernels k_out"
        layers = []
        layers.append(nn.Conv2d(k_in, k_out//2, kernel_size, stride, padding))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(k_out//2, k_out, kernel_size, stride, padding))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Accepts an input of shape (batch_size, k_in, linear_size, linear_size)
        Returns a tensor of shape (batch_size, 2*k_out, linear_size, linear_size)
        """
        if debug:
            print("x.shape (before Convolution): ", x.shape)
        if len(x.shape) <= 3:
            x = x.unsqueeze(0)
        x = self.net(x)
        if debug:
            print("x.shape (ExtractEntities): ", x.shape)
        return x
    
class PositionalEncoding(nn.Module):
    """
    Adds two extra channels to the feature dimension, indicating the spatial 
    position (x and y) of each cell in the feature map using evenly spaced values
    between −1 and 1. Then projects the feature dimension to n_features through a 
    linear layer.
    """
    def __init__(self, n_kernels, n_features):
        super(PositionalEncoding, self).__init__()
        self.projection = nn.Linear(n_kernels + 2, n_features)

    def forward(self, x):
        """
        Accepts an input of shape (batch_size, linear_size, linear_size, n_kernels)
        Returns a tensor of shape (linear_size**2, batch_size, n_features)
        """
        x = self.add_encoding2D(x)
        if debug:
            print("x.shape (After encoding): ", x.shape)
        x = x.view(x.shape[0], x.shape[1],-1)
        if debug:
            print("x.shape (Before transposing and projection): ", x.shape)
        x = self.projection(x.transpose(2,1))
        x = x.transpose(1,0)
        
        if debug:
            print("x.shape (PositionalEncoding): ", x.shape)
        return x
    
    @staticmethod
    def add_encoding2D(x):
        x_ax = x.shape[-2]
        y_ax = x.shape[-1]
        
        x_lin = torch.linspace(-1,1,x_ax)
        xx = x_lin.repeat(x.shape[0],y_ax,1).view(-1, 1, y_ax, x_ax).transpose(3,2)
        
        y_lin = torch.linspace(-1,1,y_ax).view(-1,1)
        yy = y_lin.repeat(x.shape[0],1,x_ax).view(-1, 1, y_ax, x_ax).transpose(3,2)
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
        x = torch.cat((x,xx.to(device),yy.to(device)), axis=1)
        return x
    
class FeaturewiseMaxPool(nn.Module):
    """Applies max pooling along a given axis of a tensor"""
    def __init__(self, pixel_axis):
        super(FeaturewiseMaxPool, self).__init__()
        self.max_along_axis = pixel_axis
        
    def forward(self, x):
        x, _ = torch.max(x, axis=self.max_along_axis)
        if debug:
            print("x.shape (FeaturewiseMaxPool): ", x.shape)
        return x
    
class ResidualLayer(nn.Module):
    """
    Implements residual layer. Use LayerNorm and ReLU activation before applying the layers.
    """
    def __init__(self, n_features, n_hidden):
        super(ResidualLayer, self).__init__()
        self.norm = nn.LayerNorm(n_features)
        self.w1 = nn.Linear(n_features, n_hidden)
        self.w2 = nn.Linear(n_hidden, n_features)

    def forward(self, x):
        out = F.relu(self.w1(self.norm(x)))
        out = self.w2(out)
        return out + x

class ResidualConvolutional(nn.Module):
    
    def __init__(self, linear_size, n_channels, hidden_channels=12, kernel_size=3):
        super(ResidualConvolutional, self).__init__()
        
        padding = (kernel_size - 1) // 2
        assert (kernel_size - 1) % 2 == 0, 'Provide odd kernel size to use this layer'
        
        self.net = nn.Sequential(
                                nn.LayerNorm((linear_size, linear_size)),
                                nn.Conv2d(n_channels, hidden_channels, kernel_size, stride=1, padding=padding),
                                nn.ReLU(),
                                nn.Conv2d(hidden_channels, n_channels, kernel_size, stride=1, padding=padding)
                                )
        
    def forward(self, x):
        out = self.net(x)
        out = out + x
        return out 
    
### sample spatial parameters from a matrix-like state representation

class SpatialParameters(nn.Module):
    
    def __init__(self, n_channels, linear_size):
        super(SpatialParameters, self).__init__()
        
        self.size = linear_size
        self.conv = nn.Conv2d(n_channels, 1, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape((x.shape[0],-1))
        log_probs = F.log_softmax(x, dim=(-1))
        if debug: 
            print("log_probs.shape: ", log_probs.shape)
            print("log_probs.shape (reshaped): ", log_probs.view(self.size, self.size).shape)
        probs = torch.exp(log_probs)
        
        # assume squared space
        x_lin = torch.arange(self.size)
        xx = x_lin.repeat(self.size,1)
        args = torch.cat([xx.view(self.size,self.size,1), xx.T.view(self.size,self.size,1)], axis=2)
        args = args.reshape(-1,2)
        
        distribution = Categorical(probs)
        index = distribution.sample().item() # detaching it, is it okay? maybe...
        arg = args[index] # and this are the sampled coordinates
        arg = list(arg.detach().numpy())
        
        return arg, log_probs.view(self.size, self.size)[arg[0], arg[1]], probs   