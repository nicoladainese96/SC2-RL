from AC_modules.Layers import *

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

debug = False

### Spatial networks: preserve screen resolution ###
class SpatialFeatures(nn.Module):
    def __init__(self, n_layers, linear_size, in_channels, n_channels, **HPs):
        super(SpatialFeatures, self).__init__()
        
        self.linear_size = linear_size # screen resolution
        
        layers =  nn.ModuleList([ResidualConvolutional(linear_size, n_channels, **HPs) for _ in range(n_layers)])
        
        self.net = nn.Sequential(
                                nn.Conv2d(in_channels, n_channels, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                *layers
                                )
        
    def forward(self, x):
        x = self.net(x)
        return x

# change name of the class and use more of that
class NonSpatialFeatures(nn.Module):
    
    def __init__(self, linear_size, n_channels, pixel_hidden_dim=128, pixel_n_residuals=4, 
                 feature_hidden_dim=64, feature_n_residuals=4):
        super(NonSpatialFeatures, self).__init__()
        
        pixel_res_layers = nn.ModuleList([ResidualLayer(linear_size**2, pixel_hidden_dim) 
                                          for _ in range(pixel_n_residuals)])
        self.pixel_res_block = nn.Sequential(*pixel_res_layers)

        self.maxpool = FeaturewiseMaxPool(pixel_axis=2)

        feature_res_layers = nn.ModuleList([ResidualLayer(n_channels, feature_hidden_dim) 
                                            for _ in range(feature_n_residuals)])
        self.feature_res_block = nn.Sequential(*feature_res_layers)
        
    def forward(self, x):
        """ Input shape (batch_dim, n_channels, linear_size, linear_size) """
        x = x.view(x.shape[0], x.shape[1],-1)
        if debug: print("x.shape: ", x.shape)
            
        x = self.pixel_res_block(x) # Interaction between pixels feature-wise
        if debug: print("x.shape: ", x.shape)
            
        x = self.maxpool(x) # Feature-wise maxpooling
        if debug: print("x.shape: ", x.shape)
            
        x = self.feature_res_block(x) # Interaction between features -> final representation
        if debug: print("x.shape: ", x.shape)
        
        return x    
    
class FullyConvSpatial(nn.Module):
    def __init__(self, in_channels, n_channels=32):
        super(FullyConvSpatial, self).__init__()
        self.net = nn.Sequential(
                                nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2),
                                nn.ReLU(),
                                nn.Conv2d(16, n_channels, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        
    def forward(self, x):
        return self.net(x)
    
class FullyConvNonSpatial(nn.Module):
    def __init__(self, n_features=256, n_channels=32, hidden_channels=64, resolution=16, kernel_size=3, stride=2):
        super(FullyConvNonSpatial, self).__init__()
        self.flatten_size = hidden_channels*((resolution-kernel_size)//stride + 1)**2 # after conv 3x3, custom stride 
        self.conv = nn.Sequential(
                                  nn.Conv2d(n_channels, hidden_channels, kernel_size=kernel_size, stride=stride),
                                  nn.ReLU()
                                 )
        
        self.net = nn.Sequential(
                                nn.Linear(self.flatten_size, n_features),
                                nn.ReLU()
                                )
    def forward(self, x):
        B = x.shape[0]
        x = self.conv(x)
        x = x.reshape(B,-1) 
        x = self.net(x)
        return x
        

class FullyConvSpatial_v1(nn.Module):
    def __init__(self, in_channels, n_channels=32, resolution=16):
        super(FullyConvSpatial_v1, self).__init__()
        self.net = nn.Sequential(
                                nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2),
                                nn.ReLU(),
                                nn.Conv2d(16, n_channels, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                ResidualConvolutional(resolution, n_channels),
                                nn.ReLU()
                                )
        
    def forward(self, x):
        return self.net(x)
    
class FullyConvNonSpatial_v1(nn.Module):
    def __init__(self, n_features=256, n_channels=32, resolution=16):
        super(FullyConvNonSpatial_v1, self).__init__()
        self.net = nn.Sequential(
                                 nn.Linear((resolution**2)*n_channels, n_features),
                                 nn.ReLU(),
                                 ResidualLayer(n_features, 64),
                                 nn.ReLU()
                                )
    def forward(self, x):
        B = x.shape[0]
        x = x.reshape(B,-1) # flatten n_channels*resolution**2
        x = self.net(x)
        return x

class FullyConvSpatial_v2(nn.Module):
    def __init__(self, in_channels, n_channels=32, resolution=32):
        super(FullyConvSpatial_v2, self).__init__()
        self.net = nn.Sequential(
                                nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2),
                                nn.LayerNorm([resolution, resolution]),
                                nn.ReLU(),
                                nn.Conv2d(16, n_channels, kernel_size=3, stride=1, padding=1),
                                nn.LayerNorm([resolution, resolution]),
                                nn.ReLU()
                                )
        
    def forward(self, x):
        return self.net(x)

class FullyConvNonSpatial_v2(nn.Module):
    def __init__(self, n_features=256, n_channels=32, resolution=16):
        super(FullyConvNonSpatial_v2, self).__init__()
        self.net = nn.Sequential(
                                 nn.Linear((resolution**2)*n_channels, n_features),
                                 nn.LayerNorm(n_features),
                                 nn.ReLU()
                                )
    def forward(self, x):
        B = x.shape[0]
        x = x.reshape(B,-1) # flatten n_channels*resolution**2
        x = self.net(x)
        return x

### NonSpatial networks: start from (batch, n_channels, size, size) and return (batch, n_features) ###
# Usually n_channels = n_features, but can be adapted

class GatedRelationalNet(nn.Module):
    def __init__(self, n_kernels=24, n_features=32, n_heads=2, n_attn_modules=4, 
                 feature_hidden_dim=64, feature_n_residuals=4, device=None):

        super(GatedRelationalNet, self).__init__()
        
        self.n_features = n_features

        MLP = nn.ModuleList([ResidualLayer(n_features, feature_hidden_dim) for _ in range(feature_n_residuals)])
        
        self.net = nn.Sequential(
            GatedRelationalModule(n_kernels, n_features, n_heads, n_attn_modules, device=device),
            FeaturewiseMaxPool(pixel_axis = 0),
            *MLP)
   
        if debug:
            print(self.net)
        
    def forward(self, x):
        x = self.net(x)
        if debug:
            print("x.shape (BoxWorldNet): ", x.shape)
        return x
    
class CategoricalNet(nn.Module):
    
    def __init__(self, n_features, size, hiddens=[256]):
        super(CategoricalNet, self).__init__()
        layers = []
        
        layers.append(nn.Linear(n_features, hiddens[0]))
        layers.append(nn.ReLU())
            
        for i in range(0,len(hiddens)-1):
            layers.append(nn.Linear(hiddens[i], hiddens[i+1]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hiddens[-1], size))
        self.net = nn.Sequential(*layers)
        
    def forward(self, state_rep):
        logits = self.net(state_rep)
        log_probs = F.log_softmax(logits, dim=(-1))
        probs = torch.exp(log_probs)
        arg = Categorical(probs).sample()
        arg = arg.detach().cpu().numpy()
        return arg.reshape(-1,1), log_probs[range(len(arg)), arg], probs

### sample spatial parameters from a matrix-like state representation

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

class SpatialParameters(nn.Module):
    
    def __init__(self, n_channels, linear_size, in_channels=2):
        super(SpatialParameters, self).__init__()
        
        self.size = linear_size
        self.conv = nn.Sequential(nn.Conv2d(n_channels, 1, kernel_size=3, stride=1, padding=1))
    
    def forward(self, x, x_first=True):
        B = x.shape[0]
        x = self.conv(x)
        x = x.reshape((x.shape[0],-1))
        log_probs = F.log_softmax(x, dim=(-1))
        probs = torch.exp(log_probs)
        index = Categorical(probs).sample()
        y, x = unravel_index(index, (self.size,self.size))
        if x_first:
            arg_lst = [[xi.item(),yi.item()] for xi, yi in zip(x,y)]
        else:
            arg_lst = [[yi.item(),xi.item()] for xi, yi in zip(x,y)]
        log_prob = log_probs[torch.arange(B), index]
        return arg_lst, log_prob, probs
    
class SpatialNet(nn.Module):
    
    def __init__(self, n_features, size=[16,16], n_channels=12):
        super(SpatialNet, self).__init__()
        
        self.size = size[0]
        
        self.linear = nn.Linear(n_features, (size[0]-6)*(size[1]-6))
        
        self.conv_block = nn.Sequential(
                                        nn.ConvTranspose2d(in_channels=1, 
                                                           out_channels=n_channels, 
                                                            kernel_size=3),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(in_channels=n_channels, 
                                                           out_channels=n_channels, 
                                                           kernel_size=3),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(in_channels=n_channels, 
                                                              out_channels=n_channels, 
                                                              kernel_size=3)
                                        )
        
    def forward(self, state_rep):
        if debug: print("state_rep.shape: ", state_rep.shape)
            
        x = F.relu(self.linear(state_rep))
        if debug: print("x.shape (after linear): ", x.shape)
            
        x = x.reshape(x.shape[0], 1, self.size-6, self.size-6)
        if debug: print("x.shape (after reshape): ", x.shape)
            
        x = self.conv_block(x)
        if debug: print("x.shape (after conv block): ", x.shape)
            
        x, _ = torch.max(x, axis=1)
        if debug: print("x.shape (after maxpool): ", x.shape)
            
        x = x.reshape(x.shape[:-2]+(-1,))
        
        log_probs = F.log_softmax(x, dim=(-1))
        if debug: 
            print("log_probs.shape: ", log_probs.shape)
            print("log_probs.shape (reshaped): ", log_probs.view(-1, self.size, self.size).shape)
        probs = torch.exp(log_probs)
        
        # assume squared space
        x_lin = torch.arange(self.size)
        xx = x_lin.repeat(self.size,1)
        args = torch.cat([xx.view(self.size,self.size,1), xx.T.view(self.size,self.size,1)], axis=2)
        args = args.reshape(-1,2)
        
        index = Categorical(probs).sample()
        if debug: 
            print("index.shape: ", index.shape)
        arg = args[index] # and this are the sampled coordinates
        print("index: ", index)
        print("arg: ", arg)
        arg = list(arg.detach().numpy())
        
        return arg, log_probs.view(self.size, self.size)[arg[0], arg[1]], probs
    
class OheNet(nn.Module):
    """Learns a vectorial state representation starting from a multi-channel image-like state."""
    
    def __init__(self, map_size, k_in=3, k_out=24, n_features=32, pixel_hidden_dim=128, 
                 pixel_n_residuals=4, feature_hidden_dim=64, feature_n_residuals=4):
        """
        map_size: int
            If input is (batch_dim, n_channels, linear_size, linear_size), then map_size = linear_size - 2
        k_in: int (default 3)
            Number of channels of the input "image"
        k_out: int (default 24)
            Number of channels at the end of the two convolutional layers
        n_features: int (default 32)
            Number of features extracted from each pixel - obtained through a linear projection of the channel axis 
            pixel-wise
        pixel_hidden_dim: int (default 128)
            Number of neurons in the hidden layers of the pixel residual layers
        pixel_n_residuals: int (default 4)
            Number of pixel residual layers
        feature_hidden_dim: int (default 64)
            Number of neurons in the hidden layers of the feature residual layers
        feature_n_residuals: int (default 4)
            Number of feature residual layers
        """
        
        super(OheNet, self).__init__()
        
        self.n_features = n_features
        
        self.OHE_conv = Convolution(k_in=k_in, k_out=k_out)
        self.pos_enc = PositionalEncoding(n_kernels=k_out, n_features=n_features)

        pixel_res_layers = nn.ModuleList([ResidualLayer(map_size**2, pixel_hidden_dim) for _ in range(pixel_n_residuals)])
        self.pixel_res_block = nn.Sequential(*pixel_res_layers)

        self.maxpool = FeaturewiseMaxPool(pixel_axis=2)

        feature_res_layers = nn.ModuleList([ResidualLayer(n_features, feature_hidden_dim) for _ in range(feature_n_residuals)])
        self.feature_res_block = nn.Sequential(*feature_res_layers)
        
    def forward(self, x):
        """ Input shape (batch_dim, k_in, map_size+2, map_size+2) """
        
        x = self.OHE_conv(x)
        if debug: print("conv_state.shape: ", x.shape)
            
        x = self.pos_enc(x)
        if debug: print("After positional enc + projection: ", x.shape)
            
        x = x.permute(1,2,0)
        if debug: print("x.shape: ", x.shape)
            
        x = self.pixel_res_block(x) # Interaction between pixels feature-wise
        if debug: print("x.shape: ", x.shape)
            
        x = self.maxpool(x) # Feature-wise maxpooling
        if debug: print("x.shape: ", x.shape)
            
        x = self.feature_res_block(x) # Interaction between features -> final representation
        if debug: print("x.shape: ", x.shape)
        
        return x     