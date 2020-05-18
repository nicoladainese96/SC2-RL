from AC_modules.Layers import *

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

debug = True

class SpatialFeatures(nn.Module):
    def __init__(self, n_layers, linear_size, in_channels, n_channels, **HPs):
        super(SpatialFeatures, self).__init__()
        
        self.linear_size = linear_size # screen resolution
        
        layers =  nn.ModuleList([ResidualConvolutional(linear_size, n_channels, **HPs) for _ in range(n_layers-1)])
        
        self.net = nn.Sequential(
                                nn.Conv2d(in_channels, n_channels, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                *layers
                                )
        
    def forward(self, x):
        x = self.net(x)
        return x
    
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
    
class CategoricalNet(nn.Module):
    
    def __init__(self, n_features, size, hiddens=[32,16]):
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
        print("index.shape: ", index.shape)
        arg = args[index] # and this are the sampled coordinates
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