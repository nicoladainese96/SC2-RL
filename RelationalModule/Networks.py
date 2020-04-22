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
        
        
### Network used for parameters sampling starting from array-like state representation ###

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
        distribution = Categorical(probs)
        arg = distribution.sample().item() 
        return [arg], log_probs.view(-1)[arg], probs
    
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
