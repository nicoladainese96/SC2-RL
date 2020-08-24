from AC_modules.Layers import *
from ConvLSTM_pytorch.convlstm import ConvLSTM

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

class FullyConvPlayerAndSpatial(nn.Module):
    def __init__(self, in_channels, in_player, player_features, conv_channels=32):
        super(FullyConvPlayerAndSpatial, self).__init__()
        self.conv_net = FullyConvSpatial(in_channels, conv_channels)
        self.fc_net = nn.Sequential(
                                    nn.Linear(in_player, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, player_features),
                                    nn.ReLU()
                                    )
        
    def forward(self, spatial_state, player_state):
        spatial_x = self.conv_net(spatial_state)
        player_x = self.fc_net(player_state)
        spatial_features = self._cat_player_to_spatial(player_x, spatial_x)
        return spatial_features
        
    def _cat_player_to_spatial(self, player_x, spatial_x):
        """ 
        Assume spatial_x of shape (B, conv_channels, res, res).
        Cast player_x from (B, player_features) to (B, player_features, res, res)
        Concatenate spatial_x with the broadcasted player_x along the channel dim.
        """
        res = spatial_x.shape[-1]
        player_x = player_x.reshape((player_x.shape[:2]+(1,1,)))
        player_x = player_x.repeat(1,1,res,res)
        spatial_features = torch.cat([spatial_x, player_x], dim=1)
        return spatial_features

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

# ## NonSpatial networks: start from (batch, n_channels, size, size) and return (batch, n_features) ###
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
        return arg.reshape(-1,1), log_probs[range(len(arg)), arg], log_probs

class ParallelCategoricalNet(nn.Module):

    def __init__(self, n_features, sizes, n_arguments, hiddens=[256]):
        """
        Parameters
        ----------
        n_features: int, last dimension of input tensor used in forward
        sizes: shape (n_arguments,), contains the number of values ranging from 0 to sizes[i]
            that each argument i can assume. Used for masking out impossible values while sampling
            all of them together.
        n_arguments: int, number of categorical arguments to sample
        """
        super(ParallelCategoricalNet, self).__init__()
        self.sizes = sizes
        self.max_size = sizes.max()
        self.n_args = n_arguments
        self.sizes_mask = torch.tensor(self.sizes).view(-1,1) <= torch.arange(self.max_size).repeat(self.n_args,1)
        
        layers = []

        layers.append(nn.Linear(n_features, hiddens[0]))
        layers.append(nn.ReLU())

        for i in range(0,len(hiddens)-1):
            layers.append(nn.Linear(hiddens[i], hiddens[i+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hiddens[-1], self.n_args*self.max_size))
        self.net = nn.Sequential(*layers)

    def forward(self, state_rep):
        """
        Input
        -----
        state_rep: (batch_size, n_features)
        
        Returns
        -------
        arg: (batch_size, n_args)
        log_prob: (batch_size, n_args)
        """
        logits = self.net(state_rep).view(-1, self.n_args, self.max_size) # (batch_size, n_args, max_size)
        # Infer device from spatial_params_net output with parallel_log_prob.is_cuda
        if logits.is_cuda:
            device = 'cuda' # Assume only 1 GPU device is used 
        else:
            device = 'cpu'
        self.sizes_mask = self.sizes_mask.to(device)
        log_probs = F.log_softmax(logits.masked_fill(self.sizes_mask.bool(), float('-inf')), dim=(-1)) 
        probs = torch.exp(log_probs)
        arg = Categorical(probs).sample() #(batch_size, n_args)
        log_prob = log_probs.view(-1, self.max_size)[torch.arange(arg.shape[0]*arg.shape[1]), arg.flatten()]\
                    .view(arg.shape[0], arg.shape[1])
        arg = arg.detach().cpu().numpy()
        return arg, log_prob

class CategoricalIMPALA(ParallelCategoricalNet):
    def __init__(
        self, 
        n_features, 
        sizes, 
        n_arguments
    ):
        super(CategoricalIMPALA, self).__init__(n_features, sizes, n_arguments)
        
    def forward(self, state_rep):
        log_probs = self.get_log_probs(state_rep)
        probs = torch.exp(log_probs)
        torch_arg = Categorical(probs).sample() #(batch_size, n_args)
        log_prob = log_probs.view(-1, self.max_size)[torch.arange(torch_arg.shape[0]*torch_arg.shape[1]), torch_arg.flatten()]\
                    .view(torch_arg.shape[0], torch_arg.shape[1])
        arg = torch_arg.detach().cpu().numpy()
        return arg, log_prob, torch_arg
    
    def get_log_probs(self, state_rep):
        logits = self.net(state_rep).view(-1, self.n_args, self.max_size) # (batch_size, n_args, max_size)
        # Infer device from spatial_params_net output with parallel_log_prob.is_cuda
        if logits.is_cuda:
            device = 'cuda' # Assume only 1 GPU device is used 
        else:
            device = 'cpu'
        self.sizes_mask = self.sizes_mask.to(device)
        log_probs = F.log_softmax(logits.masked_fill(self.sizes_mask.bool(), float('-inf')), dim=(-1)) # sample all arguments in parallel
        return log_probs

# ## sample spatial parameters from a matrix-like state representation

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

class SpatialParameters(nn.Module):
    
    def __init__(self, n_channels, linear_size):
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
        return arg_lst, log_prob, log_probs

class ParallelSpatialParameters(nn.Module):
    
    def __init__(self, n_channels, linear_size, n_arguments):
        super(ParallelSpatialParameters, self).__init__()
        
        self.size = linear_size
        self.n_args = n_arguments
        self.conv = nn.Sequential(nn.Conv2d(n_channels, n_arguments, kernel_size=3, stride=1, padding=1))
    
    def forward(self, x, x_first=True):
        """
        Input
        -----
        x : (B, n_channels, size, size)
        
        Returns
        -------
        arg_lst : (B, n_args, 2)
        log_prob: (B, n_args)
        log_probs: (B, n_args, size**2) # not used anymore
        """
        B = x.shape[0]
        x = self.conv(x)
        x = x.reshape((x.shape[0],self.n_args,-1))
        log_probs = F.log_softmax(x, dim=(-1))
        probs = torch.exp(log_probs)
        index = Categorical(probs).sample()
        y, x = self.unravel_index(index, (self.size,self.size)) # shape (B, n_args)
        if x_first:
            arg_lst = np.array([[xi.detach().cpu().numpy(),yi.detach().cpu().numpy()] for xi, yi in zip(x,y)])
        else:
            arg_lst = np.array([[yi.detach().cpu().numpy(),xi.detach().cpu().numpy()] for xi, yi in zip(x,y)])
        arg_lst = arg_lst.transpose(0,2,1)  #shape (batch, n_arguments, [x,y]) (or [y,x])                 
        log_prob = log_probs.view(B*self.n_args, self.size**2)[torch.arange(B*self.n_args), index.flatten()]\
                    .view(B, self.n_args) 
        return arg_lst, log_prob, log_probs
    
    @staticmethod
    def unravel_index(index, shape):
        """
        Retrieve row and col of a flattened index so that:
        probs_flat = probs.flatten()
        index = Categorical(probs_flat).sample()
        probs_flat[index] == probs[row, col] is True
        
        Input
        -----
        index: (B, n_args)
        shape: tuple (size, size)
        
        Returns
        -------
        rows, cols: (B, n_args), (B, n_args)
        """
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return tuple(reversed(out))

class SpatialIMPALA(ParallelSpatialParameters):
    def __init__(self, n_channels, linear_size, n_arguments):
        super(SpatialIMPALA, self).__init__(n_channels, linear_size, n_arguments)
        
    def forward(self, x, x_first=True):
        B = x.shape[0]
        log_probs = self.get_log_probs(x)
        probs = torch.exp(log_probs)
        index = Categorical(probs).sample() # shape (B, n_args)
        # method inherited from ParallelSpatialParameters
        y, x = self.unravel_index(index, (self.size,self.size)) # both x and y of shape (B, n_args)
        if x_first:
            arg_lst = np.array([[xi.detach().numpy(),yi.detach().numpy()] for xi, yi in zip(x,y)])
        else:
            arg_lst = np.array([[yi.detach().numpy(),xi.detach().numpy()] for xi, yi in zip(x,y)])
        arg_lst = arg_lst.transpose(0,2,1)  #shape (batch, n_arguments, [y,x]) (or [x,y])                 
        log_prob = log_probs.view(B*self.n_args, self.size**2)[torch.arange(B*self.n_args), index.flatten()]\
                    .view(B, self.n_args) 
        return arg_lst, log_prob, index
    
    def get_log_probs(self, x):
        """Compute flatten log_probs for all arguments - shape: (batch_size, n_args, size**2)"""
        x = self.conv(x)
        x = x.reshape((x.shape[0],self.n_args,-1))
        log_probs = F.log_softmax(x, dim=(-1))
        return log_probs



""
### Big architecture IMPALA (v2) ###
# StateEncodingConvBlock
# ConvLSTM_RL
# ResidualConvLayer (in AC_modules.Layers)
# ResidualConvBlock
# DeepResidualBlock
# NonSpatialBlock
# Inputs2D_Net
# SpatialProcessingBlock
# SpatialIMPALA_v2

class StateEncodingConvBlock(nn.Module):
    """ 
    - First conv layer halves the spatial dimensions
    - 2 residual convolutional layers with ReLU pre-activations (they act on the input and not the output)
    - 2x2 MaxPool to halve again the dimensions
    """
    def __init__(self, res, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(StateEncodingConvBlock, self).__init__()
        new_res = (res - kernel_size + 2*padding)//stride + 1
        self.new_res = new_res # useful info to access from outside the class
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            ResidualConvLayer(new_res, out_channels, kernel_size=3),
            ResidualConvLayer(new_res, out_channels, kernel_size=3),
            nn.MaxPool2d(2),
        )
        
    def forward(self, x):
        return self.net(x)

class ConvLSTM_RL(ConvLSTM):
    def forward(self, input_tensor, hidden_state=None, done=None):
        """

        Parameters
        ----------
        input_tensor: 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: 
            list of tuples [(hidden,cell), ..., (hidden,cell)] with 
            length of the list = num_layers
            hidden.shape == cell.shape == (b, c, h, w)
        done:
            Tensor of shape (t,b) 
            
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, t, c, h, w = input_tensor.size()

        if hidden_state is None:
        # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        
        if done is None:
            device = self.cell_list[0].conv.weight.device
            notdone = torch.ones(t, b, c, h, w).float().to(device)
        else:
            notdone = (~done).float().view(t, b, 1, 1, 1) #for broadcasting
            
        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                # reset hidden state to zero whenever an episode ended
                # done is saying that this input is from a new episode
                h = h*notdone[t] # (b,c,w,h) * (b,c,w,h)  [or (b,1,1,1)]
                c = c*notdone[t]
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)
                
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, res):
        super(ResidualConvBlock, self).__init__()
        # pre-activations as in Identity Mappings in Deep Residual Networks https://arxiv.org/abs/1603.05027
        self.net = nn.Sequential(
            ResidualConvLayer(res, in_channels, kernel_size=5),
            ResidualConvLayer(res, in_channels, kernel_size=3),
            ResidualConvLayer(res, in_channels, kernel_size=3)
        )
        
    def forward(self, x):
        return self.net(x)

class DeepResidualBlock(nn.Module):
    def __init__(self, in_channels, res, n_blocks=3):
        super(DeepResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.res = res
        self.net = nn.Sequential(
            *[ResidualConvBlock(in_channels, res) for _ in range(n_blocks)]
        )
        
    def forward(self, x):
        x = x.reshape(-1, self.in_channels, self.res, self.res)
        return self.net(x)

class NonSpatialBlock(nn.Module):
    def __init__(self, in_channels, res):
        super(NonSpatialBlock, self).__init__()
        self.flattened_size = int(in_channels*(res**2))
        self.out_features = 512
        self.net = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = x.reshape(-1, self.flattened_size)
        return self.net(x)

class Inputs2D_Net(nn.Module):
    def __init__(
        self, 
        in_player, 
        n_actions, 
        embedding_dim=10
    ):
        super(Inputs2D_Net, self).__init__()
        self.out_features = 64 # in case needed from outside
        self.embedding = nn.Embedding(n_actions, embedding_dim, padding_idx=0) # no_op action mapped to 0
        self.MLP = nn.Sequential(
            nn.Linear(in_player+embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
    def forward(self, player_info, last_action):
        """
        player_info: (batch, in_player)
        last_action: (batch,)
        """
        embedded_action = self.embedding(last_action).float()
        nonspatial_input = torch.cat([player_info, embedded_action], dim=1)
        out = self.MLP(nonspatial_input)
        return out

class SpatialProcessingBlock(nn.Module):
    def __init__(
        self, 
        res, 
        screen_channels, 
        minimap_channels,
        encoding_channels,
        lstm_channels=96,
    ):
        super(SpatialProcessingBlock, self).__init__()
        assert res%4 == 0, "Provide an input with resolution divisible by 4"
        self.res = res
        self.new_res = int(res/4)
        self.lstm_channels = lstm_channels
        self.screen_state_enc_net = StateEncodingConvBlock(res, screen_channels, encoding_channels)
        self.minimap_state_enc_net = StateEncodingConvBlock(res, minimap_channels, encoding_channels)
        self.conv_lstm = ConvLSTM_RL(
                     encoding_channels*2, 
                     lstm_channels, 
                     kernel_size=(3,3), 
                     num_layers=1,
                     batch_first=False, # first time dimension, but return is with batch first
                     bias=True,
                     return_all_layers=True
                    )
        self.deep_residual_block = DeepResidualBlock(lstm_channels, self.new_res)
        self.nonspatial_block = NonSpatialBlock(lstm_channels, self.new_res)
        
    def forward(self, screen_layers, minimap_layers, hidden_state=None, cell_state=None, done=None):
        """
        Inputs
        ------
        screen_layers: (time, batch_size, screen_channels, res, res)
        minimap_layers: (time, batch_size, minimap_channels, res, res)
        hidden_state: (batch_size, lstm_channels, new_res, new_res)
        cell_state: (batch_size, lstm_channels, new_res, new_res)
        
        Intermediate variables
        ----------------------
        inputs_3D: (batch_size, encoding_channels*2, new_res, new_res)
        """
        T = screen_layers.shape[0]
        B = screen_layers.shape[1]
        # merge T and B dimensions for standard layers
        screen_layers = screen_layers.view((-1,*screen_layers.shape[-3:]))
        minimap_layers = minimap_layers.view((-1,*minimap_layers.shape[-3:]))
        # State Encoding
        screen_enc = self.screen_state_enc_net(screen_layers)
        minimap_enc = self.minimap_state_enc_net(minimap_layers)
        # concatenate along channel dim + (T*B,...) -> (T, B, ...)
        inputs_3D = torch.cat([screen_enc, minimap_enc], dim=1).view(T,B,-1,self.new_res,self.new_res) 
        # Memory Processing
        if hidden_state is None:
            layer_output_list, last_state_list = self.conv_lstm(inputs_3D, done=done)
        else:
            assert cell_state is not None, \
                "hidden_state provided, but cell_state is None"
            assert hidden_state.shape == cell_state.shape, \
                ("hidden_state and cell_state have different shapes", hidden_state.shape, cell_state.shape)
            layer_output_list, last_state_list = self.conv_lstm(inputs_3D,
                                                                [(hidden_state, cell_state)], 
                                                                done=done
                                                               )
        # output is 5d with batch-first 
        outputs_3D = layer_output_list[-1].transpose(1,0) # (T,B,c,w,h)
        # this works only if num_layers = 1
        hidden_state = last_state_list[-1][0]
        cell_state = last_state_list[-1][1]
        
        # Spatial and Non-Spatial Processing
        spatial_features = self.deep_residual_block(outputs_3D) # (T*B, c, w, h) 
        nonspatial_features = self.nonspatial_block(outputs_3D) # (T*B, n_features)
        
        return spatial_features, nonspatial_features, hidden_state, cell_state

class SpatialIMPALA_v2(SpatialIMPALA):
    """
    Same as SpatialIMPALA, but handles inputs of linear size 4 times smaller than the output size,
    e.g. 8x8 in input but have to create a logit matrix of 32x32
    """
    def __init__(
        self, 
        n_channels, 
        linear_size, 
        n_arguments
    ):
        super(SpatialIMPALA_v2, self).__init__(n_channels, linear_size, n_arguments)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(n_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, n_arguments, kernel_size=3, stride=1, padding=1)
        )

""
class ConditionedSpatialParameters(nn.Module):
    
    def __init__(self, linear_size):
        super(ConditionedSpatialParameters, self).__init__()
        
        self.size = linear_size
    
    def forward(self, x, embedded_a, x_first=True):
        B = x.shape[0]
        x = torch.einsum('bc, bcwh -> bwh', embedded_a, x)
        x = x.reshape((B,-1))
        log_probs = F.log_softmax(x, dim=(-1))
        probs = torch.exp(log_probs)
        index = Categorical(probs).sample()
        y, x = unravel_index(index, (self.size,self.size))
        if x_first:
            arg_lst = [[xi.item(),yi.item()] for xi, yi in zip(x,y)]
        else:
            arg_lst = [[yi.item(),xi.item()] for xi, yi in zip(x,y)]
        log_prob = log_probs[torch.arange(B), index]
        return arg_lst, log_prob, log_probs

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
        
        return arg, log_probs.view(self.size, self.size)[arg[0], arg[1]], log_probs

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
