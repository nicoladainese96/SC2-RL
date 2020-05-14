import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import itertools as it

from AC_modules.Networks import *
from pysc2.lib import actions

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id

debug = False

### Shared ActorCritic architecture

class SharedActor(nn.Module):
    def __init__(self, action_space, n_features):
        super(SharedActor, self).__init__()
        self.linear = nn.Sequential(
                        nn.Linear(n_features, 256),
                        nn.ReLU(),
                        nn.Linear(256, action_space))

    def forward(self, shared_repr):
        log_probs = F.log_softmax(self.linear(shared_repr), dim=1)
        return log_probs
    
class SharedCritic(nn.Module):
    def __init__(self, n_features):
        super(SharedCritic, self).__init__()
        self.net = nn.Sequential(
                        nn.Linear(n_features, 256),
                        nn.ReLU(),
                        nn.Linear(256, 1))

    def forward(self, shared_repr):
        V = self.net(shared_repr)
        return V
    
class SpatialActorCritic(nn.Module):
    """
    
    """
    def __init__(self, action_space, env, n_layers, linear_size, in_channels, n_channels, **non_spatial_HPs):
        """
        Parameters
        ----------
        action_space: int
            Number of (discrete) possible actions to take
        map_size: int
            If input is (batch_dim, n_channels, linear_size, linear_size), 
            then map_size = linear_size - 2
        env: int
            Instance of the environment
        **net_args: dict (optional)
            Dictionary of {'key':value} pairs valid for OheNet
        """
        super(SpatialActorCritic, self).__init__()
        
        # Environment-related attributes
        self.action_dict = {0:_NO_OP, 1:_SELECT_ARMY, 2:_MOVE_SCREEN}
        self.screen_res = env.observation_spec()[0]['feature_screen'][1:]
        self.all_actions = env.action_spec()[0][1]
        self.all_arguments = env.action_spec()[0][0]
        
        # Useful HyperParameters as attributes
        self.n_channels = n_channels
        
        # Networks
        self.spatial_features_net = SpatialFeatures(n_layers, linear_size, in_channels, n_channels)
        self.nonspatial_features_net = NonSpatialFeatures(linear_size, n_channels, **non_spatial_HPs)
        self.actor = SharedActor(action_space, n_features=n_channels)
        self.critic = SharedCritic(n_features=n_channels)
        self._init_params_nets()
    
    def _init_params_nets(self):
        self.arguments_networks = {}
        self.arguments_dict = {}

        for a in self.action_dict:
            action = self.all_actions[self.action_dict[a]]
            args = action.args

            for arg in args:
                self.arguments_dict[arg.name] = arg.id # store 'name':id pairs for future use
                if debug: print('\narg.name: ', arg.name)

                size = self.all_arguments[arg.id].sizes
                if debug: print('size: ', size)
                if len(size) == 1:
                    if debug: print("Init CategoricalNet for "+arg.name+' argument')
                    self.arguments_networks[arg.name] = CategoricalNet(self.n_channels, size[0]) 
                else:
                    if debug: print("Init SpatialNet for "+arg.name+' argument')
                    self.arguments_networks[arg.name] = SpatialParameters(self.n_channels, size[0]) 
        return
    
    def pi(self, state, available_actions):
        
        spatial_features = self.spatial_features_net(state)
        nonspatial_features = self.nonspatial_features_net(spatial_features)
        
        logits = self.actor(nonspatial_features)
        if debug: print("logits: ", logits)
            
        mask = self.get_action_mask(available_actions)
        if debug: print("mask: ", mask)
            
        logits[:,mask] = torch.tensor(np.NINF)
        if debug: print("logits (after mask): ", logits)
            
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, spatial_features, nonspatial_features
    
    def V_critic(self, state):
        spatial_features = self.spatial_features_net(state)
        nonspatial_features = self.nonspatial_features_net(spatial_features)
        V = self.critic(nonspatial_features)
        return V
    
    def sample_param(self, state_rep, arg_name):
        """
        Takes as input spatial_features if it's a spatial parameter,
        nonspatial_features otherwise.
        """
        return self.arguments_networks[arg_name](state_rep)
        
    def get_action_mask(self, available_actions):
        action_mask = ~np.array([self.action_dict[i] in available_actions for i in self.action_dict.keys()])
        return action_mask
    
    def to(self, device):
        """Wrapper of .to method to load the model on GPU/CPU"""
        self.spatial_features_net.to(device)
        self.nonspatial_features_net.to(device)
        self.actor.to(device)
        self.critic.to(device)
        # this part here is not loaded automatically
        for key in self.arguments_networks:
            self.arguments_networks[key].to(device)
    