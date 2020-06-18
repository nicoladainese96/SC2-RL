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

debug = True

### Shared ActorCritic architecture

class SharedActor(nn.Module):
    def __init__(self, action_space, n_features):
        super(SharedActor, self).__init__()
        #self.linear = nn.Sequential(
        #                nn.Linear(n_features, 256),
        #                nn.ReLU(),
        #                nn.Linear(256, action_space))
        self.linear = nn.Linear(n_features, action_space)
        
    def forward(self, shared_repr):
        logits = self.linear(shared_repr)
        return logits
    
class SharedCritic(nn.Module):
    def __init__(self, n_features):
        super(SharedCritic, self).__init__()
        #self.net = nn.Sequential(
        #                nn.Linear(n_features, 256),
        #                nn.ReLU(),
        #                nn.Linear(256, 1))
        self.net = nn.Linear(n_features, 1)
        
    def forward(self, shared_repr):
        V = self.net(shared_repr)
        return V
    
class SpatialActorCritic(nn.Module):
    def __init__(self, action_space, env, spatial_model, nonspatial_model, spatial_dict, nonspatial_dict, 
                 n_features, n_channels, action_dict=None):
        super(SpatialActorCritic, self).__init__()
        
        # Environment-related attributes
        if action_dict is None:
            self.action_dict = {0:_NO_OP, 1:_SELECT_ARMY, 2:_MOVE_SCREEN}
        else:
            self.action_dict = action_dict
        self.screen_res = env.observation_spec()[0]['feature_screen'][1:]
        self.all_actions = env.action_spec()[0][1]
        self.all_arguments = env.action_spec()[0][0]
        
        # Useful HyperParameters as attributes
        self.n_features = n_features
        self.n_channels = n_channels
        
        # Networks
        self.spatial_features_net = spatial_model(**spatial_dict)
        self.nonspatial_features_net = nonspatial_model(**nonspatial_dict) 
        self.actor = SharedActor(action_space, n_features)
        self.critic = SharedCritic(n_features)
        self._init_params_nets()
    
    def _init_params_nets(self):
        arguments_networks = {}
        self.arguments_dict = {}
        self.arguments_type = {}
        self.act_to_arg_names = {}
        
        for a in self.action_dict:
            action = self.all_actions[self.action_dict[a]]
            args = action.args
            self.act_to_arg_names[a] = [arg.name for arg in args]
            for arg in args:
                self.arguments_dict[arg.name] = arg.id # store 'name':id pairs for future use
                if debug: print('\narg.name: ', arg.name)

                size = self.all_arguments[arg.id].sizes
                if debug: print('size: ', size)
                if len(size) == 1:
                    if debug: 
                        print("Init CategoricalNet for "+arg.name+' argument')
                    arguments_networks[arg.name] = CategoricalNet(self.n_features, size[0]) 
                    self.arguments_type[arg.name] = 'categorical'
                else:
                    if debug: print("Init SpatialNet for "+arg.name+' argument')
                    arguments_networks[arg.name] = SpatialParameters(self.n_channels, size[0]) 
                    self.arguments_type[arg.name] = 'spatial'
                    
        self.arguments_networks = nn.ModuleDict(arguments_networks)
        
        return
    
    def pi(self, state, mask):
        spatial_features = self.spatial_features_net(state)
        nonspatial_features = self.nonspatial_features_net(spatial_features)
        logits = self.actor(nonspatial_features) 
        log_probs = F.log_softmax(logits.masked_fill((mask).bool(), float('-inf')), dim=-1) 
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

class SpatialActorCritic_v1(SpatialActorCritic):
    def __init__(self, action_space, env, spatial_model, nonspatial_model, spatial_dict, nonspatial_dict, 
                 n_features, n_channels, action_dict=None, embed_dim=16):
        self.embed_dim = embed_dim
        super(SpatialActorCritic_v1, self).__init__(action_space, env, spatial_model, nonspatial_model,
                                                 spatial_dict, nonspatial_dict, n_features, n_channels, action_dict)
        self.embedding = nn.Embedding(action_space, embed_dim)
    
    def _init_params_nets(self):
        arguments_networks = {}
        self.arguments_dict = {}
        self.arguments_type = {}
        self.act_to_arg_names = {}
        
        for a in self.action_dict:
            action = self.all_actions[self.action_dict[a]]
            args = action.args
            self.act_to_arg_names[a] = [arg.name for arg in args]
            for arg in args:
                self.arguments_dict[arg.name] = arg.id # store 'name':id pairs for future use
                if debug: print('\narg.name: ', arg.name)

                size = self.all_arguments[arg.id].sizes
                if debug: print('size: ', size)
                if len(size) == 1:
                    if debug: 
                        print("Init CategoricalNet for "+arg.name+' argument')
                    arguments_networks[arg.name] = CategoricalNet(self.n_features+self.embed_dim, size[0]) 
                    self.arguments_type[arg.name] = 'categorical'
                else:
                    if debug: print("Init SpatialNet for "+arg.name+' argument')
                    arguments_networks[arg.name] = SpatialParameters(self.n_channels+self.embed_dim, size[0]) 
                    self.arguments_type[arg.name] = 'spatial'
                    
        self.arguments_networks = nn.ModuleDict(arguments_networks)
        
        return
    
class SpatialActorCritic_v2(SpatialActorCritic):
    def __init__(self, action_space, env, spatial_model, nonspatial_model, spatial_dict, nonspatial_dict, 
                 n_features, n_channels, action_dict=None):
        super(SpatialActorCritic_v2, self).__init__(action_space, env, spatial_model, nonspatial_model,
                                                 spatial_dict, nonspatial_dict, n_features, n_channels, action_dict)
    
    def _init_params_nets(self):
        arguments_networks = {}
        self.arguments_dict = {}
        self.arguments_type = {}
        self.act_to_arg_names = {}
        
        for a in self.action_dict:
            action_id = self.action_dict[a]
            action = self.all_actions[action_id]
            args = action.args
            self.act_to_arg_names[a] = [str(action_id.name)+"/"+arg.name for arg in args]
            for arg in args:
                arg_name = str(action_id.name)+"/"+arg.name
                self.arguments_dict[arg_name] = arg.id # store 'name':id pairs for future use
                if debug: 
                    print('\narg.name: ', arg.name)
                    print('arg_name: ', arg_name)

                size = self.all_arguments[arg.id].sizes
                if debug: print('size: ', size)
                if len(size) == 1:
                    if debug: 
                        print("Init CategoricalNet for "+arg_name+' argument')
                    arguments_networks[arg_name] = CategoricalNet(self.n_features, size[0]) 
                    self.arguments_type[arg_name] = 'categorical'
                else:
                    if debug: print("Init SpatialNet for "+arg_name+' argument')
                    arguments_networks[arg_name] = SpatialParameters(self.n_channels, size[0]) 
                    self.arguments_type[arg_name] = 'spatial'
                    
        self.arguments_networks = nn.ModuleDict(arguments_networks)
        
        return
    
class SpatialActorCritic_v3(SpatialActorCritic):
    def __init__(self, action_space, env, spatial_model, nonspatial_model, spatial_dict, nonspatial_dict, 
                 n_features, n_channels, action_dict=None):
        self.embed_dim = n_channels
        super(SpatialActorCritic_v3, self).__init__(action_space, env, spatial_model, nonspatial_model,
                                                 spatial_dict, nonspatial_dict, n_features, n_channels, action_dict)
        self.embedding = nn.Embedding(action_space, n_channels)
    
    def _init_params_nets(self):
        arguments_networks = {}
        self.arguments_dict = {}
        self.arguments_type = {}
        self.act_to_arg_names = {}
        
        for a in self.action_dict:
            action = self.all_actions[self.action_dict[a]]
            args = action.args
            self.act_to_arg_names[a] = [arg.name for arg in args]
            for arg in args:
                self.arguments_dict[arg.name] = arg.id # store 'name':id pairs for future use
                if debug: print('\narg.name: ', arg.name)

                size = self.all_arguments[arg.id].sizes
                if debug: print('size: ', size)
                if len(size) == 1:
                    if debug: 
                        print("Init CategoricalNet for "+arg.name+' argument')
                    arguments_networks[arg.name] = CategoricalNet(self.n_features+self.embed_dim, size[0]) 
                    self.arguments_type[arg.name] = 'categorical'
                else:
                    if debug: print("Init SpatialNet for "+arg.name+' argument')
                    arguments_networks[arg.name] = ConditionedSpatialParameters(size[0]) 
                    self.arguments_type[arg.name] = 'spatial'
                    
        self.arguments_networks = nn.ModuleDict(arguments_networks)
        
        return
    
    def sample_param(self, arg_name, *args):
        """
        Takes as input spatial_features if it's a spatial parameter,
        nonspatial_features otherwise.
        """
        return self.arguments_networks[arg_name](*args)