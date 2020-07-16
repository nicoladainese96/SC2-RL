import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import itertools as it

from AC_modules.Networks import *
from pysc2.lib import actions as sc_actions

_NO_OP = sc_actions.FUNCTIONS.no_op.id
_SELECT_ARMY = sc_actions.FUNCTIONS.select_army.id
_MOVE_SCREEN = sc_actions.FUNCTIONS.Attack_screen.id

debug = False

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
    
########################################################################################################################

class SpatialActorCritic(nn.Module):
    """
    Used in SpatialA2C
    """
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

########################################################################################################################

class SpatialActorCritic_v1(SpatialActorCritic):
    """
    Used in SpatialActorCritic_v1
    """
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
    
########################################################################################################################

class SpatialActorCritic_v2(SpatialActorCritic):
    """
    Used in SpatialA2C_v2, SpatialA2C_MaxEnt and SpatialA2C_MaxEnt_v2 - compatible with SpatialA2C too
    """
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
    """
    Used in SpatialActorCritic_v3
    """
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


########################################################################################################################

class SpatialActorCritic_v4(SpatialActorCritic_v2):
    """
    Used in FullSpaceA2C
    """
    def __init__(self, action_space, env, spatial_model, nonspatial_model, spatial_dict, nonspatial_dict, 
                 n_features, n_channels, action_dict=None):
        # Init SpatialActorCritic_v2
        super().__init__(action_space, env, spatial_model, nonspatial_model,
                                                 spatial_dict, nonspatial_dict, n_features, n_channels, action_dict)
        
    def pi(self, spatial_state, player_state, mask):
        spatial_features = self.spatial_features_net(spatial_state, player_state)
        nonspatial_features = self.nonspatial_features_net(spatial_features)
        logits = self.actor(nonspatial_features) 
        log_probs = F.log_softmax(logits.masked_fill((mask).bool(), float('-inf')), dim=-1) 
        return log_probs, spatial_features, nonspatial_features
    
    def V_critic(self, spatial_state, player_state):
        spatial_features = self.spatial_features_net(spatial_state, player_state)
        nonspatial_features = self.nonspatial_features_net(spatial_features)
        V = self.critic(nonspatial_features)
        return V
    
########################################################################################################################

class ParallelActorCritic(nn.Module):
    """
    Used in FullSpaceA2C_v2
    
    Description of some attributes:
    - action_table: numpy array of shape (n_actions,) 
        is a look-up table that associates an action index to its StarCraft action id
    - spatial_arg_mask: numpy array of shape (n_actions, n_spatial_args) 
        spatial_arg_mask[a] is a mask telling which of the n_spatial_args sampled args
        belong to action `a`. Same thing for categorical_arg_mask
    """
    def __init__(self, env, spatial_model, nonspatial_model, spatial_dict, nonspatial_dict, 
                 n_features, n_channels, action_names):
        super(ParallelActorCritic, self).__init__()
        
        self.action_names = action_names
        self._set_action_table() # creates self.action_table
        self.screen_res = env.observation_spec()[0]['feature_screen'][1:]
        self.all_actions = env.action_spec()[0][1]
        self.all_arguments = env.action_spec()[0][0]
        
        # Useful HyperParameters as attributes
        self.n_features = n_features
        self.n_channels = n_channels
        action_space = len(action_names)
        
        # Networks
        self.spatial_features_net = spatial_model(**spatial_dict)
        self.nonspatial_features_net = nonspatial_model(**nonspatial_dict) 
        self.actor = SharedActor(action_space, n_features)
        self.critic = SharedCritic(n_features)
        self._init_arg_names()
        self._set_spatial_arg_mask()
        self._set_categorical_arg_mask()
        self._init_params_nets()
    
    def _set_action_table(self):
        action_ids = [sc_actions.FUNCTIONS[a_name].id for a_name in self.action_names]
        action_table = np.array([action_ids[i] for i in range(len(action_ids))])
        self.action_table = action_table
    
    def _init_arg_names(self):
        spatial_arg_names = []
        categorical_arg_names = []
        categorical_sizes = []
        act_to_arg_names = {}

        for action_id in self.action_table:
            action = self.all_actions[action_id]
            args = action.args
            act_to_arg_names[action_id] = [str(action.name)+"/"+arg.name for arg in args]
            spatial = []
            categorical = []
            for arg in args:
                arg_name = str(action.name)+"/"+arg.name
                size = self.all_arguments[arg.id].sizes
                if len(size) == 1:
                    categorical.append(arg_name)
                    categorical_sizes.append(size[0])
                else:
                    spatial.append(arg_name)
            spatial_arg_names+=spatial
            categorical_arg_names+=categorical
    
        self.spatial_arg_names = spatial_arg_names
        self.n_spatial_args = len(spatial_arg_names)
        self.categorical_arg_names = categorical_arg_names
        self.n_categorical_args = len(categorical_arg_names)
        self.categorical_sizes = np.array(categorical_sizes)
        self.act_to_arg_names = act_to_arg_names 

    def _set_spatial_arg_mask(self):
        spatial_arg_mask = np.zeros((self.action_table.shape[0], self.n_spatial_args))
        for i, action_id in enumerate(self.action_table):
            action_arg_names = self.act_to_arg_names[action_id]
            spatial_arg_mask[i] = np.array([1 if self.spatial_arg_names[j] in action_arg_names else 0 \
                                            for j in range(self.n_spatial_args)])
        self.spatial_arg_mask = spatial_arg_mask
    
    def _set_categorical_arg_mask(self):
        categorical_arg_mask = np.zeros((self.action_table.shape[0], self.n_categorical_args))
        for i, action_id in enumerate(self.action_table):
            action_arg_names = self.act_to_arg_names[action_id]
            categorical_arg_mask[i] = np.array([1 if self.categorical_arg_names[j] in action_arg_names else 0 \
                                            for j in range(self.n_categorical_args)])
        self.categorical_arg_mask = categorical_arg_mask

    def _init_params_nets(self):
        self.spatial_params_net = ParallelSpatialParameters(self.n_channels, self.screen_res[0], self.n_spatial_args)
        self.categorical_params_net = ParallelCategoricalNet(self.n_features, self.categorical_sizes, self.n_categorical_args)
        
    def pi(self, spatial_state, player_state, mask):
        spatial_features = self.spatial_features_net(spatial_state, player_state)
        nonspatial_features = self.nonspatial_features_net(spatial_features)
        logits = self.actor(nonspatial_features) 
        log_probs = F.log_softmax(logits.masked_fill((mask).bool(), float('-inf')), dim=-1) 
        return log_probs, spatial_features, nonspatial_features
    
    def V_critic(self, spatial_state, player_state):
        spatial_features = self.spatial_features_net(spatial_state, player_state)
        nonspatial_features = self.nonspatial_features_net(spatial_features)
        V = self.critic(nonspatial_features)
        return V
    
    def sample_spatial_params(self, spatial_features, actions):
        """
        Input
        -----
        spatial_features: tensor, (batch_size, n_channels, screen_res, screen_res)
        actions: array, (batch_size,)
        
        Returns
        -------
        arg_list: list of lists
        """
        batch_size = actions.shape[0]
        parallel_args, parallel_log_prob, _ = self.spatial_params_net(spatial_features)
       
        # Select only spatial arguments needed by sampled actions
        
        arg_mask = self.spatial_arg_mask[actions,:] # shape (batch_size, n_spatial_args)
        batch_pos = arg_mask.nonzero()[0]
        arg_pos = arg_mask.nonzero()[1]
        args = parallel_args[batch_pos, arg_pos]
        arg_list = [list(args[batch_pos==i]) for i in range(batch_size)]
        
        # Compute composite log_probs of selected arguments 
        
        # Infer device from spatial_params_net output with parallel_log_prob.is_cuda
        if parallel_log_prob.is_cuda:
            device = 'cuda' # Assume only 1 GPU device is used 
        else:
            device = 'cpu'     
        # for every arg index contains the index of the action that uses that parameter
        main_action_ids = torch.tensor(self.spatial_arg_mask.nonzero()[0]).to(device)
        sum_log_prob = torch.zeros(batch_size, len(self.action_table)) # (batch_size, action_space)
        sum_log_prob.index_add_(1, main_action_ids, parallel_log_prob)
        sampled_actions = torch.tensor(actions) # of shape (batch_size,)
        # sum of log_probs of the relevant parameters by
        log_prob = sum_log_prob[torch.arange(batch_size), sampled_actions]

        return arg_list, log_prob
    
    def sample_categorical_params(self, categorical_features, actions):
        """
        Input
        -----
        categorical_features: tensor, (batch_size, n_channels, screen_res, screen_res)
        actions: array, (batch_size,)
        """
        batch_size = actions.shape[0]
        parallel_args, parallel_log_prob = self.categorical_params_net(categorical_features)
        arg_mask = self.categorical_arg_mask[actions,:] # shape (batch_size, n_spatial_args)
        
        # select correct arguments
        
        batch_pos = arg_mask.nonzero()[0]
        arg_pos = arg_mask.nonzero()[1]
        args = parallel_args[batch_pos, arg_pos]
        arg_list = [list(args[batch_pos==i]) for i in range(batch_size )]

        # select and sum correct log probs
        
        if parallel_log_prob.is_cuda:
            device = 'cuda' # Assume only 1 GPU device is used 
        else:
            device = 'cpu'
        # for every arg index contains the index of the action that uses that parameter
        main_action_ids = torch.tensor(self.categorical_arg_mask.nonzero()[0]).to(device)
        sum_log_prob = torch.zeros(batch_size, len(self.action_table)) # (batch_size, action_space)
        sum_log_prob.index_add_(1, main_action_ids, parallel_log_prob)
        sampled_actions = torch.tensor(actions) # of shape (batch_size,)
        # sum of log_probs of the relevant parameters by
        log_prob = sum_log_prob[torch.arange(batch_size), sampled_actions]

        return arg_list, log_prob
    
    def sample_params(self, nonspatial_features, spatial_features, actions):
        categorical_arg_list, categorical_log_prob = self.sample_categorical_params(nonspatial_features, actions)
        spatial_arg_list, spatial_log_prob = self.sample_spatial_params(spatial_features, actions)
        
        # merge arg lists
        assert len(categorical_arg_list) == len(spatial_arg_list), ("Expected same length for arg lists", \
                                                                len(categorical_arg_list), len(spatial_arg_list))
        
        assert categorical_log_prob.shape == spatial_log_prob.shape, ("Expected same log_prob shape", \
                                                                 categorical_log_prob.shape, spatial_log_prob.shape)
        log_prob = categorical_log_prob + spatial_log_prob
        arg_list = []
        for cat, spa in zip(categorical_arg_list, spatial_arg_list):
            args = []
            if len(cat) != 0:
                args.append(cat)
            args += [list(s) for s in spa] # hopefully is the right format [[arg1],[arg2],...] x batch time
            arg_list.append(args)
            
        return arg_list, log_prob
    
########################################################################################################################

class FullSpatialActorCritic(nn.Module):
    """
    Uses full action and state space.
    """
    def __init__(self, env, spatial_model, nonspatial_model, spatial_dict, nonspatial_dict, 
                 n_features, n_channels):
        super(FullSpatialActorCritic, self).__init__()
        
        # Environment variables
        self.screen_res = env.observation_spec()[0]['feature_screen'][1:]
        self.all_actions = env.action_spec()[0][1]
        self.all_arguments = env.action_spec()[0][0]
        self.action_space = len(self.all_actions) #573
        
        # Useful HyperParameters as attributes
        self.n_features = n_features
        self.n_channels = n_channels
        
        # Networks
        self.spatial_features_net = spatial_model(**spatial_dict)
        self.nonspatial_features_net = nonspatial_model(**nonspatial_dict) 
        self.actor = SharedActor(self.action_space, n_features)
        self.critic = SharedCritic(n_features)
        self._init_params_nets()
    
    def _init_params_nets(self):
        arguments_networks = {}
        self.arguments_names_lst = [] #750
        self.arguments_type = {}
        self.act_to_arg_names = {}
        
        for a in range(self.action_space):
            # access proper action object with more info
            action = self.all_actions[a] 
            if debug:
                print("Action: ", a, action)
                print("Action.name: ", action.name)
            args = action.args
            self.act_to_arg_names[a] = [str(action.name)+"/"+arg.name for arg in args] 
            for arg in args:
                arg_name = str(action.name)+"/"+arg.name
                self.arguments_names_lst.append(arg_name) # store arg_name for future use
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
        
    def pi(self, spatial_state, player_state, mask):
        spatial_features = self.spatial_features_net(spatial_state, player_state)
        nonspatial_features = self.nonspatial_features_net(spatial_features)
        logits = self.actor(nonspatial_features) 
        log_probs = F.log_softmax(logits.masked_fill((mask).bool(), float('-inf')), dim=-1) 
        return log_probs, spatial_features, nonspatial_features
    
    def V_critic(self, spatial_state, player_state):
        spatial_features = self.spatial_features_net(spatial_state, player_state)
        nonspatial_features = self.nonspatial_features_net(spatial_features)
        V = self.critic(nonspatial_features)
        return V
    
    def sample_param(self, state_rep, arg_name):
        """
        Takes as input spatial_features if it's a spatial parameter,
        nonspatial_features otherwise.
        """
        return self.arguments_networks[arg_name](state_rep)
