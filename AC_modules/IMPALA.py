import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical

from AC_modules.Networks import *
from AC_modules.ActorCriticArchitecture import ParallelActorCritic

from pysc2.lib import actions as sc_actions

debug = True

class IMPALA_AC(ParallelActorCritic):
    """
    Main changes from ParallelActorCritic:
    1. Use SpatialIMPALA and CategoricalIMPALA instead of ParallelSpatialParameters and ParallelCategoricalNet
       Same architecture but provide also a method for computing the full log_probs without sampling - will
       be used by the learner to compute the log_probs of the actions sampled by the actor
    2. Include actor_step and learner_step, so it has full capability of interacting with the environment (like
       BatchedA2C implementations), but relies on torchbeast's vtrace algorithm for updates.
    3. actor_step returns a dictionary with 
        - `log_prob` of the composite actions sampled (main_action + categorical_args + spatial_args)
        - `main_action` IDs sampled
        - `categorical_indexes` padded with -1 to always have the same length regardless of the main actions sampled
        - `spatial_indexes` padded with -1 to always have the same length regardless of the main actions sampled
        - `sc2_env_action`, that basically are sc_actions.FunctionCall[main_action, [categorical_indexes, spatial_indexes]
           these actions will not be stored in the buffer, because the learner doesn't need them
    4. actor_step receives a dictionary of tensors env_output, so it just has to move them on the correct device (cpu for
       actor, cuda for learner)
    
    Notes
    -----
    - No entropy is computed for the behavior policy, but only for the learner's policy during the learner_step
    
    With respect to BatchedA2C's agents we also have the following changes:
    - Discount factor gamma and entropy cost H are specified inside the monobeast.py script
    - Is no more possible to specify n_steps, but the maximum number of steps + bootstrapping will be used by vtrace
      algorithm
    """
    def __init__(self, env, spatial_model, nonspatial_model, spatial_dict, nonspatial_dict, 
                 n_features, n_channels, action_names, device):
        super(IMPALA_AC, self).__init__(env, spatial_model, nonspatial_model, spatial_dict, nonspatial_dict, 
                 n_features, n_channels, action_names)
        # let's pay attention to the device with which we are working
        self.device = device 
        
        # number of categorical and spatial arguments that we expect at most for any action 
        # used for padding arguments and writing them into the buffers always with the same length
        self.max_num_categorical_args = torch.max(self.categorical_arg_mask.sum(axis=1)) # should be 1
        self.max_num_spatial_args = torch.max(self.spatial_arg_mask.sum(axis=1)) # should be 2 because of select_rect
        
        if debug:
            # they should be SpatialIMPALA and CategoricalIMPALA
            print(self.spatial_params_net)
            print(self.categorical_params_net)
            assert self.max_num_categorical_args == 1, \
                "Expected at most 1 categorical arg per action, found %d"%self.max_num_categorical_args
            assert self.max_num_spatial_args == 2, \
                "Expected at most 2 spatial args per action, found %d"%self.max_num_spatial_args
            
    def _init_params_nets(self):
        """ Overwrite super()._init_params_nets used in super().__init__(...) - check needed"""
        self.spatial_params_net = SpatialIMPALA(self.n_channels, self.screen_res[0], self.n_spatial_args)
        self.categorical_params_net = CategoricalIMPALA(self.n_features, self.categorical_sizes, self.n_categorical_args)
        
    def V_critic(self, spatial_state=None, player_state=None, nonspatial_features=None):
        """
        Introduce the possibility of skipping spatial and nonspatial computations by providing directly
        nonspatial_features. (Useful during learner_step, when we compute the baseline together with the
        policy)
        """
        if nonspatial_features is None:
            assert (spatial_state is not None) and (player_state is not None), "provided None input"
            spatial_features = self.spatial_features_net(spatial_state, player_state)
            nonspatial_features = self.nonspatial_features_net(spatial_features)
        V = self.critic(nonspatial_features)
        return V
    
    def actor_step(self, env_output):
        # TODO: check all batch dimensions and shapes for compatibility with enviroment and buffers
        spatial_state = env_output['spatial'].to(self.device)
        player_state = env_output['player'].to(self.device)
        action_mask = env_output['action_mask'].to(self.device)
        
        log_probs, spatial_features, nonspatial_features = self.pi(spatial_state, player_state, action_mask)
        probs = torch.exp(log_probs)
        main_action = Categorical(probs).sample()
        main_action = a.detach().cpu().numpy()
        log_prob = log_probs[range(len(main_action)), main_action]
        
        args, args_log_prob, args_indexes = self.sample_params(nonspatial_features, spatial_features, main_action)
        assert args_log_prob.shape == log_prob.shape, ("Shape mismatch between arg_log_prob and log_prob ",\
                                                      args_log_prob.shape, log_prob.shape)
        log_prob = log_prob + args_log_prob
        
        action_id = np.array([self.AC.action_table[act] for act in a])
        sc2_env_action = [sc_actions.FunctionCall(action_id[i], args[i]) for i in range(len(action_id))]
        
        actor_output = {'log_prob':log_prob,
                        'main_action':main_action,
                        'sc_env_action':sc2_env_action,
                        **args_indexes} # args_indexes = {'categorical_args_indexes', 'spatial_args_indexes'}
        
        return actor_output
    
    def learner_step(self, batch):
        """
        batch contains tensors of shape (T, B, *other_dims), where 
        - T = unroll_length (number of steps in the trajectory)
        - B = batch_size
        
        Keywords needed:
        - spatial_state
        - player_state
        - action_mask
        - main_action
        - categorical_indexes
        - spatial_indexes
        """
        spatial_state = batch['spatial_state'].to(self.device)
        player_state = batch['player_state'].to(self.device)
        action_mask = batch['action_mask'].to(self.device)
        main_action = batch['main_action'].to(self.device)
        categorical_indexes = batch['categorical_indexes'].to(self.device)
        spatial_indexes = batch['spatial_indexes'].to(self.device)
        if debug:
            print("spatial_state.shape ", spatial_state.shape)
            print("player_state.shape ", player_state.shape)
            print("action_mask.shape ", action_mask.shape)
            print("main_action.shape ", main_action.shape)
            print("categorical_indexes.shape ", categorical_indexes.shape)
            print("spatial_indexes.shape ", spatial_indexes.shape)
            
        # useful dimensions
        T = spatial_state.shape[0]
        B = spatial_state.shape[1]
        res = self.screen_res[0]
        
        # merge all batch and time dimensions
        spatial_state = spatial_state.view((-1,)+spatial_state.shape[2:])
        player_state = player_state.view((-1,)+player_state.shape[2:])
        action_mask = action_mask.view((-1,)+action_mask.shape[2:])
        main_action = main_action.view((-1,)+main_action.shape[2:])
        # how and when to remove padding without making any error?
        categorical_indexes = categorical_indexes.view((-1,)+categorical_indexes.shape[2:])
        spatial_indexes = spatial_indexes.view((-1,)+spatial_indexes.shape[2:])
        
        if debug:
            print("After view: ")
            print("spatial_state.shape ", spatial_state.shape)
            print("player_state.shape ", player_state.shape)
            print("action_mask.shape ", action_mask.shape)
            print("main_action.shape ", main_action.shape)
            print("categorical_indexes.shape ", categorical_indexes.shape)
            print("spatial_indexes.shape ", spatial_indexes.shape)
            
        log_probs, spatial_features, nonspatial_features = self.pi(spatial_state, player_state, action_mask)
        log_prob = log_probs[range(len(main_action)), main_action]
        entropy = torch.sum(torch.exp(log_prob) * log_prob) # negative entropy of the main actions
        
        categorical_log_probs = self.categorical_params_net.get_log_probs(nonspatial_features)\
            .view(B*T, self.n_categorical_args, self.categorical_params_net.max_size)
        categorical_mask = categorical_arg_mask[main_action,:].view(-1,self.n_categorical_args)
        categorical_indexes = categorical_indexes[categorical_indexes!=-1] # remove padding
        batch_index = categorical_mask.nonzero()[:,0]
        arg_index = categorical_mask.nonzero()[:,1]
        categorical_log_prob = categorical_log_probs[batch_index, arg_index, categorical_indexes]
        log_prob.index_add_(0, batch_index, categorical_log_prob)
        
        # repeat for spatial params
        spatial_log_probs = self.spatial_params_net.get_log_probs(spatial_features)\
            .view(B*T, self.n_spatial_args, res**2)
        spatial_mask = spatial_arg_mask[main_action,:].view(-1,self.n_spatial_args)
        spatial_indexes = spatial_indexes[spatial_indexes!=-1] # remove padding
        batch_index = spatial_mask.nonzero()[:,0]
        arg_index = spatial_mask.nonzero()[:,1]
        spatial_log_prob = spatial_log_probs[batch_index, arg_index, spatial_indexes]
        log_prob.index_add_(0, batch_index, spatial_log_prob)
        
        baseline = self.V_critic(nonspatial_features=nonspatial_features)
        
        return dict(log_prob=log_prob, baseline=baseline, entropy=entropy)
    
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
        assert batch_size == 1, "Expected batch_size of 1, got %d"%batch_size
        parallel_args, parallel_log_prob, parallel_sampled_indexes = self.spatial_params_net(spatial_features)
       
        # Select only spatial arguments needed by sampled actions
        
        arg_mask = self.spatial_arg_mask[actions,:] # shape (batch_size, n_spatial_args)
        batch_pos = arg_mask.nonzero()[0]
        arg_pos = arg_mask.nonzero()[1]
        args = parallel_args[batch_pos, arg_pos]
        arg_list = [list(args[batch_pos==i]) for i in range(batch_size)]
        
        # check this
        torch_mask = torch.tensor(arg_mask, dtype=torch.bool)
        # parallel_sampled_indexes: shape (1, n_spatial_args)
        # indexes: shape (selected_args, ) # depends on the action(s) in the batch
        indexes = parallel_sampled_indexes[torch_mask] 
        # padded_indexes: shape (max_num_spatial_args, )
        padded_indexes = self.pad_to_len(indexes, self.max_num_spatial_args) 
        
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

        return arg_list, log_prob, indexes
    
    def sample_categorical_params(self, categorical_features, actions):
        """
        Input
        -----
        categorical_features: tensor, (batch_size, n_channels, screen_res, screen_res)
        actions: array, (batch_size,)
        """
        batch_size = actions.shape[0]
        assert batch_size == 1, "Expected batch_size of 1, got %d"%batch_size
        parallel_args, parallel_log_prob, parallel_sampled_indexes = self.categorical_params_net(categorical_features)
        arg_mask = self.categorical_arg_mask[actions,:] # shape (batch_size, n_spatial_args)
        
        # select correct arguments and indexes
        
        batch_pos = arg_mask.nonzero()[0]
        arg_pos = arg_mask.nonzero()[1]
        args = parallel_args[batch_pos, arg_pos]
        arg_list = [list(args[batch_pos==i]) for i in range(batch_size )]
        
        torch_mask = torch.tensor(arg_mask, dtype=torch.bool)
        # parallel_sampled_indexes: shape (1, n_categorical_args)
        # indexes: shape (selected_args, ) # depends on the action(s) in the batch
        indexes = parallel_sampled_indexes[torch_mask]  
        # padded_indexes: shape (max_num_categorical_args, )
        padded_indexes = self.pad_to_len(indexes, self.max_num_categorical_args) 
        
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

        return arg_list, log_prob, padded_indexes
    
    def sample_params(self, nonspatial_features, spatial_features, main_actions):
        cat_arg_list, cat_log_prob, cat_arg_indexes = self.sample_categorical_params(nonspatial_features, main_actions)
        spa_arg_list, spa_log_prob, spa_arg_indexes = self.sample_spatial_params(spatial_features, main_actions)
        
        # merge arg lists
        assert len(cat_arg_list) == len(spa_arg_list), ("Expected same length for arg lists", \
                                                                len(cat_arg_list), len(spa_arg_list))
        
        assert cat_log_prob.shape == spa_log_prob.shape, ("Expected same log_prob shape", \
                                                                 cat_log_prob.shape, spa_log_prob.shape)
        log_prob = cat_log_prob + spa_log_prob
        arg_list = []
        for cat, spa in zip(cat_arg_list, spa_arg_list):
            args = []
            if len(cat) != 0:
                args.append(cat)
            args += [list(s) for s in spa] # hopefully is the right format [[arg1],[arg2],...] x batch time
            arg_list.append(args)
            
        args_indexes = {'categorical_indexes':cat_arg_indexes, 'spatial_indexes':spa_arg_indexes}
        return arg_list, log_prob, args_indexes
    
    @staticmethod
    def pad_to_len(t, length, fill_value=-1):
        """ Assuming t of shape L <= length """
        shape = t.shape 
        if len(shape) == 2:
            # handle case with batch dimension = 1
            assert shape[0] == 1, "can only handle case with batch dim = 1"
            t = t.flatten() 
        elif len(shape) > 2:
            raise Exception("Trying to pad array with shape ", shape)
        else:
            continue
        assert t.shape[0] <= length, "tensor too long to be padded"
        padding = torch.ones(length-len(t), dtype=torch.int64)*-1
        padded_t = torch.cat([t, padding])
        return padded_t.view(shape[0]+(-1,))
    
        