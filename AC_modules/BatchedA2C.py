import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical
from AC_modules.ActorCriticArchitecture import *
from pysc2.lib import actions

debug = False

class SpatialA2C():
    """
    Advantage Actor-Critic RL agent restricted to MoveToBeacon StarCraftII minigame.
    
    Notes
    -----
      
    """ 
    
    def __init__(self, action_space, env, spatial_model, nonspatial_model, spatial_dict, 
                 nonspatial_dict, n_features, n_channels, gamma, action_dict=None, H=1e-3, n_steps=20, device='cpu'):
        """
        Parameters
        ----------
        action_space: int
            Number of (discrete) possible actions to take
            Does not count the actions'parameters
        env: SC2Env instance
            Needed for action specs
        spatial_model: nn.Module (not intialized)
            Takes as input (batch, in_channels, resolution, resolution)
            Returns (batch, n_channels, resolution, resolution)
        nonspatial_model: nn.Module (not intialized)
            Takes as input (batch, n_channels, resolution, resolution)
            Returns (batch, n_features)
        spatial_dict, nonspatial_dict: dict
            HPs of the two models, used as spatial_model(**spatial_dict)
            to initialize them
        n_features: int
            Output dim of nonspatial_model
        n_channels: int
            Output channels of spatial_model and input channels for nonspatial_model
            and SpatialParameters networks
        gamma: float in [0,1]
            Discount factor
        action_dict: dict (default None)
            Dictionary associating to every action in the action space a SC2 action id.
            If action_dict = None, it will be set by default to {0:0, 1:7, 2:12}, meaning
            respectively no_op, select_army, attack_screen.
        H: float (default 1e-3)
            Entropy multiplicative factor in actor's loss
        n_steps: int (default 20)
            Number of steps considered in TD update
        device: str in {'cpu','cuda'}
            Whether to use CPU or GPU
        """
        
        self.gamma = gamma
        self.n_actions = action_space
        self.n_steps = n_steps
        self.H = H
        self.AC = SpatialActorCritic(action_space, env, spatial_model, nonspatial_model, spatial_dict, 
                                     nonspatial_dict, n_features, n_channels, action_dict=action_dict)
        self.device = device 
        self.AC.to(self.device) 

    def step(self, state, action_mask):
        state = torch.from_numpy(state).float().to(self.device)
        action_mask = torch.tensor(action_mask).to(self.device)
        
        log_probs, spatial_features, nonspatial_features = self.AC.pi(state, action_mask)
        probs = torch.exp(log_probs)
        entropy = self.compute_entropy(probs)
        a = Categorical(probs).sample()
        a = a.detach().cpu().numpy()
        log_prob = log_probs[range(len(a)), a]
        
        args, args_log_prob, args_entropy = self.get_arguments(spatial_features, nonspatial_features, a)
        log_prob = log_prob + args_log_prob
        entropy = entropy + args_entropy

        action_id = np.array([self.AC.action_dict[act] for act in a])
        action = [actions.FunctionCall(action_id[i], args[i]) for i in range(len(action_id))]

        return action, log_prob, torch.mean(entropy)

    def get_arguments(self, spatial_features, nonspatial_features, action):
        """
        Samples all possible arguments for each sample in the batch, then selects only those that
        apply to the selected actions and returns a list containing the list of arguments for every 
        sampled action, the logarithm of the probability of sampling those arguments and the entropy 
        of their distributions. If an action has more arguments the log probs and the entropies returned
        are the sum of all those of the single arguments.
        """
        ### Sample and store each argument with its log prob and entropy ###
        results = {}    
        for arg_name in self.AC.arguments_dict.keys():
            if self.AC.arguments_type[arg_name] == 'categorical':
                arg_sampled, log_prob, probs = self.AC.sample_param(nonspatial_features, arg_name)
            elif self.AC.arguments_type[arg_name] == 'spatial':
                arg_sampled, log_prob, probs = self.AC.sample_param(spatial_features, arg_name)
            else:
                raise Exception("argument type for "+arg_name+" not understood")  
            entropy = self.compute_entropy(probs)
            results[arg_name] = (arg_sampled, log_prob, entropy)
           
        ### For every action get the list of arguments and their log prob and entropy ###
        args, args_log_prob, args_entropy = [], [], []
        for i, a in enumerate(action):
            # default return values if no argument is sampled (like if there was a single value obtained with p=1)
            arg = []
            arg_log_prob = torch.tensor([0]).float().to(self.device)
            entropies = torch.tensor([0]).float().to(self.device)
            
            arg_names = self.AC.act_to_arg_names[a]
            values = list( map(results.get, arg_names) )
            if len(values) != 0:
                for j in range(len(values)):
                    # j is looping on the tuples (arg, log_prob, ent)
                    # Second index is for accessing tuples items
                    # i is for the sample index inside the batch
                    arg.append(list(values[j][0][i]))
                    arg_log_prob = arg_log_prob + values[j][1][i] # sum log_probs
                    entropies = entropies + values[j][2][i] # sum entropies
            args.append(arg)
            args_log_prob.append(arg_log_prob) 
            args_entropy.append(entropies)
            
        args_log_prob = torch.stack(args_log_prob, axis=0).squeeze()
        args_entropy = torch.stack(args_entropy, axis=0).squeeze()
        return args, args_log_prob, args_entropy
 
    def compute_entropy(self, probs):
        """
        Computes NEGATIVE entropy of a batch (b, n_actions) of probabilities.
        Returns the entropy of each sample in the batch (b,)
        """
        probs = probs + torch.tensor([1e-5]).float().to(self.device) # add a small regularization to probs 
        entropy = torch.sum(probs*torch.log(probs), axis=1)
        return entropy
    
    def compute_ac_loss(self, rewards, log_probs, entropies, states, done, bootstrap, trg_states): 
        # merge batch and episode dimensions
        old_states = torch.tensor(states).float().to(self.device).reshape((-1,)+states.shape[2:])

        average_n_steps = False # TRY ME
        if average_n_steps:
            # Use as V target the mean of 1-step to n-step V targets
            V_trg = []
            for n in range(1, self.n_steps + 1):
                n_step_V_trg = self.compute_n_step_V_trg(n, rewards, done, bootstrap, states)
                V_trg.append(n_step_V_trg)
            V_trg = torch.mean(torch.stack(V_trg, axis=0), axis=0)
        else:
            V_trg = self.compute_n_step_V_trg(self.n_steps, rewards, done, bootstrap, states)
            
        ### Wrap variables into tensors - merge batch and episode dimensions ###    
        log_probs = torch.stack(log_probs).to(self.device).transpose(1,0).reshape(-1)
        entropies = torch.stack(entropies, axis=0).to(self.device).reshape(-1)
        
        ### Compute critic and actor losses ###
        critic_loss = self.compute_critic_loss(old_states, V_trg)
        actor_loss, entropy = self.compute_actor_loss(log_probs, entropies, old_states, V_trg)

        return critic_loss, actor_loss, entropy
    
    def compute_n_step_V_trg(self, n_steps, rewards, done, bootstrap, states):
        """
        Compute m-steps value target, with m = min(n, steps-to-episode-end).
        Formula (for precisely n-steps):
            V^{(n)}(t) = \sum_{k=0}^{n-1} gamma^k r_{t+k} + gamma^n * V(s_t)
        """
        n_step_rewards, episode_mask, n_steps_mask_b = self.compute_n_step_rewards(rewards, done, n_steps)
        done[bootstrap] = False 
        new_states, Gamma_V, done = self.compute_n_step_states(states, done, episode_mask, n_steps_mask_b)
        
        new_states = torch.tensor(new_states).float().to(self.device).reshape((-1,)+states.shape[2:])
        done = torch.LongTensor(done.astype(int)).to(self.device).reshape(-1)
        n_step_rewards = torch.tensor(n_step_rewards).float().to(self.device).reshape(-1)
        Gamma_V = torch.tensor(Gamma_V).float().to(self.device).reshape(-1)
        
        with torch.no_grad():
            V_pred = self.AC.V_critic(new_states).squeeze()
            V_trg = (1-done)*Gamma_V*V_pred + n_step_rewards
            V_trg = V_trg.squeeze()
        return V_trg
    
    def compute_critic_loss(self, old_states, V_trg):
        V = self.AC.V_critic(old_states).squeeze()
        loss = F.mse_loss(V, V_trg)
        return loss
    
    def compute_actor_loss(self, log_probs, entropies, old_states, V_trg):
        with torch.no_grad():
            V_pred = self.AC.V_critic(old_states).squeeze()
        A = V_trg - V_pred
        policy_gradient = - log_probs*A
        policy_grad = torch.mean(policy_gradient)
        entropy = torch.mean(entropies)
        loss = policy_grad + self.H*entropy
        return loss, entropy
                
    def compute_n_step_rewards(self, rewards, done, n_steps=None):
        """
        Computes n-steps discounted reward. 
        Note: the rewards considered are AT MOST n, but can be less for the last n-1 elements.
        """
        if n_steps is None:
            n_steps = self.n_steps
        B = done.shape[0]
        T = done.shape[1]
        
        # Compute episode mask (i-th row contains 1 if col j is in the same episode of col i, 0 otherwise)
        episode_mask = [[] for _ in range(B)]
        last = [-1 for _ in range(B)]
        xs, ys = np.nonzero(done)
        
        # Add done at the end of every batch to avoid exceptions -> not used in real target computations
        xs = np.concatenate([xs, np.arange(B)])
        ys = np.concatenate([ys, np.full(B, T-1)])
        for x, y in zip(xs, ys):
            m = [1 if (i > last[x] and i <= y) else 0 for i in range(T)]
            for _ in range(y-last[x]):
                episode_mask[x].append(m)
            last[x] = y
        episode_mask = np.array(episode_mask)
        
        # Compute n-steps mask and repeat it B times
        n_steps_mask = []
        for i in range(T):
            m = [1 if (j>=i and j<i+n_steps) else 0 for j in range(T)]
            n_steps_mask.append(m)
        n_steps_mask = np.array(n_steps_mask)
        n_steps_mask_b = np.repeat(n_steps_mask[np.newaxis,...] , B, axis=0)
        
        # Broadcast rewards to use multiplicative masks
        rewards_repeated = np.repeat(rewards[:,np.newaxis,:], T, axis=1)
        
        # Exponential discount factor
        Gamma = np.array([self.gamma**i for i in range(T)]).reshape(1,-1)
        n_steps_r = (Gamma*rewards_repeated*episode_mask*n_steps_mask_b).sum(axis=2)/Gamma
        return n_steps_r, episode_mask, n_steps_mask_b
    
    def compute_n_step_states(self, trg_states, done, episode_mask, n_steps_mask_b):
        """
        Computes n-steps target states (to be used by the critic as target values together with the
        n-steps discounted reward). For last n-1 elements the target state is the last one available.
        Adjusts also the `done` mask used for disabling the bootstrapping in the case of terminal states
        and returns Gamma_V, that are the discount factors for the target state-values, since they are 
        n-steps away (except for the last n-1 states, whose discount is adjusted accordingly).
        
        Return
        ------
        new_states, Gamma_V, done: arrays with first dimension = len(states)-1
        """
        
        B = done.shape[0]
        T = done.shape[1]
        V_mask = episode_mask*n_steps_mask_b
        b, x, y = np.nonzero(V_mask)
        V_trg_index = [[] for _ in range(B)]
        for b_i in range(B):
            valid_x = (b==b_i)
            for i in range(T):
                matching_x = (x==i)
                V_trg_index[b_i].append(y[valid_x*matching_x][-1])
        V_trg_index = np.array(V_trg_index)
        
        cols = np.array([], dtype=np.int)
        rows = np.array([], dtype=np.int)
        for i, v in enumerate(V_trg_index):
            cols = np.concatenate([cols, v], axis=0)
            row = np.full(V_trg_index.shape[1], i)
            rows = np.concatenate([rows, row], axis=0)
        new_states = trg_states[rows, cols].reshape(trg_states.shape)
        pw = V_trg_index - np.arange(V_trg_index.shape[1]) + 1
        Gamma_V = self.gamma**pw
        shifted_done = done[rows, cols].reshape(done.shape)
        return new_states, Gamma_V, shifted_done

### Using separate networks for same parameter belonging to different actions ###
class SpatialA2C_v2(SpatialA2C):
    def __init__(self, action_space, env, spatial_model, nonspatial_model, 
                 spatial_dict,  nonspatial_dict, n_features, n_channels,
                 gamma, action_dict=None, H=1e-3, n_steps=20, device='cpu'):
        # Do not use super().__init__()
        self.gamma = gamma
        self.n_actions = action_space
        self.n_steps = n_steps
        self.H = H
        self.AC = SpatialActorCritic_v2(action_space, env, spatial_model, nonspatial_model, spatial_dict, 
                                     nonspatial_dict, n_features, n_channels, action_dict)
        self.device = device 
        self.AC.to(self.device)
        
### Conditioning parameters on actions ###
    
class SpatialA2C_v1(SpatialA2C):
    def __init__(self, action_space, env, spatial_model, nonspatial_model, 
                 spatial_dict,  nonspatial_dict, n_features, n_channels, embed_dim,
                 gamma, action_dict=None, H=1e-3, n_steps=20, device='cpu'):
        # Do not use super().__init__()
        self.gamma = gamma
        self.n_actions = action_space
        self.n_steps = n_steps
        self.H = H
        self.AC = SpatialActorCritic_v1(action_space, env, spatial_model, nonspatial_model, spatial_dict, 
                                     nonspatial_dict, n_features, n_channels, action_dict, embed_dim)
        self.device = device 
        self.AC.to(self.device)
        
    def step(self, state, action_mask):
        state = torch.from_numpy(state).float().to(self.device)
        action_mask = torch.tensor(action_mask).to(self.device)
        
        log_probs, spatial_features, nonspatial_features = self.AC.pi(state, action_mask)
        probs = torch.exp(log_probs)
        entropy = self.compute_entropy(probs)
        a = Categorical(probs).sample()
        a = a.detach().cpu().numpy()
        embedded_a = self._embed_action(a)
        
        log_prob = log_probs[range(len(a)), a]
        
        # Concatenate embedded action to spatial and nonspatial features
        spatial_features = self._cat_action_to_spatial(embedded_a, spatial_features)
        nonspatial_features = self._cat_action_to_nonspatial(embedded_a, nonspatial_features)
        
        args, args_log_prob, args_entropy = self.get_arguments(spatial_features, nonspatial_features, a)
        log_prob = log_prob + args_log_prob
        entropy = entropy + args_entropy

        action_id = np.array([self.AC.action_dict[act] for act in a])
        action = [actions.FunctionCall(action_id[i], args[i]) for i in range(len(action_id))]

        return action, log_prob, torch.mean(entropy)

    def get_arguments(self, spatial_features, nonspatial_features, action):
        """
        Samples all possible arguments for each sample in the batch, then selects only those that
        apply to the selected actions and returns a list containing the list of arguments for every 
        sampled action, the logarithm of the probability of sampling those arguments and the entropy 
        of their distributions. If an action has more arguments the log probs and the entropies returned
        are the sum of all those of the single arguments.
        """
        ### Sample and store each argument with its log prob and entropy ###
        results = {}    
        for arg_name in self.AC.arguments_dict.keys():
            if self.AC.arguments_type[arg_name] == 'categorical':
                arg_sampled, log_prob, probs = self.AC.sample_param(nonspatial_features, arg_name)
            elif self.AC.arguments_type[arg_name] == 'spatial':
                arg_sampled, log_prob, probs = self.AC.sample_param(spatial_features, arg_name)
            else:
                raise Exception("argument type for "+arg_name+" not understood")  
            entropy = self.compute_entropy(probs)
            results[arg_name] = (arg_sampled, log_prob, entropy)
           
        ### For every action get the list of arguments and their log prob and entropy ###
        args, args_log_prob, args_entropy = [], [], []
        for i, a in enumerate(action):
            # default return values if no argument is sampled (like if there was a single value obtained with p=1)
            arg = []
            arg_log_prob = torch.tensor([0]).float().to(self.device)
            entropies = torch.tensor([0]).float().to(self.device)
            
            arg_names = self.AC.act_to_arg_names[a]
            values = list( map(results.get, arg_names) )
            if len(values) != 0:
                for j in range(len(values)):
                    # j is looping on the tuples (arg, log_prob, ent)
                    # Second index is for accessing tuples items
                    # i is for the sample index inside the batch
                    arg.append(list(values[j][0][i]))
                    arg_log_prob = arg_log_prob + values[j][1][i] # sum log_probs
                    entropies = entropies + values[j][2][i] # sum entropies
            args.append(arg)
            args_log_prob.append(arg_log_prob) 
            args_entropy.append(entropies)
            
        args_log_prob = torch.stack(args_log_prob, axis=0).squeeze()
        args_entropy = torch.stack(args_entropy, axis=0).squeeze()
        return args, args_log_prob, args_entropy
    
    def _embed_action(self, action):
        a = torch.LongTensor(action).to(self.device)
        a = self.AC.embedding(a)
        return a
    
    def _cat_action_to_spatial(self, embedded_action, spatial_repr):
        """ 
        Assume spatial_repr of shape (B, n_channels, res, res).
        Cast embedded_action from (B, embedd_dim) to (B, embedd_dim, res, res)
        Concatenate spatial_repr with the broadcasted embedded action along the channel dim.
        """
        res = spatial_repr.shape[-1]
        embedded_action = embedded_action.reshape((embedded_action.shape[:2]+(1,1,)))
        spatial_a = embedded_action.repeat(1,1,res,res)
        spatial_repr = torch.cat([spatial_repr, spatial_a], dim=1)
        return spatial_repr
    
    def _cat_action_to_nonspatial(self, embedded_action, nonspatial_repr):
        """
        nonspatial_repr: (B, n_features)
        embedded_action: (B, embedd_dim)
        Concatenate them so that the result is of shape (B, n_features+embedd_dim)
        """
        return torch.cat([nonspatial_repr, embedded_action], dim=1)
    
# works similarly to v1 for the categorical parameters, but is different for the spatial ones
class SpatialA2C_v3(SpatialA2C):
    def __init__(self, action_space, env, spatial_model, nonspatial_model, 
                 spatial_dict,  nonspatial_dict, n_features, n_channels,
                 gamma, action_dict=None, H=1e-3, n_steps=20, device='cpu'):
        # Do not use super().__init__()
        self.gamma = gamma
        self.n_actions = action_space
        self.n_steps = n_steps
        self.H = H
        self.AC = SpatialActorCritic_v3(action_space, env, spatial_model, nonspatial_model, spatial_dict, 
                                     nonspatial_dict, n_features, n_channels, action_dict)
        self.device = device 
        self.AC.to(self.device)
        
    def step(self, state, action_mask):
        state = torch.from_numpy(state).float().to(self.device)
        action_mask = torch.tensor(action_mask).to(self.device)
        
        log_probs, spatial_features, nonspatial_features = self.AC.pi(state, action_mask)
        probs = torch.exp(log_probs)
        entropy = self.compute_entropy(probs)
        a = Categorical(probs).sample()
        a = a.detach().cpu().numpy()
        embedded_a = self._embed_action(a)
        
        log_prob = log_probs[range(len(a)), a]
        
        # Concatenate embedded action to nonspatial features only
        nonspatial_features = self._cat_action_to_nonspatial(embedded_a, nonspatial_features)
        
        args, args_log_prob, args_entropy = self.get_arguments(spatial_features, nonspatial_features, a, embedded_a)
        log_prob = log_prob + args_log_prob
        entropy = entropy + args_entropy

        action_id = np.array([self.AC.action_dict[act] for act in a])
        action = [actions.FunctionCall(action_id[i], args[i]) for i in range(len(action_id))]

        return action, log_prob, torch.mean(entropy)

    def get_arguments(self, spatial_features, nonspatial_features, action, embedded_a):
        """
        Samples all possible arguments for each sample in the batch, then selects only those that
        apply to the selected actions and returns a list containing the list of arguments for every 
        sampled action, the logarithm of the probability of sampling those arguments and the entropy 
        of their distributions. If an action has more arguments the log probs and the entropies returned
        are the sum of all those of the single arguments.
        """
        ### Sample and store each argument with its log prob and entropy ###
        results = {}    
        for arg_name in self.AC.arguments_dict.keys():
            if self.AC.arguments_type[arg_name] == 'categorical':
                arg_sampled, log_prob, probs = self.AC.sample_param(arg_name, nonspatial_features)
            elif self.AC.arguments_type[arg_name] == 'spatial':
                arg_sampled, log_prob, probs = self.AC.sample_param(arg_name, spatial_features, embedded_a)
            else:
                raise Exception("argument type for "+arg_name+" not understood")  
            entropy = self.compute_entropy(probs)
            results[arg_name] = (arg_sampled, log_prob, entropy)
           
        ### For every action get the list of arguments and their log prob and entropy ###
        args, args_log_prob, args_entropy = [], [], []
        for i, a in enumerate(action):
            # default return values if no argument is sampled (like if there was a single value obtained with p=1)
            arg = []
            arg_log_prob = torch.tensor([0]).float().to(self.device)
            entropies = torch.tensor([0]).float().to(self.device)
            
            arg_names = self.AC.act_to_arg_names[a]
            values = list( map(results.get, arg_names) )
            if len(values) != 0:
                for j in range(len(values)):
                    # j is looping on the tuples (arg, log_prob, ent)
                    # Second index is for accessing tuples items
                    # i is for the sample index inside the batch
                    arg.append(list(values[j][0][i]))
                    arg_log_prob = arg_log_prob + values[j][1][i] # sum log_probs
                    entropies = entropies + values[j][2][i] # sum entropies
            args.append(arg)
            args_log_prob.append(arg_log_prob) 
            args_entropy.append(entropies)
            
        args_log_prob = torch.stack(args_log_prob, axis=0).squeeze()
        args_entropy = torch.stack(args_entropy, axis=0).squeeze()
        return args, args_log_prob, args_entropy
    
    def _embed_action(self, action):
        a = torch.LongTensor(action).to(self.device)
        a = self.AC.embedding(a)
        return a
    
    def _cat_action_to_nonspatial(self, embedded_action, nonspatial_repr):
        """
        nonspatial_repr: (B, n_features)
        embedded_action: (B, embedd_dim)
        Concatenate them so that the result is of shape (B, n_features+embedd_dim)
        """
        return torch.cat([nonspatial_repr, embedded_action], dim=1)