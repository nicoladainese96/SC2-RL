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
    1. Uses n-step updates
    2. Adds entropy regularization H*\sum_{t=0}^T Entropy[pi(*|s_t)] to the actor loss
    3. Uses shared and unconditioned networks for parameters of different actions that have the same name,
       e.g. Attack_screen and Move_screen require the same 2 parameters 'queue' and 'screen', that are provided
       by 2 networks AC.arguments_networks['queue'] and AC.arguments_networks['screen'], disregarding which action 
       requires those parameters.
    """ 
    
    def __init__(self, action_space, env, spatial_model, nonspatial_model, spatial_dict, 
                 nonspatial_dict, n_features, n_channels, gamma=0.99, action_dict=None, H=1e-3, n_steps=20, device='cpu'):
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
        #self.n_actions = action_space
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
        entropy = self.compute_entropy(log_probs)
        probs = torch.exp(log_probs)
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
                arg_sampled, log_prob, log_probs = self.AC.sample_param(nonspatial_features, arg_name)
            elif self.AC.arguments_type[arg_name] == 'spatial':
                arg_sampled, log_prob, log_probs = self.AC.sample_param(spatial_features, arg_name)
            else:
                raise Exception("argument type for "+arg_name+" not understood")  
            entropy = self.compute_entropy(log_probs)
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
 
    def compute_entropy(self, log_probs):
        """
        Computes NEGATIVE entropy of a batch (b, n_actions) of log probabilities.
        Returns the entropy of each sample in the batch (b,)
        """
        probs = torch.exp(log_probs) 
        distr = Categorical(probs=probs)
        entropy = -distr.entropy()
        return entropy
    
    def compute_ac_loss(self, rewards, log_probs, entropies, states, done, bootstrap, trg_states): 
        assert states.shape == trg_states.shape, \
            ("Expected states and trg_states with same shape ", states.shape, trg_states.shape)
        # merge batch and episode dimensions
        old_states = torch.tensor(states).float().to(self.device).reshape((-1,)+states.shape[2:])

        average_n_steps = False # TRY ME
        if average_n_steps:
            # Use as V target the mean of 1-step to n-step V targets
            V_trg = []
            for n in range(1, self.n_steps + 1):
                n_step_V_trg = self.compute_n_step_V_trg(n, rewards, done, bootstrap, trg_states)
                V_trg.append(n_step_V_trg)
            V_trg = torch.mean(torch.stack(V_trg, axis=0), axis=0)
        else:
            V_trg = self.compute_n_step_V_trg(self.n_steps, rewards, done, bootstrap, trg_states)
            
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

########################################################################################################################

class SpatialA2C_v2(SpatialA2C):
    """
    Difference from SpatialA2C:
        1. Using separate networks for same parameter belonging to different actions
        e.g. Attack_screen and Move_screen require the same 2 parameters 'queue' and 'screen', that this time are 
        provided each one by a single network. For example AC.arguments_networks['Attack_screen/queue'] will have the
        same architecture but completely independent weights from AC.arguments_networks['Move_screen/queue']
    """
    def __init__(self, action_space, env, spatial_model, nonspatial_model, 
                 spatial_dict,  nonspatial_dict, n_features, n_channels,
                 gamma=0.99, action_dict=None, H=1e-3, n_steps=20, device='cpu'):
        # Do not use super().__init__()
        self.gamma = gamma
        #self.n_actions = action_space
        self.n_steps = n_steps
        self.H = H
        self.AC = SpatialActorCritic_v2(action_space, env, spatial_model, nonspatial_model, spatial_dict, 
                                     nonspatial_dict, n_features, n_channels, action_dict)
        self.device = device 
        self.AC.to(self.device)

    def step(self, state, action_mask):
        """Use this to regularize only on the entropy of the main action"""
        state = torch.from_numpy(state).float().to(self.device)
        action_mask = torch.tensor(action_mask).to(self.device)
        
        log_probs, spatial_features, nonspatial_features = self.AC.pi(state, action_mask)
        entropy = self.compute_entropy(log_probs)
        probs = torch.exp(log_probs)
        a = Categorical(probs).sample()
        a = a.detach().cpu().numpy()
        log_prob = log_probs[range(len(a)), a]
        
        args, args_log_prob, args_entropy = self.get_arguments(spatial_features, nonspatial_features, a)
        log_prob = log_prob + args_log_prob
        #entropy = entropy + args_entropy

        action_id = np.array([self.AC.action_dict[act] for act in a])
        action = [actions.FunctionCall(action_id[i], args[i]) for i in range(len(action_id))]

        return action, log_prob, torch.mean(entropy)

########################################################################################################################

class FullSpaceA2C(SpatialA2C): 
    def __init__(self, action_space, env, spatial_model, nonspatial_model, spatial_dict, nonspatial_dict, 
                 n_features, n_channels, gamma=0.99, action_dict=None, H=1e-2, n_steps=20, device='cpu'):
        self.gamma = gamma
        #self.n_actions = action_space
        self.n_steps = n_steps
        self.H = H
        self.AC = SpatialActorCritic_v4(action_space, env, spatial_model, nonspatial_model, spatial_dict, 
                                     nonspatial_dict, n_features, n_channels, action_dict=action_dict)
        self.device = device 
        self.AC.to(self.device) 
        
    def step(self, state, action_mask):
        spatial_state = state['spatial']
        player_state = state['player']
        spatial_state = torch.from_numpy(spatial_state).float().to(self.device)
        player_state = torch.from_numpy(player_state).float().to(self.device)
        action_mask = torch.tensor(action_mask).to(self.device)
        
        log_probs, spatial_features, nonspatial_features = self.AC.pi(spatial_state, player_state, action_mask)
        entropy = self.compute_entropy(log_probs)
        probs = torch.exp(log_probs)
        a = Categorical(probs).sample()
        a = a.detach().cpu().numpy()
        log_prob = log_probs[range(len(a)), a]
        
        args, args_log_prob, args_entropy = self.get_arguments(spatial_features, nonspatial_features, a)
        log_prob = log_prob + args_log_prob
        # Use only entropy of main actions for regularization
        #entropy = entropy + args_entropy

        action_id = np.array([self.AC.action_dict[act] for act in a])
        action = [actions.FunctionCall(action_id[i], args[i]) for i in range(len(action_id))]

        return action, log_prob, torch.mean(entropy)
    
    def compute_ac_loss(self, rewards, log_probs, entropies, states, done, bootstrap, trg_states): 
        """
        Notes:
        - Convention for tensors: batch first
        """
        
        # from list of dictionaries of arrays to 2 separate arrays
        spatial_states_lst = [s['spatial'] for s in states] #[(batch, other dims) x traj_len times]
        player_states_lst = [s['player'] for s in states] 
        spatial_states = torch.tensor(spatial_states_lst).float().to(self.device).transpose(1,0)
        player_states = torch.tensor(player_states_lst).float().to(self.device).transpose(1,0)
        
        spatial_states_lst_trg = [s['spatial'] for s in trg_states]
        player_states_lst_trg = [s['player'] for s in trg_states]
        spatial_states_trg = torch.tensor(spatial_states_lst_trg).float().to(self.device).transpose(1,0)
        player_states_trg = torch.tensor(player_states_lst_trg).float().to(self.device).transpose(1,0)
        
        # merge batch and episode dimensions
        old_spatial_states = spatial_states.reshape((-1,)+spatial_states.shape[2:])
        old_player_states = player_states.reshape((-1,)+player_states.shape[2:])
        
        average_n_steps = False # TRY ME
        if average_n_steps:
            # Use as V target the mean of 1-step to n-step V targets
            V_trg = []
            for n in range(1, self.n_steps + 1):
                n_step_V_trg = self.compute_n_step_V_trg(n, rewards, done, bootstrap, 
                                                         spatial_states_trg, player_states_trg)
                V_trg.append(n_step_V_trg)
            V_trg = torch.mean(torch.stack(V_trg, axis=0), axis=0)
        else:
            V_trg = self.compute_n_step_V_trg(self.n_steps, rewards, done, bootstrap, 
                                                         spatial_states_trg, player_states_trg)
            
        ### Wrap variables into tensors - merge batch and episode dimensions ###    
        log_probs = torch.stack(log_probs).to(self.device).transpose(1,0).reshape(-1)
        entropies = torch.stack(entropies, axis=0).to(self.device).reshape(-1)
        
        ### Compute critic and actor losses ###
        critic_loss = self.compute_critic_loss(old_spatial_states, old_player_states, V_trg)
        actor_loss, entropy = self.compute_actor_loss(log_probs, entropies, old_spatial_states, old_player_states, V_trg)

        return critic_loss, actor_loss, entropy
    
    def compute_n_step_V_trg(self, n_steps, rewards, done, bootstrap, spatial_states, player_states):
        """
        Compute m-steps value target, with m = min(n, steps-to-episode-end).
        Formula (for precisely n-steps):
            V^{(n)}(t) = \sum_{k=0}^{n-1} gamma^k r_{t+k} + gamma^n * V(s_t)
        """
        n_step_rewards, episode_mask, n_steps_mask_b = self.compute_n_step_rewards(rewards, done, n_steps)
        done[bootstrap] = False 
        # Check those
        new_spatial_states, Gamma_V, done = self.compute_n_step_states(spatial_states, done,
                                                                       episode_mask, n_steps_mask_b)
        new_player_states, _, _ = self.compute_n_step_states(player_states, done,
                                                                       episode_mask, n_steps_mask_b)
        
        new_spatial_states = new_spatial_states.reshape((-1,)+spatial_states.shape[2:])
        new_player_states = new_player_states.reshape((-1,)+player_states.shape[2:])
        
        done = torch.LongTensor(done.astype(int)).to(self.device).reshape(-1)
        n_step_rewards = torch.tensor(n_step_rewards).float().to(self.device).reshape(-1)
        Gamma_V = torch.tensor(Gamma_V).float().to(self.device).reshape(-1)
        
        with torch.no_grad():
            V_pred = self.AC.V_critic(new_spatial_states, new_player_states).squeeze()
            V_trg = (1-done)*Gamma_V*V_pred + n_step_rewards
            V_trg = V_trg.squeeze()
        return V_trg
    
    def compute_critic_loss(self, old_spatial_states, old_player_states, V_trg):
        V = self.AC.V_critic(old_spatial_states, old_player_states).squeeze()
        loss = F.mse_loss(V, V_trg)
        return loss
    
    def compute_actor_loss(self, log_probs, entropies, old_spatial_states, old_player_states, V_trg):
        with torch.no_grad():
            V_pred = self.AC.V_critic(old_spatial_states, old_player_states).squeeze()
        A = V_trg - V_pred
        policy_gradient = - log_probs*A
        policy_grad = torch.mean(policy_gradient)
        entropy = torch.mean(entropies)
        loss = policy_grad + self.H*entropy
        return loss, entropy
    
########################################################################################################################

class FullSpaceA2C_v2(FullSpaceA2C):
    def __init__(self, env, spatial_model, nonspatial_model, spatial_dict, nonspatial_dict, 
                 n_features, n_channels, action_names, gamma=0.99, H=1e-2, n_steps=20, device='cpu'):
        self.gamma = gamma
        self.n_steps = n_steps
        self.H = H
        self.AC = ParallelActorCritic(env, spatial_model, nonspatial_model, spatial_dict, 
                                     nonspatial_dict, n_features, n_channels, action_names)
        self.device = device 
        self.AC.to(self.device) 
        
    def step(self, state, action_mask):
        spatial_state = state['spatial']
        player_state = state['player']
        spatial_state = torch.from_numpy(spatial_state).float().to(self.device)
        player_state = torch.from_numpy(player_state).float().to(self.device)
        action_mask = torch.tensor(action_mask).to(self.device)
        
        log_probs, spatial_features, nonspatial_features = self.AC.pi(spatial_state, player_state, action_mask)
        entropy = self.compute_entropy(log_probs)
        probs = torch.exp(log_probs)
        a = Categorical(probs).sample()
        a = a.detach().cpu().numpy()
        log_prob = log_probs[range(len(a)), a]
        
        args, args_log_prob = self.AC.sample_params(nonspatial_features, spatial_features, a)
        assert args_log_prob.shape == log_prob.shape, ("Shape mismatch between arg_log_prob and log_prob ",\
                                                      args_log_prob.shape, log_prob.shape)
        log_prob = log_prob + args_log_prob
        
        action_id = np.array([self.AC.action_table[act] for act in a])
        action = [actions.FunctionCall(action_id[i], args[i]) for i in range(len(action_id))]

        return action, log_prob, torch.mean(entropy)
########################################################################################################################

class GeneralA2C(FullSpaceA2C): 
    """
    Features:
    1. Supports automatically full action and state space
    2. Uses entropy regularization only on main actions
    3. Uses different networks for same parameters referred to different actions 
       (see SpatialA2C_v2)
    """
    def __init__(self, env, spatial_model, nonspatial_model, spatial_dict, nonspatial_dict, 
                 n_features, n_channels, gamma=0.99, H=1e-2, n_steps=20, device='cpu'):
        self.gamma = gamma
        self.n_steps = n_steps
        self.H = H
        self.AC = FullSpatialActorCritic(env, spatial_model, nonspatial_model, spatial_dict, 
                                         nonspatial_dict, n_features, n_channels)
        self.device = device 
        self.AC.to(self.device) 
        
    def step(self, state, action_mask):
        spatial_state = state['spatial']
        player_state = state['player']
        spatial_state = torch.from_numpy(spatial_state).float().to(self.device)
        player_state = torch.from_numpy(player_state).float().to(self.device)
        action_mask = torch.tensor(action_mask).to(self.device)
        
        log_probs, spatial_features, nonspatial_features = self.AC.pi(spatial_state, player_state, action_mask)
        entropy = self.compute_entropy(log_probs)
        probs = torch.exp(log_probs)
        a = Categorical(probs).sample()
        a = a.detach().cpu().numpy()
        log_prob = log_probs[range(len(a)), a]
        
        args, args_log_prob, args_entropy = self.get_arguments(spatial_features, nonspatial_features, a)
        log_prob = log_prob + args_log_prob

        # check if it's alright to pass integers as action identifiers
        #action_id = np.array([self.AC.action_dict[act] for act in a])
        action = [actions.FunctionCall(a[i], args[i]) for i in range(len(a))]
        if debug:
            print("action: ", action)
        return action, log_prob, torch.mean(entropy)
    
    def get_arguments(self, spatial_features, nonspatial_features, action):
        """
        1. Loop on actions
        2. Get arg_names
        3. Loop on them
        4. Discriminate between spatial and categorical
        5. Slice the spatial or nonspatial features selecting the index i along the batch axis, 
           then add again a fictious batch dimension with .unsqueeze(0)
        6. Collect the results 
        7. Stack list of results
        """
        ### For every action get the list of arguments and their log prob and entropy ###
        args, args_log_prob, args_entropy = [], [], []
        # 1)
        for i, a in enumerate(action):
            # default return values if no argument is sampled (like if there was a single value obtained with p=1)
            arg = []
            arg_log_prob = torch.tensor([0]).float().to(self.device)
            entropies = torch.tensor([0]).float().to(self.device)
            # 2)
            arg_names = self.AC.act_to_arg_names[a]
            if len(arg_names) != 0:
                # 3)
                for arg_name in arg_names:
                    # 4)
                    if self.AC.arguments_type[arg_name] == 'categorical':
                        # 5)
                        x = nonspatial_features[i].unsqueeze(0)
                        arg_sampled, log_prob, log_probs = self.AC.sample_param(x, arg_name)
                    elif self.AC.arguments_type[arg_name] == 'spatial':
                        x = spatial_features[i].unsqueeze(0)
                        arg_sampled, log_prob, log_probs = self.AC.sample_param(x, arg_name)
                    else:
                        raise Exception("argument type for "+arg_name+" not understood")  
                    entropy = self.compute_entropy(log_probs)

                    # 6)
                    #print("arg_sampled: ", arg_sampled)
                    #print("arg_sampled[0]: ", arg_sampled[0])
                    arg.append(arg_sampled[0])
                    arg_log_prob = arg_log_prob + log_prob 
                    entropies = entropies + entropy
            args.append(arg)
            args_log_prob.append(arg_log_prob) 
            args_entropy.append(entropies)

        # 7)
        args_log_prob = torch.stack(args_log_prob, axis=0).squeeze()
        args_entropy = torch.stack(args_entropy, axis=0).squeeze()
        return args, args_log_prob, args_entropy

    # Not used
    def get_arguments_old(self, spatial_features, nonspatial_features, action):
        """
        1) Samples all possible arguments for each sample in the batch
        2) Saves them together with their log prob and their entropy
        2) Selects only those arguments that apply to the selected actions (loop on batch dimension)
        3) Returns a list containing: 
            - the list of arguments for every sampled action, 
            - the logarithm of the probability of sampling those arguments and 
            - the entropy of their distributions. 

        Notes
        -----
        If an action has more arguments the log probs and the entropies returned
        are the sum of all those of the single arguments.
        """
        ### Sample and store each argument with its log prob and entropy ###
        results = {}    
        for arg_name in self.AC.arguments_names_lst:
            if self.AC.arguments_type[arg_name] == 'categorical':
                arg_sampled, log_prob, log_probs = self.AC.sample_param(nonspatial_features, arg_name)
            elif self.AC.arguments_type[arg_name] == 'spatial':
                arg_sampled, log_prob, log_probs = self.AC.sample_param(spatial_features, arg_name)
            else:
                raise Exception("argument type for "+arg_name+" not understood")  
            entropy = self.compute_entropy(log_probs)
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
    

########################################################################################################################

class SpatialA2C_MaxEnt(SpatialA2C_v2):
    """
    Maximum Entropy Reinforcement Learning objective following Schulman et al. 
    Equivalence Between Policy Gradients and Soft Q-Learning.
    
    Highlights:
        1. Augmented reward : r_t - H*D_KL[pi(*|s_t)|| uniform_policy] ~ r_t + H*Entropy[pi(*|s_t)]
        2. Augmented reward attached from computational graph 
    """
    def __init__(self, action_space, env, spatial_model, nonspatial_model, 
                 spatial_dict,  nonspatial_dict, n_features, n_channels,
                 gamma=0.99, action_dict=None, H=1e-3, n_steps=20, device='cpu'):
        super().__init__(action_space, env, spatial_model, nonspatial_model, 
                 spatial_dict,  nonspatial_dict, n_features, n_channels,
                 gamma, action_dict, H, n_steps, device)
      
    def step(self, state, action_mask):
        state = torch.from_numpy(state).float().to(self.device)
        action_mask = torch.tensor(action_mask).to(self.device)
        
        log_probs, spatial_features, nonspatial_features = self.AC.pi(state, action_mask)
        entropy = self.compute_entropy(log_probs)
        probs = torch.exp(log_probs)
        a = Categorical(probs).sample()
        a = a.detach().cpu().numpy()
        log_prob = log_probs[range(len(a)), a]
        
        args, args_log_prob, args_entropy = self.get_arguments(spatial_features, nonspatial_features, a)
        log_prob = log_prob + args_log_prob
        # to use only main action entropy comment next line
        #entropy = entropy + args_entropy

        action_id = np.array([self.AC.action_dict[act] for act in a])
        action = [actions.FunctionCall(action_id[i], args[i]) for i in range(len(action_id))]
        return action, log_prob, entropy
    
    def compute_entropy(self, log_probs):
        """
        Computes - D_KL(probs|uniform_probs)
        Computes POSITIVE entropy of a batch (b, n_actions) of log probabilities.
        Returns the entropy of each sample in the batch (b,)
        """
        probs = torch.exp(log_probs) 
        distr = Categorical(probs=probs)
        
        # \sum_{actions} -p(a) logp(a)
        entropy = distr.entropy() 

        # \sum_{actions} p(a) log(1\|actions|) = log(1\|actions|) = -log(|action|) < 0
        #cross_entropy = - torch.log(torch.ones(probs.shape)/probs.shape[-1]).mean(axis=-1).float().to(self.device)
        
        # - D_KL[pi|uniform]
        #entropy = entropy - cross_entropy 
        return entropy
    
    def compute_ac_loss(self, rewards, log_probs, entropies, states, done, bootstrap, trg_states): 
        assert states.shape == trg_states.shape, \
            ("Expected states and trg_states with same shape ", states.shape, trg_states.shape)
        # merge batch and episode dimensions
        old_states = torch.tensor(states).float().to(self.device).reshape((-1,)+states.shape[2:])
        entropies = torch.stack(entropies, axis=0).transpose(1,0).to(self.device)
        average_n_steps = False # TRY ME
        if average_n_steps:
            # Use as V target the mean of 1-step to n-step V targets
            V_trg = []
            for n in range(1, self.n_steps + 1):
                n_step_V_trg = self.compute_n_step_V_trg(n, rewards, done, bootstrap, trg_states, entropies)
                V_trg.append(n_step_V_trg)
            V_trg = torch.mean(torch.stack(V_trg, axis=0), axis=0)
        else:
            V_trg = self.compute_n_step_V_trg(self.n_steps, rewards, done, bootstrap, trg_states, entropies)
            
        ### Wrap variables into tensors - merge batch and episode dimensions ###    
        log_probs = torch.stack(log_probs).to(self.device).transpose(1,0).reshape(-1)
        
        ### Compute critic and actor losses ###
        critic_loss = self.compute_critic_loss(old_states, V_trg)
        actor_loss = self.compute_actor_loss(log_probs, old_states, V_trg)

        return critic_loss, actor_loss
    
    def compute_n_step_V_trg(self, n_steps, rewards, done, bootstrap, states, entropies):
        """
        Compute m-steps value target, with m = min(n, steps-to-episode-end).
        Formula (for precisely n-steps):
            V^{(n)}(t) = \sum_{k=0}^{n-1} gamma^k r_{t+k} + gamma^n * V(s_t)
        """
        n_step_rewards, episode_mask, n_steps_mask_b = self.compute_n_step_rewards(rewards, done, entropies, n_steps)
        done[bootstrap] = False 
        new_states, Gamma_V, done = self.compute_n_step_states(states, done, episode_mask, n_steps_mask_b)
        
        new_states = torch.tensor(new_states).float().to(self.device).reshape((-1,)+states.shape[2:])
        done = torch.LongTensor(done.astype(int)).to(self.device).reshape(-1)
        n_step_rewards = n_step_rewards.reshape(-1)
        Gamma_V = torch.tensor(Gamma_V).float().to(self.device).reshape(-1)
        
        with torch.no_grad():
            V_pred = self.AC.V_critic(new_states).squeeze()
        V_trg = (1-done)*Gamma_V*V_pred + n_step_rewards
        V_trg = V_trg.squeeze()
        return V_trg
    
    def compute_critic_loss(self, old_states, V_trg):
        V = self.AC.V_critic(old_states).squeeze()
        loss = F.mse_loss(V, V_trg.detach())
        return loss
    
    def compute_actor_loss(self, log_probs, old_states, V_trg):
        with torch.no_grad():
            V_pred = self.AC.V_critic(old_states).squeeze()
        A = V_trg - V_pred
        policy_gradient = - log_probs*A
        loss = torch.mean(policy_gradient)
        return loss
                
    def compute_n_step_rewards(self, rewards, done, entropies, n_steps=None):
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
        
        r = torch.tensor(rewards).float().to(self.device)
        augmented_r = r + self.H*entropies
        rewards_repeated = augmented_r.view(B,1,T).repeat(1,T,1)
        
        # Exponential discount factor
        Gamma = torch.tensor([self.gamma**i for i in range(T)]).reshape(1,-1).float().to(self.device)
        t_episode_mask = torch.tensor(episode_mask).float().to(self.device)
        t_n_steps_mask_b = torch.tensor(n_steps_mask_b).float().to(self.device)
        n_steps_r = torch.sum(Gamma*rewards_repeated*t_episode_mask*t_n_steps_mask_b, axis=2)/Gamma
        return n_steps_r, episode_mask, n_steps_mask_b

########################################################################################################################
    
class SpatialA2C_MaxEnt_v2(SpatialA2C_v2):
    """
    Maximum Entropy Reinforcement Learning objective following Levine 
    Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review.
    Highlights:
        1. Augmented reward : r_t - H*log prob(a_t|s_t)
        2. Augmented reward detached from computational graph
    """
    def __init__(self, action_space, env, spatial_model, nonspatial_model, 
                 spatial_dict,  nonspatial_dict, n_features, n_channels,
                 gamma=0.99, action_dict=None, H=1e-3, n_steps=20, device='cpu'):
        super().__init__(action_space, env, spatial_model, nonspatial_model, 
                 spatial_dict,  nonspatial_dict, n_features, n_channels,
                 gamma, action_dict, H, n_steps, device)
      
    def step(self, state, action_mask):
        state = torch.from_numpy(state).float().to(self.device)
        action_mask = torch.tensor(action_mask).to(self.device)
        
        log_probs, spatial_features, nonspatial_features = self.AC.pi(state, action_mask)
        entropy = self.compute_entropy(log_probs)
        probs = torch.exp(log_probs)
        a = Categorical(probs).sample()
        a = a.detach().cpu().numpy()
        main_log_prob = log_probs[range(len(a)), a]
        
        args, args_log_prob, args_entropy = self.get_arguments(spatial_features, nonspatial_features, a)
        log_prob = main_log_prob + args_log_prob
        entropy = entropy + args_entropy # we are not going to use them anyways

        action_id = np.array([self.AC.action_dict[act] for act in a])
        action = [actions.FunctionCall(action_id[i], args[i]) for i in range(len(action_id))]
        assert log_prob.shape == entropy.shape, ("Expected log_prob with same shape of entropy", log_prob.shape, entropy.shape)
        return action, log_prob, main_log_prob
    
    def compute_entropy(self, log_probs):
        """
        DELEAT ME: Not really used in this implementation.
        Computes POSITIVE entropy of a batch (b, n_actions) of log probabilities.
        Returns the entropy of each sample in the batch (b,)
        """
        probs = torch.exp(log_probs) 
        distr = Categorical(probs=probs)
        entropy = distr.entropy()
        return entropy
    
    def compute_ac_loss(self, rewards, log_probs, main_log_prob, states, done, bootstrap, trg_states): 
        assert states.shape == trg_states.shape, \
            ("Expected states and trg_states with same shape ", states.shape, trg_states.shape)
        # merge batch and episode dimensions
        old_states = torch.tensor(states).float().to(self.device).reshape((-1,)+states.shape[2:])
        main_log_prob = torch.stack(main_log_prob).to(self.device).transpose(1,0)
        average_n_steps = False # TRY ME
        if average_n_steps:
            # Use as V target the mean of 1-step to n-step V targets
            V_trg = []
            for n in range(1, self.n_steps + 1):
                n_step_V_trg = self.compute_n_step_V_trg(n, rewards, done, bootstrap, trg_states, main_log_prob)
                V_trg.append(n_step_V_trg)
            V_trg = torch.mean(torch.stack(V_trg, axis=0), axis=0)
        else:
            V_trg = self.compute_n_step_V_trg(self.n_steps, rewards, done, bootstrap, trg_states, main_log_prob)
            
        ### Wrap variables into tensors - merge batch and episode dimensions ###    
        log_probs = torch.stack(log_probs).to(self.device).transpose(1,0).reshape(-1)
        
        ### Compute critic and actor losses ###
        critic_loss = self.compute_critic_loss(old_states, V_trg)
        actor_loss = self.compute_actor_loss(log_probs, old_states, V_trg)

        return critic_loss, actor_loss
    
    def compute_n_step_V_trg(self, n_steps, rewards, done, bootstrap, states, log_probs):
        """
        Compute m-steps value target, with m = min(n, steps-to-episode-end).
        Formula (for precisely n-steps):
            V^{(n)}(t) = \sum_{k=0}^{n-1} gamma^k r_{t+k} + gamma^n * V(s_t)
        """
        n_step_rewards, episode_mask, n_steps_mask_b = self.compute_n_step_rewards(rewards, done, log_probs, n_steps)
        done[bootstrap] = False 
        new_states, Gamma_V, done = self.compute_n_step_states(states, done, episode_mask, n_steps_mask_b)
        
        new_states = torch.tensor(new_states).float().to(self.device).reshape((-1,)+states.shape[2:])
        done = torch.LongTensor(done.astype(int)).to(self.device).reshape(-1)
        n_step_rewards = n_step_rewards.reshape(-1)
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
    
    def compute_actor_loss(self, log_probs, old_states, V_trg):
        with torch.no_grad():
            V_pred = self.AC.V_critic(old_states).squeeze()
        A = V_trg - V_pred
        policy_gradient = - log_probs*A
        loss = torch.mean(policy_gradient)
        return loss
                
    def compute_n_step_rewards(self, rewards, done, log_probs, n_steps=None):
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
        
        r = torch.tensor(rewards).float().to(self.device)
        augmented_r = r - self.H*log_probs
        rewards_repeated = augmented_r.view(B,1,T).repeat(1,T,1)
        
        # Exponential discount factor
        Gamma = torch.tensor([self.gamma**i for i in range(T)]).reshape(1,-1).float().to(self.device)
        t_episode_mask = torch.tensor(episode_mask).float().to(self.device)
        t_n_steps_mask_b = torch.tensor(n_steps_mask_b).float().to(self.device)
        n_steps_r = torch.sum(Gamma*rewards_repeated*t_episode_mask*t_n_steps_mask_b, axis=2)/Gamma
        return n_steps_r, episode_mask, n_steps_mask_b

########################################################################################################################

class SpatialA2C_v1(SpatialA2C):
    """
    Differences from SpatialA2C:
    Conditions parameter sampling on the main action that has been sampled by 
    embedding the action and concatenating it to the spatial_features along the 
    channel dimension (and broadcasting it along the x and y axes) and also
    concatenating the embedded action to the nonspatial_features (this time
    is a simple concatenation).  
    """
    def __init__(self, action_space, env, spatial_model, nonspatial_model, 
                 spatial_dict,  nonspatial_dict, n_features, n_channels, embed_dim,
                 gamma=0.99, action_dict=None, H=1e-3, n_steps=20, device='cpu'):
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
        entropy = self.compute_entropy(log_probs)
        probs = torch.exp(log_probs)
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
                arg_sampled, log_prob, log_probs = self.AC.sample_param(nonspatial_features, arg_name)
            elif self.AC.arguments_type[arg_name] == 'spatial':
                arg_sampled, log_prob, log_probs = self.AC.sample_param(spatial_features, arg_name)
            else:
                raise Exception("argument type for "+arg_name+" not understood")  
            entropy = self.compute_entropy(log_probs)
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
    
########################################################################################################################

class SpatialA2C_v3(SpatialA2C):
    """
    Differences from SpatialA2C_v1:
    Same conditioning for the nonspatial_features; Instead uses the embedded actions
    as weights for the 1x1 convolutional layers used in the networks that extract from
    the spatial_features the logits. (this implies that the embedding dimension is equal
    to the number of channels of the spatial_features)
    """
    def __init__(self, action_space, env, spatial_model, nonspatial_model, 
                 spatial_dict,  nonspatial_dict, n_features, n_channels,
                 gamma=0.99, action_dict=None, H=1e-3, n_steps=20, device='cpu'):
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
        entropy = self.compute_entropy(log_probs)
        probs = torch.exp(log_probs)
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
                arg_sampled, log_prob, log_probs = self.AC.sample_param(arg_name, nonspatial_features)
            elif self.AC.arguments_type[arg_name] == 'spatial':
                arg_sampled, log_prob, log_probs = self.AC.sample_param(arg_name, spatial_features, embedded_a)
            else:
                raise Exception("argument type for "+arg_name+" not understood")  
            entropy = self.compute_entropy(log_probs)
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

########################################################################################################################
