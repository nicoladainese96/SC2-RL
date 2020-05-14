import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical

from AC_modules.ActorCriticArchitecture import *

from pysc2.lib import actions
from pysc2.lib import features

# indexes of useful layers of the screen_features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index 
_SELECTED = features.SCREEN_FEATURES.selected.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Identifiers in player_relative feature layer
_BACKGROUND = 0
_PLAYER_FRIENDLY = 1
_PLAYER_ALLIES = 2
_PLAYER_NEUTRAL = 3
_PLAYER_HOSTILE = 4

# Ids of the actions that we'll use
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id

# Meaning of some arguments required by the actions
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

debug = False

class MoveToBeaconSpatialA2C():
    """
    Advantage Actor-Critic RL agent for BoxWorld environment described in the paper
    Relational Deep Reinforcement Learning.
    
    Notes
    -----
      
    """ 
    
    def __init__(self, action_space, n_layers, linear_size, in_channels, n_channels,
                 env, gamma, H=1e-3, n_steps = 20, device='cpu', **net_args):
        """
        Parameters
        ----------
        action_space: int
            Number of (discrete) possible actions to take
        gamma: float in [0,1]
            Discount factor
        H: float (default 1e-3)
            Entropy multiplicative factor in actor's loss
        n_steps: int (default 1)
            Number of steps considered in TD update
        device: str in {'cpu','cuda'}
            Whether to use CPU or GPU
        **net_args: dict (optional)
            Dictionary of {'key':value} pairs valid for OheNet.
            
        """
        
        self.gamma = gamma
        
        self.n_actions = action_space
        self.n_steps = n_steps
        self.H = H
        
        self.AC = SpatialActorCritic(action_space, env, n_layers, linear_size, in_channels, n_channels, **net_args)
     
        self.device = device 
        self.AC.to(self.device) 

        if debug:
            print("="*10 +" A2C HyperParameters "+"="*10)
            print("Discount factor: ", self.gamma)
            print("Action space: ", self.n_actions)
            print("n_steps for TD: ", self.n_steps)
            print("Device used: ", self.device)
            print("\n\n"+"="*10 +" A2C Architecture "+"="*10)
            print("Architecture: \n", self.AC)

    def step(self, obs):
        
        state = self.get_ohe_state(obs)
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        available_actions = obs[0].observation.available_actions
        if debug: print("\navailable actions: ", available_actions)
            
        log_probs, spatial_features, nonspatial_features = self.AC.pi(state, available_actions)
        if debug: 
            print("log_probs: ", log_probs)
            print("spatial_features.shape: ", spatial_features.shape)
            print("spatial_features: ", spatial_features)
            print("spatial_features (cuda): ", spatial_features.is_cuda)
            print("nonspatial_features.shape: ", nonspatial_features.shape)
            print("nonspatial_features: ", nonspatial_features)
            print("nonspatial_features (cuda): ", nonspatial_features.is_cuda)
            
            
        probs = torch.exp(log_probs)
        if debug: print("probs: ", probs)
            
        distribution = Categorical(probs)
        a = distribution.sample().item()
        log_prob = log_probs.view(-1)[a]
        if debug: print("log_prob: ", log_prob)
        
        action_id = self.AC.action_dict[a]
        if debug: print("action_id: ", action_id)
            
        args, args_log_prob = self.get_arguments(spatial_features, nonspatial_features, action_id)
        if debug: print("args: ", args)
        
        if args_log_prob is not None:
            log_prob = log_prob + args_log_prob
            if debug: 
                print("args_log_prob: ", args_log_prob)
                print("log_prob (after sum): ", log_prob)
                
        action = actions.FunctionCall(action_id, args)
        if debug: print("action: ", action)
            
        return action, log_prob, probs

    
    @staticmethod
    def get_ohe_state(obs):
    
        player_relative = obs[0].observation['feature_screen'][_PLAYER_RELATIVE]
        selected = obs[0].observation['feature_screen'][_SELECTED].astype(float)

        friendly = (player_relative == _PLAYER_FRIENDLY).astype(float)
        neutral = (player_relative == _PLAYER_NEUTRAL).astype(float)

        state = np.zeros((3,)+player_relative.shape).astype(float)
        state[0] = friendly
        state[1] = neutral
        state[2] = selected

        return state
    
    def get_arguments(self, spatial_features, nonspatial_features, action_id):
        action = self.AC.all_actions[action_id]
        list_of_args = action.args
        
        args = []
        if len(list_of_args) != 0:
            log_probs = []
            for arg in list_of_args:
                size = self.AC.all_arguments[arg.id].sizes
                if len(size) == 1:
                    arg_sampled, log_prob, _ = self.AC.sample_param(nonspatial_features, arg.name)
                else:
                    arg_sampled, log_prob, _ = self.AC.sample_param(spatial_features, arg.name)
                args.append(arg_sampled)
                log_probs.append(log_prob)
            log_prob = torch.stack(log_probs).sum()
            
        else:
            log_prob = None
            
        return args, log_prob
        
    def compute_ac_loss(self, rewards, log_probs, distributions, states, done, bootstrap, trg_states): 
        ### Compute n-steps rewards, states, discount factors and done mask ###
        
        n_step_rewards, episode_mask, n_steps_mask_b = self.compute_n_step_rewards(rewards, done)
        if debug:
            print("n_step_rewards.shape: ", n_step_rewards.shape)
            print("rewards.shape: ", rewards.shape)
            print("n_step_rewards: ", n_step_rewards)
            print("rewards: ", rewards)
            print("bootstrap: ", bootstrap)
        # seems to work
        done[bootstrap] = False 
        
        if debug:
            print("done.shape: (before n_steps)", done.shape)
            print("done: (before n_steps)", done)
        
        # merge batch and episode dimensions
        old_states = torch.tensor(states).float().to(self.device).reshape((-1,)+states.shape[2:])

        new_states, Gamma_V, done = self.compute_n_step_states(states, done, episode_mask, n_steps_mask_b)
        # merge batch and episode dimensions
        new_states = torch.tensor(new_states).float().to(self.device).reshape((-1,)+states.shape[2:])

        if debug:
            print("done.shape: (after n_steps)", done.shape)
            print("Gamma_V.shape: ", Gamma_V.shape)
            print("done: (after n_steps)", done)
            print("Gamma_V: ", Gamma_V)
            print("old_states.shape: ", old_states.shape)
            print("new_states.shape: ", new_states.shape)
            
        ### Wrap variables into tensors - merge batch and episode dimensions ###
        
        done = torch.LongTensor(done.astype(int)).to(self.device).reshape(-1)
            
        log_probs = torch.stack(log_probs).to(self.device).transpose(1,0).reshape(-1)
        if debug: print("log_probs.shape: ", log_probs.shape)
            
        distributions = torch.stack(distributions, axis=0).to(self.device).transpose(1,0).reshape(-1, self.n_actions)
        mask = (distributions == 0).nonzero()
        distributions[mask[:,0], mask[:,1]] = 1e-5
        if debug: print("distributions.shape: ", distributions.shape)
            
        n_step_rewards = torch.tensor(n_step_rewards).float().to(self.device).reshape(-1)
        Gamma_V = torch.tensor(Gamma_V).float().to(self.device).reshape(-1)
        
        ### Update critic and then actor ###
        critic_loss = self.compute_critic_loss(n_step_rewards, new_states, old_states, done, Gamma_V)

        actor_loss, entropy = self.compute_actor_loss(n_step_rewards, log_probs, distributions, 
                                                       new_states, old_states, done, Gamma_V)

        return critic_loss, actor_loss, entropy
    
    def compute_critic_loss(self, n_step_rewards, new_states, old_states, done, Gamma_V):
        
        # Compute loss 
        if debug: print("Updating critic...")
        with torch.no_grad():
            V_trg = self.AC.V_critic(new_states).squeeze()
            if debug:
                print("V_trg.shape (after critic): ", V_trg.shape)
            V_trg = (1-done)*Gamma_V*V_trg + n_step_rewards
            if debug:
                print("V_trg.shape (after sum): ", V_trg.shape)
            V_trg = V_trg.squeeze()
            if debug:
                print("V_trg.shape (after squeeze): ", V_trg.shape)
                print("V_trg.shape (after squeeze): ", V_trg)
            
        V = self.AC.V_critic(old_states).squeeze()
        if debug: 
            print("V.shape: ",  V.shape)
            print("V: ",  V)
        loss = F.mse_loss(V, V_trg)

        return loss
    
    def compute_actor_loss(self, n_step_rewards, log_probs, distributions, new_states, old_states, done, Gamma_V):
        
        # Compute gradient 
        if debug: print("Updating actor...")
        with torch.no_grad():
            V_pred = self.AC.V_critic(old_states).squeeze()
            V_trg = (1-done)*Gamma_V*self.AC.V_critic(new_states).squeeze()  + n_step_rewards
        
        A = V_trg - V_pred
        policy_gradient = - log_probs*A
        if debug:
            print("V_trg.shape: ",V_trg.shape)
            print("V_trg: ", V_trg)
            print("V_pred.shape: ",V_pred.shape)
            print("V_pred: ", V_pred)
            print("A.shape: ", A.shape)
            print("A: ", A)
            print("policy_gradient.shape: ", policy_gradient.shape)
            print("policy_gradient: ", policy_gradient)
        policy_grad = torch.mean(policy_gradient)
        if debug: print("policy_grad: ", policy_grad)
            
        # Compute negative entropy (no - in front)
        entropy = torch.mean(distributions*torch.log(distributions))
        if debug: print("Negative entropy: ", entropy)
        
        loss = policy_grad + self.H*entropy
        if debug: print("Actor loss: ", loss)
             
        return loss, entropy
                
    def compute_n_step_rewards(self, rewards, done):
        """
        Computes n-steps discounted reward padding with zeros the last elements of the trajectory.
        This means that the rewards considered are AT MOST n, but can be less for the last n-1 elements.
        """
        B = done.shape[0]
        T = done.shape[1]
        if debug:
            print("batch size: ", B)
            print("unroll len: ", T)
        
        
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
            m = [1 if (j>=i and j<i+self.n_steps) else 0 for j in range(T)]
            n_steps_mask.append(m)
        n_steps_mask = np.array(n_steps_mask)
        n_steps_mask_b = np.repeat(n_steps_mask[np.newaxis,...] , B, axis=0)
        
        # Broadcast rewards to use multiplicative masks
        rewards_repeated = np.repeat(rewards[:,np.newaxis,:], T, axis=1)
        
        # Exponential discount factor
        Gamma = np.array([self.gamma**i for i in range(T)]).reshape(1,-1)
        if debug:
            print("Gamma.shape: ", Gamma.shape)
            print("rewards_repeated.shape: ", rewards_repeated.shape)
            print("episode_mask.shape: ", episode_mask.shape)
            print("n_steps_mask_b.shape: ", n_steps_mask_b.shape)
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