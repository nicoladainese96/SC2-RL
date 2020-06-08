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
    
    def __init__(self, action_space, env, spatial_model, nonspatial_model, spatial_dict, 
                 nonspatial_dict, n_features, n_channels, gamma, H=1e-3, n_steps = 20, device='cpu'):
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
        
        self.AC = SpatialActorCritic(action_space, env, spatial_model, nonspatial_model, spatial_dict, 
                                     nonspatial_dict, n_features, n_channels)
     
        self.device = device 
        self.AC.to(self.device) 

    def step(self, state, action_mask):
        state = torch.from_numpy(state).float().to(self.device)
        action_mask = torch.tensor(action_mask).to(self.device)
            
        log_probs, spatial_features, nonspatial_features = self.AC.pi(state, action_mask)
        if debug: 
            print("log_probs: ", log_probs)
            
        probs = torch.exp(log_probs)
        entropy = self.compute_entropy(probs)
        if debug: 
            print("probs: ", probs)
            print("entropy (main actor): ", entropy)
            
        a = Categorical(probs).sample()
        a = a.detach().cpu().numpy()
        log_prob = log_probs[range(len(a)), a]
        if debug: 
            print("log_prob: ", log_prob)

        action_id = np.array([self.AC.action_dict[act] for act in a])
        if debug: 
            print("action_id: ", action_id)
            print("a: ", a)
        args, args_log_prob, args_entropy = self.get_arguments(state, nonspatial_features, a)
        if debug: 
            print("\nargs: ", args)
            print("args_log_prob.shape; ", args_log_prob.shape)
            print("args_log_prob: ", args_log_prob)
            print("log_prob.shape: ", log_prob.shape)
            print("log_prob: ", log_prob)

        log_prob = log_prob + args_log_prob
        entropy = entropy + args_entropy
        if debug: 
            print("args_log_prob: ", args_log_prob)
            print("log_prob (after sum): ", log_prob)
            print("entropy (after sum): ", entropy)
            
        action = [actions.FunctionCall(action_id[i], args[i]) for i in range(len(action_id))]
        if debug: print("action: ", action)
        
        return action, log_prob, torch.mean(entropy)

    def get_arguments(self, state, nonspatial_features, action):
        
        results = {}    
        for arg_name in self.AC.arguments_dict.keys():
            if self.AC.arguments_type[arg_name] == 'categorical':
                arg_sampled, log_prob, probs = self.AC.sample_param(nonspatial_features, arg_name)
            elif self.AC.arguments_type[arg_name] == 'spatial':
                arg_sampled, log_prob, probs = self.AC.sample_param(state, arg_name)
            else:
                raise Exception("argument type for "+arg_name+" not understood")
                
            entropy = self.compute_entropy(probs)
            if debug:
                print("\narg_name : "+arg_name)
                print("arg_sampled: ", arg_sampled)
                print("log_prob: ", log_prob)
                print("entropy: ", entropy)
            
            results[arg_name] = (arg_sampled, log_prob, entropy)
           
        args, args_log_prob, args_entropy = [], [], []
        for i, a in enumerate(action):
            arg = []
            arg_log_prob = torch.tensor([0]).float().to(self.device)
            entropies = torch.tensor([0]).float().to(self.device)
            
            arg_names = self.AC.act_to_arg_names[a]
            if debug: print("\narg_names: ", arg_names)
            values = list( map(results.get, arg_names) )
            if debug: 
                print("values: ", values)
                print("len(values): ", len(values))
                print("len(arg_names): ", len(arg_names))
            if len(values) != 0:
                if debug: 
                    print("len(values[0]): ", len(values[0]))
                    print("values[0]: ", values[0])
                for j in range(len(values)):
                    if debug:
                        print('values[%d][0,i]'%j, values[j][0][i])
                        print('values[%d][1,i]'%j, values[j][1][i])
                        print('values[%d][2,i]'%j, values[j][2][i])
                    arg.append(list(values[j][0][i]))
                    arg_log_prob = arg_log_prob + values[j][1][i] # sum log_probs
                    entropies = entropies + values[j][2][i] # sum entropies
            if debug:
                print("arg: ", arg)
                print('arg_log_prob: ', arg_log_prob) # requires gradient now?
            args.append(arg)
            args_log_prob.append(arg_log_prob) 
            args_entropy.append(entropies)
        args_log_prob = torch.stack(args_log_prob, axis=0).squeeze()
        args_entropy = torch.stack(args_entropy, axis=0).squeeze()
        return args, args_log_prob, args_entropy
 
    def compute_entropy(self, probs):
        """
        Computes negative entropy of a batch (b, n_actions) of probabilities.
        Returns the entropy of each sample in the batch (b,)
        """
        #mask = (probs == 0).nonzero()
        #probs[mask[:,0], mask[:,1]] = 1e-5
        # add a small regularization to probs
        probs = probs + torch.tensor([1e-5]).float().to(self.device)
        entropy = torch.sum(probs*torch.log(probs), axis=1)
        return entropy
    
    def compute_ac_loss(self, rewards, log_probs, entropies, states, done, bootstrap, trg_states): 
        # merge batch and episode dimensions
        old_states = torch.tensor(states).float().to(self.device).reshape((-1,)+states.shape[2:])

        average_n_steps = False
        if average_n_steps:
            ### Use as V target the mean of 1-step to n-step V targets
            V_trg = []
            for n in range(1, self.n_steps + 1):
                n_step_V_trg = self.compute_n_step_V_trg(n, rewards, done, bootstrap, states)
                V_trg.append(n_step_V_trg)
            V_trg = torch.mean(torch.stack(V_trg, axis=0), axis=0)
        else:
            V_trg = self.compute_n_step_V_trg(self.n_steps, rewards, done, bootstrap, states)
        if debug: 
            print("V_trg.shape: ", V_trg.shape)
            print("V_trg: ", V_trg)
            
        ### Wrap variables into tensors - merge batch and episode dimensions ###    
        log_probs = torch.stack(log_probs).to(self.device).transpose(1,0).reshape(-1)
        if debug: 
            print("log_probs.shape: ", log_probs.shape)
            print("log_probs: ", log_probs)
        entropies = torch.stack(entropies, axis=0).to(self.device).reshape(-1)
        if debug: print("entropies.shape: ", entropies.shape)
        
        ### Update critic and then actor ###
        critic_loss = self.compute_critic_loss(old_states, V_trg)
        actor_loss, entropy = self.compute_actor_loss(log_probs, entropies, old_states, V_trg)

        return critic_loss, actor_loss, entropy
    
    def compute_n_step_V_trg(self, n_steps, rewards, done, bootstrap, states):
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
        if debug: print("V_trg.shape; ", V_trg.shape)
        return V_trg
    
    def compute_critic_loss(self, old_states, V_trg):
        V = self.AC.V_critic(old_states).squeeze()
        loss = F.mse_loss(V, V_trg)
        return loss
    
    def compute_actor_loss(self, log_probs, entropies, old_states, V_trg):
        with torch.no_grad():
            V_pred = self.AC.V_critic(old_states).squeeze()
        A = V_trg - V_pred
        #A = (A - A.mean())/(A.std()+1e-5)
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
        
        entropy = torch.mean(entropies)
        #print("policy_grad: ", policy_grad)
        #print("Entropy: ", entropy)
        loss = policy_grad + self.H*entropy
        if debug: print("Actor loss: ", loss)
             
        return loss, entropy
                
    def compute_n_step_rewards(self, rewards, done, n_steps=None):
        """
        Computes n-steps discounted reward padding with zeros the last elements of the trajectory.
        This means that the rewards considered are AT MOST n, but can be less for the last n-1 elements.
        """
        if n_steps is None:
            n_steps = self.n_steps
            
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
            m = [1 if (j>=i and j<i+n_steps) else 0 for j in range(T)]
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