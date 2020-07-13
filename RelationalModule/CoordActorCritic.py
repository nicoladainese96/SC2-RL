### Agent 1 ###

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical

from pysc2.lib import actions
from pysc2.lib import features

from RelationalModule.MLP_AC_networks import Actor, Critic #custom module

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

class MoveToBeaconA2C():
    """
    Advantage Actor-Critic RL agent for BoxWorld environment described in the paper
    Relational Deep Reinforcement Learning.
    
    Notes
    -----
    * GPU implementation is still work in progress.
    * Always uses 2 separate networks for the critic,one that learns from new experience 
      (student/critic) and the other one (critic_target/teacher)that is more conservative 
      and whose weights are updated through an exponential moving average of the weights 
      of the critic, i.e.
          target.params = (1-tau)*target.params + tau* critic.params
    * In the case of Monte Carlo estimation the critic_target is never used
    * Possible to use twin networks for the critic and the critic target for improved 
      stability. Critic target is used for updates of both the actor and the critic and
      its output is the minimum between the predictions of its two internal networks.
      
    """ 
    
    def __init__(self, action_space, observation_space, lr, gamma, TD=True, twin=False, tau = 1., 
                 H=1e-2, n_steps = 1, device='cpu', **control_net_args):
        """
        Parameters
        ----------
        lr: float in [0,1]
            Learning rate
        gamma: float in [0,1]
            Discount factor
        TD: bool (default=True)
            If True, uses Temporal Difference for the critic's estimates
            Otherwise uses Monte Carlo estimation
        twin: bool (default=False)
            Enables twin networks both for critic and critic_target
        tau: float in [0,1] (default = 1.)
            Regulates how fast the critic_target gets updates, i.e. what percentage of the weights
            inherits from the critic. If tau=1., critic and critic_target are identical 
            at every step, if tau=0. critic_target is unchangable. 
            As a default this feature is disabled setting tau = 1, but if one wants to use it a good
            empirical value is 0.005.
        H: float (default 1e-2)
            Entropy multiplicative factor in actor's loss
        n_steps: int (default=1)
            Number of steps considered in TD update
        device: str in {'cpu','cuda'}
            Implemented, but GPU slower than CPU because it's difficult to optimize a RL agent without
            a replay buffer, that can be used only in off-policy algorithms.
        """
        
        self.gamma = gamma
        self.lr = lr
        
        self.TD = TD
        self.twin = twin 
        self.tau = tau
        self.n_steps = n_steps
        self.H = H
        
        if debug: print("control_net_args: ", control_net_args)
            
        self.actor = Actor(action_space, observation_space, **control_net_args)
        self.critic = Critic(observation_space, twin=twin, **control_net_args)
        
        if self.TD:
            self.critic_trg = Critic(observation_space, target=True, twin=twin, **control_net_args)

            # Init critic target identical to critic
            for trg_params, params in zip(self.critic_trg.parameters(), self.critic.parameters()):
                trg_params.data.copy_(params.data)
            
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        self.device = device 
        self.actor.to(self.device) 
        self.critic.to(self.device)
        if self.TD:
            self.critic_trg.to(self.device)
        
        if debug:
            print("="*10 +" A2C HyperParameters "+"="*10)
            print("Discount factor: ", self.gamma)
            print("Learning rate: ", self.lr)
            print("Temporal Difference learning: ", self.TD)
            print("Twin networks: ", self.twin)
            print("Update critic target factor: ", self.tau)
            if self.TD:
                print("n_steps for TD: ", self.n_steps)
            print("Device used: ", self.device)
            print("\n\n"+"="*10 +" A2C Architecture "+"="*10)
            print("Actor architecture: \n", self.actor)
            print("Critic architecture: \n",self.critic)
            print("Critic target architecture: ")
            if self.TD:
                print(self.critic_trg)
            else:
                print("Not used")
        
    def step(self, obs, return_log=False):
        
        state = self.get_coord_state(obs)
        if debug: print("state: ", state)
        state = torch.tensor(state).float().to(self.device)
        available_actions = obs[0].observation.available_actions
        if debug: print("available actions: ", available_actions)
        
        log_probs = self.actor(state, available_actions)
        if debug: print("log_probs: ", log_probs)
        probs = torch.exp(log_probs)
        if debug: print("probs: ", probs)
        distribution = Categorical(probs)
        a = distribution.sample().item()
        
        action_id = self.actor.action_dict[a]
        args = self.get_scripted_arguments(action_id, obs)
        action = actions.FunctionCall(action_id, args)
        
        if return_log:
            return action, log_probs.view(-1)[a], probs
        else:
            return action
    
    @staticmethod
    def get_coord_state(obs):
        
        player_relative = obs[0].observation['feature_screen'][_PLAYER_RELATIVE]
        if debug: print("player_relative: \n", player_relative)
            
        beacon_ys, beacon_xs = (player_relative == _PLAYER_NEUTRAL).nonzero()
        player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
        
        if beacon_ys.any():
            beacon_pos = [beacon_xs.mean(), beacon_ys.mean()]
        else:
            beacon_pos = [-1., -1.] # not present
        
        if player_y.any():
            player_pos = [player_x.mean(), player_y.mean()]
        else:
            player_pos = beacon_pos # if the two are superimposed, only beacon cells are showed
            
        beacon_exists = float(beacon_ys.any())

        selected = obs[0].observation['feature_screen'][_SELECTED]
        if debug: print("selected layer: \n", selected)
        is_selected = np.any((selected==1).nonzero()[0]).astype(float) 

        state = np.concatenate([player_pos, beacon_pos, [beacon_exists, is_selected]])

        return state

    @staticmethod
    def get_scripted_arguments(action_id, obs):
    
        if action_id == _SELECT_ARMY:
            args = [_SELECT_ALL]
            
        elif action_id == _MOVE_SCREEN:
            player_relative = obs[0].observation['feature_screen'][_PLAYER_RELATIVE]
            
            player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
            beacon_ys, beacon_xs = (player_relative == _PLAYER_NEUTRAL).nonzero()
            
            if player_y.any():
                player_pos = [int(player_x.mean()), int(player_y.mean())]
            else:
                player_pos = [int(beacon_xs.mean()), int(beacon_ys.mean())]
 
            if beacon_ys.any():
                coord = [int(beacon_xs.mean()), int(beacon_ys.mean())]
            else:
                coord = player_pos
                
            args = [_NOT_QUEUED, coord]
            
        else:
            args = [] # _NO_OP case
        
        return args

    
    def update(self, *args):
        if self.TD:
            return self.update_TD(*args)
        else:
            return self.update_MC(*args)
    
    def update_TD(self, rewards, log_probs, distributions, states, done, bootstrap=None):   
        
        ### Compute n-steps rewards, states, discount factors and done mask ###
        
        n_step_rewards = self.compute_n_step_rewards(rewards)
        if debug:
            print("n_step_rewards.shape: ", n_step_rewards.shape)
            print("rewards.shape: ", rewards.shape)
            print("n_step_rewards: ", n_step_rewards)
            print("rewards: ", rewards)
            print("bootstrap: ", bootstrap)
                
        if bootstrap is not None:
            done[bootstrap] = False 
        if debug:
            print("done.shape: (before n_steps)", done.shape)
            print("done: (before n_steps)", done)
        
        old_states = torch.tensor(states[:-1]).float().to(self.device)

        new_states, Gamma_V, done = self.compute_n_step_states(states, done)
        new_states = torch.tensor(new_states).float().to(self.device)

        if debug:
            print("done.shape: (after n_steps)", done.shape)
            print("Gamma_V.shape: ", Gamma_V.shape)
            print("done: (after n_steps)", done)
            print("Gamma_V: ", Gamma_V)
            print("old_states.shape: ", old_states.shape)
            print("new_states.shape: ", new_states.shape)
            
        ### Wrap variables into tensors ###
        
        done = torch.LongTensor(done.astype(int)).to(self.device)
        if debug: print("log_probs: ", log_probs)
            
        log_probs = torch.stack(log_probs).to(self.device)
        if debug: print("log_probs: ", log_probs)
            
        distributions = torch.stack(distributions, axis=0).to(self.device)
        mask = (distributions == 0).nonzero()
        distributions[mask[:,0], mask[:,1]] = 1e-5
        if debug: print("distributions: ", distributions)
            
        n_step_rewards = torch.tensor(n_step_rewards).float().to(self.device)
        Gamma_V = torch.tensor(Gamma_V).float().to(self.device)
        
        ### Update critic and then actor ###
        critic_loss = self.update_critic_TD(n_step_rewards, new_states, old_states, done, Gamma_V)
        actor_loss, entropy = self.update_actor_TD(n_step_rewards, log_probs, distributions, new_states, old_states, done, Gamma_V)
        
        return critic_loss, actor_loss, entropy
    
    def update_critic_TD(self, n_step_rewards, new_states, old_states, done, Gamma_V):
        
        # Compute loss 
        if debug: print("Updating critic...")
        with torch.no_grad():
            V_trg = self.critic_trg(new_states).squeeze()
            if debug:
                print("V_trg.shape (after critic): ", V_trg.shape)
            V_trg = (1-done)*Gamma_V*V_trg + n_step_rewards
            if debug:
                print("V_trg.shape (after sum): ", V_trg.shape)
            V_trg = V_trg.squeeze()
            if debug:
                print("V_trg.shape (after squeeze): ", V_trg.shape)
                print("V_trg.shape (after squeeze): ", V_trg)
            
        if self.twin:
            V1, V2 = self.critic(old_states)
            if debug:
                print("V1.shape: ", V1.squeeze().shape)
                print("V1: ", V1)
            loss1 = 0.5*F.mse_loss(V1.squeeze(), V_trg)
            loss2 = 0.5*F.mse_loss(V2.squeeze(), V_trg)
            loss = loss1 + loss2
        else:
            V = self.critic(old_states).squeeze()
            if debug: 
                print("V.shape: ",  V.shape)
                print("V: ",  V)
            loss = F.mse_loss(V, V_trg)
        
        # Backpropagate and update
        
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        
        # Update critic_target: (1-tau)*old + tau*new
        
        for trg_params, params in zip(self.critic_trg.parameters(), self.critic.parameters()):
                trg_params.data.copy_((1.-self.tau)*trg_params.data + self.tau*params.data)
        
        return loss.item()
    
    def update_actor_TD(self, n_step_rewards, log_probs, distributions, new_states, old_states, done, Gamma_V):
        
        # Compute gradient 
        if debug: print("Updating actor...")
        with torch.no_grad():
            if self.twin:
                V1, V2 = self.critic(old_states)
                V_pred = torch.min(V1.squeeze(), V2.squeeze())
                V1_new, V2_new = self.critic(new_states)
                V_new = torch.min(V1_new.squeeze(), V2_new.squeeze())
                V_trg = (1-done)*Gamma_V*V_new + n_step_rewards
            else:
                V_pred = self.critic(old_states).squeeze()
                V_trg = (1-done)*Gamma_V*self.critic(new_states).squeeze()  + n_step_rewards
        
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
        entropy = self.H*torch.mean(distributions*torch.log(distributions))
        if debug: print("Negative entropy: ", entropy)
        
        loss = policy_grad + entropy
        if debug: print("Actor loss: ", loss)
        
        # Backpropagate and update
    
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        
        return policy_grad.item(), entropy.item()
    
    def compute_n_step_rewards(self, rewards):
        """
        Computes n-steps discounted reward padding with zeros the last elements of the trajectory.
        This means that the rewards considered are AT MOST n, but can be less for the last n-1 elements.
        """
        T = len(rewards)
        
        # concatenate n_steps zeros to the rewards -> they do not change the cumsum
        r = np.concatenate((rewards,[0 for _ in range(self.n_steps)])) 
        
        Gamma = np.array([self.gamma**i for i in range(r.shape[0])])
        
        # reverse everything to use cumsum in right order, then reverse again
        Gt = np.cumsum(r[::-1]*Gamma[::-1])[::-1]
        
        G_nstep = Gt[:T] - Gt[self.n_steps:] # compute n-steps discounted return
        
        Gamma = Gamma[:T]
        
        assert len(G_nstep) == T, "Something went wrong computing n-steps reward"
        
        n_steps_r = G_nstep / Gamma
        
        return n_steps_r
    
    def compute_n_step_states(self, states, done):
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
        
        # Compute indexes for (at most) n-step away states 
        
        n_step_idx = np.arange(len(states)-1) + self.n_steps
        diff = n_step_idx - len(states) + 1
        mask = (diff > 0)
        n_step_idx[mask] = len(states) - 1
        
        # Compute new states
        
        new_states = states[n_step_idx]
        
        # Compute discount factors
        
        pw = np.array([self.n_steps for _ in range(len(new_states))])
        pw[mask] = self.n_steps - diff[mask]
        Gamma_V = self.gamma**pw
        
        # Adjust done mask
        
        mask = (diff >= 0)
        done[mask] = done[-1]
        
        return new_states, Gamma_V, done
    
    def update_MC(self, rewards, log_probs, states, done, bootstrap=None):   
        print("states: ", states.shape)
        ### Compute MC discounted returns ###
        
        if bootstrap is not None:
            
            if bootstrap[-1] == True:
            
                last_state = torch.tensor(states[-1].astype(int)).to(self.device).unsqueeze(0)
                print("last_state: ", last_state.shape)
                
                if self.twin:
                    V1, V2 = self.critic(last_state)
                    V_bootstrap = torch.min(V1, V2).cpu().detach().numpy().reshape(1,)
                else:
                    V_bootstrap = self.critic(last_state).cpu().detach().numpy().reshape(1,)
 
                rewards = np.concatenate((rewards, V_bootstrap))
                
        Gamma = np.array([self.gamma**i for i in range(rewards.shape[0])])
        # reverse everything to use cumsum in right order, then reverse again
        Gt = np.cumsum(rewards[::-1]*Gamma[::-1])[::-1]
        # Rescale so that present reward is never discounted
        discounted_rewards =  Gt/Gamma
        
        if bootstrap is not None:
            if bootstrap[-1] == True:
                discounted_rewards = discounted_rewards[:-1] # drop last

        ### Wrap variables into tensors ###
        
        dr = torch.tensor(discounted_rewards).float().to(self.device) 
        
        old_states = torch.tensor(states[:-1].astype(int)).to(self.device)
        new_states = torch.tensor(states[1:].astype(int)).to(self.device)
        done = torch.LongTensor(done.astype(int)).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)
        
        ### Update critic and then actor ###
        
        critic_loss = self.update_critic_MC(dr, old_states)
        actor_loss = self.update_actor_MC(dr, log_probs, old_states)
        
        return critic_loss, actor_loss
    
    def update_critic_MC(self, dr, old_states):

        # Compute loss
        
        if self.twin:
            V1, V2 = self.critic(old_states)
            V_pred = torch.min(V1.squeeze(), V2.squeeze())
        else:
            V_pred = self.critic(old_states).squeeze()
            
        loss = F.mse_loss(V_pred, dr)
        
        # Backpropagate and update
        
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        
        return loss.item()
    
    def update_actor_MC(self, dr, log_probs, old_states):
        
        # Compute gradient 
        
        if self.twin:
            V1, V2 = self.critic(old_states)
            V_pred = torch.min(V1.squeeze(), V2.squeeze())
        else:
            V_pred = self.critic(old_states).squeeze()
            
        A = dr - V_pred
        policy_gradient = - log_probs*A
        policy_grad = torch.sum(policy_gradient)
 
        # Backpropagate and update
    
        self.actor_optim.zero_grad()
        policy_grad.backward()
        self.actor_optim.step()
        
        return policy_grad.item()

    
