"""
Used in train_v5, together with FullSpaceA2C_v2

Uses full state space, but only limited number of actions. 
Difference with A2C_inspection.py :
- supports state dictionary with player info
- different step in the agent
"""

import torch
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F 
from pysc2.lib import actions # or as sc_actions, pay attention

class InspectionDict():
    def __init__(self, step_idx, PID, agent):
        self.step_idx = step_idx
        self.PID = PID
        
        # Copy some useful internal variables - check which are needed (also in inspection plots)
        self.screen_res = agent.AC.screen_res
        self.all_actions = agent.AC.all_actions
        self.all_arguments = agent.AC.all_arguments
        self.action_table = agent.AC.action_table
        self.spatial_arg_names = agent.AC.spatial_arg_names
        self.n_spatial_args = agent.AC.n_spatial_args
        self.categorical_arg_names = agent.AC.categorical_arg_names
        self.n_categorical_args = agent.AC.n_categorical_args
        self.categorical_sizes = agent.AC.categorical_sizes
        self.act_to_arg_names = agent.AC.act_to_arg_names 
        self.dict = dict(
                        state_traj = [],
                        rewards = [],
                        action_distr = [],
                        action_sel = [],
                        top_5_actions = [],
                        top_5_action_distr = [],
                        args = [],
                        values = None,
                        trg_values = None,
                        critic_losses = None,
                        advantages = None,
                        actor_losses = None)

    def store_step(self, step_dict):
        # store every other trajectory variable except state_traj
        for k in step_dict:
            self.dict[k].append(step_dict[k])
        return
    
    def store_update(self, update_dict):
        for k in update_dict:
            self.dict[k] = update_dict[k]
        return
    
    def save_dict(self, path='../Results/MoveToBeacon/Inspection/'):
        np.save(path+self.PID+"_"+str(self.step_idx), self.dict)
        return

"""
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

    action_id = np.array([self.AC.action_table[act] for act in a]) # this can be done without list comprehension
    action = [actions.FunctionCall(action_id[i], args[i]) for i in range(len(action_id))]

    return action, log_prob, torch.mean(entropy)
    """

def inspection_step(agent, inspector, state, action_mask):
    spatial_state = state['spatial']
    player_state = state['player']
    spatial_state = torch.from_numpy(spatial_state).float().to(agent.device)
    player_state = torch.from_numpy(player_state).float().to(agent.device)
    action_mask = torch.tensor(action_mask).to(agent.device)

    log_probs, spatial_features, nonspatial_features = agent.AC.pi(spatial_state, player_state, action_mask)
    entropy = agent.compute_entropy(log_probs)
    probs = torch.exp(log_probs)
    a = Categorical(probs).sample()
    a = a.detach().cpu().numpy()
    log_prob = log_probs[range(len(a)), a]

    ### Inspection ###
    step_dict = {}
    p = probs.detach().cpu().numpy() 
    step_dict['action_distr'] = p
    step_dict['action_sel'] = a
    
    # Choose top 5 actions from the probabilities - check about the batch dim
    top_5 = np.argsort(p)[:,-5:]
    top_5_actions = np.array(top_5[:,::-1])[0] # some issues in accessing p if I don't call np.array()
    step_dict['top_5_actions'] = top_5_actions
    
    # Save SPATIAL distributions only of the top 5 actions + THEIR NAMES
    with torch.no_grad():
        _, _, log_probs = agent.AC.spatial_params_net(spatial_features)
        log_probs = log_probs.detach().cpu().numpy()[0] # batch dim 1 during inspection
        step_dict['top_5_action_distr'] = {}
        for act in top_5_actions:
            step_dict['top_5_action_distr'][act] = {}
            arg_mask = agent.AC.spatial_arg_mask[act,:].astype(bool)
            arg_names = np.array(agent.AC.spatial_arg_names)[arg_mask]
            distr = log_probs[arg_mask].reshape((-1,)+agent.AC.screen_res)
            for i, name in enumerate(arg_names):
                step_dict['top_5_action_distr'][act][name+'_distr'] = distr[i]
                
    ### End inspection ###
   
    args, args_log_prob = agent.AC.sample_params(nonspatial_features, spatial_features, a)
    step_dict['args'] = args
    
    log_prob = log_prob + args_log_prob

    action_id = np.array([agent.AC.action_table[act] for act in a])
    action = [actions.FunctionCall(action_id[i], args[i]) for i in range(len(action_id))]

    inspector.store_step(step_dict)
    return action, log_prob, torch.mean(entropy)
   
# from here on should be identical, maybe import directly those functions from v2

def inspection_update(agent, rewards, log_probs, entropies, states, done, bootstrap, trg_states): 
    # from list of dictionaries of arrays to 2 separate arrays
    spatial_states_lst = [s['spatial'] for s in states] #[(batch, other dims) x traj_len times]
    player_states_lst = [s['player'] for s in states] 
    spatial_states = torch.tensor(spatial_states_lst).float().to(agent.device).transpose(1,0)
    player_states = torch.tensor(player_states_lst).float().to(agent.device).transpose(1,0)

    spatial_states_lst_trg = [s['spatial'] for s in trg_states]
    player_states_lst_trg = [s['player'] for s in trg_states]
    spatial_states_trg = torch.tensor(spatial_states_lst_trg).float().to(agent.device).transpose(1,0)
    player_states_trg = torch.tensor(player_states_lst_trg).float().to(agent.device).transpose(1,0)

    # merge batch and episode dimensions
    old_spatial_states = spatial_states.reshape((-1,)+spatial_states.shape[2:])
    old_player_states = player_states.reshape((-1,)+player_states.shape[2:])
    
    V_trg = agent.compute_n_step_V_trg(agent.n_steps, rewards, done, bootstrap, 
                                                         spatial_states_trg, player_states_trg)
    ### Wrap variables into tensors - merge batch and episode dimensions ###    
    log_probs = torch.stack(log_probs).to(agent.device).transpose(1,0).reshape(-1)
    entropies = torch.stack(entropies, axis=0).to(agent.device).reshape(-1)

    values, trg_values, critic_losses = inspect_critic_loss(agent, old_spatial_states, old_player_states, V_trg)

    advantages, actor_losses = inspect_actor_loss(agent, log_probs, entropies, old_spatial_states, old_player_states, V_trg)

    update_dict = dict(values=values, 
                       trg_values=trg_values, 
                       critic_losses=critic_losses, 
                       advantages=advantages, 
                       actor_losses=actor_losses )
    return update_dict

def inspect_critic_loss(agent, old_spatial_states, old_player_states, V_trg):
    with torch.no_grad():
        V = agent.AC.V_critic(old_spatial_states, old_player_states).squeeze()
        V = V.cpu().numpy() 
        V_trg = V_trg.cpu().numpy()
        critic_losses = (V-V_trg)**2
    return V, V_trg, critic_losses

def inspect_actor_loss(agent, log_probs, entropies, old_spatial_states, old_player_states, V_trg):
    with torch.no_grad():
        V_pred = agent.AC.V_critic(old_spatial_states, old_player_states).squeeze()
        A = V_trg - V_pred
        policy_gradient = - log_probs*A
    A = A.cpu().numpy()
    pg = policy_gradient.cpu().numpy()
    return A, pg
