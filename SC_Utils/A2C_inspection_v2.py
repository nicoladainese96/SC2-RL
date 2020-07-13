"""
Used in train_v4, supports full action and state space.

New features:
1. Store for every timestep the 5 most probable actions that the agent can choose
2. Store for every timestep the parameter distribution of all the top 5 actions' parameters
3. Do not store every parameter distribution
4. Init passing agent instance instead of action_dict and env instance
"""

import torch
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F 
from pysc2.lib import actions

class InspectionDict():
    def __init__(self, step_idx, PID, agent):
        self.step_idx = step_idx
        self.PID = PID
        
        # Copy some useful internal variables - check which are needed
        self.screen_res = agent.AC.screen_res
        self.all_actions = agent.AC.all_actions
        self.all_arguments = agent.AC.all_arguments
        self.action_space = agent.AC.action_space
        self.arguments_names_lst = agent.AC.arguments_names_lst # not sure this one is needed
        self.arguments_type = agent.AC.arguments_type # useful in plotting
        self.act_to_arg_names = agent.AC.act_to_arg_names # useful to get top_5_action_distr
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
    #print("top_5_actions: ", top_5_actions, top_5_actions.shape)
    step_dict['top_5_actions'] = top_5_actions
    
    # Save distributions only of the top 5 actions
    step_dict['top_5_action_distr'] = {}
    with torch.no_grad():
        for act in top_5_actions:
            step_dict['top_5_action_distr'][act] = {} # first nested level
            arg_names = inspector.act_to_arg_names[act]
            for arg_name in arg_names:
                if inspector.arguments_type[arg_name] == 'spatial': # it's either 'spatial' or 'categorical'
                    insp_arg, insp_log_prob, insp_distr = agent.AC.sample_param(spatial_features, arg_name)
                    p = insp_distr.detach().cpu().numpy().reshape(spatial_state.shape[-2:]) 
                else:
                    insp_arg, insp_log_prob, insp_distr = agent.AC.sample_param(nonspatial_features, arg_name)
                    p = insp_distr.detach().cpu().numpy() 
                    
                step_dict['top_5_action_distr'][act][arg_name+'_distr'] = p # second nested level
                
    ### End inspection ###
   
    args, args_log_prob, args_entropy = agent.get_arguments(spatial_features, nonspatial_features, a)
    step_dict['args'] = args
    
    log_prob = log_prob + args_log_prob

    action = [actions.FunctionCall(a[i], args[i]) for i in range(len(a))]

    inspector.store_step(step_dict)
    return action, log_prob, torch.mean(entropy)
   
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
