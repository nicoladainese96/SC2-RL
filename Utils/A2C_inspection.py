import torch
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F 
from pysc2.lib import actions

move_only = True

class InspectionDict():
    def __init__(self, step_idx, PID):
        self.step_idx = step_idx
        self.PID = PID
        self.dict = dict(
                        state_traj = [],
                        rewards = [],
                        action_distr = [],
                        action_sel = [],
                        queue_distr = [],
                        queue_sel = [],
                        selectall_distr = [],
                        selectall_sel = [],
                        spatial_distr = [],
                        spatial_sel = [],
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
    
    def save_dict(self, path='Results/MoveToBeacon/Inspection/'):
        np.save(path+self.PID+"_"+str(self.step_idx), self.dict)
        return
    
def inspection_step(agent, state, action_mask):
    state = torch.from_numpy(state).float().to(agent.device)
    action_mask = torch.tensor(action_mask).to(agent.device)
    log_probs, spatial_features, nonspatial_features = agent.AC.pi(state, action_mask)
    probs = torch.exp(log_probs)
    entropy = agent.compute_entropy(probs)
    a = Categorical(probs).sample()
    a = a.detach().cpu().numpy()
    ### Inspection ###
    step_dict = {}
    p = probs.detach().cpu().numpy() 
    step_dict['action_distr'] = p
    step_dict['action_sel'] = a
    # All this sampling is completely wrong
    with torch.no_grad():
        # select_add
        sel_arg, sel_log_prob, sel_distr = agent.AC.sample_param(nonspatial_features, 'select_add')
        p = sel_distr.detach().cpu().numpy() 
        step_dict['selectall_distr'] = p
        #step_dict['selectall_sel'] = sel_arg
        # queued
        q_arg, q_log_prob, q_distr = agent.AC.sample_param(nonspatial_features, 'queued')
        p = q_distr.detach().cpu().numpy() 
        step_dict['queue_distr'] = p
        #step_dict['queue_sel'] = q_arg
        # screen
        screen_arg, screen_log_prob, screen_distr = agent.AC.sample_param(spatial_features, 'screen')
        p = screen_distr.detach().cpu().numpy().reshape(state.shape[-2:]) 
        step_dict['spatial_distr'] = p
        #step_dict['spatial_sel'] = screen_arg
    ### End inspection ###
    log_prob = log_probs[range(len(a)), a]
    action_id = np.array([agent.AC.action_dict[act] for act in a])
    args, args_log_prob, args_entropy = agent.get_arguments(spatial_features, nonspatial_features, a)
    
    if move_only:
        if a[0] != 2:
            step_dict['spatial_sel'] = [0,0]
        else:
            step_dict['spatial_sel'] = args[0][1]
    log_prob = log_prob + args_log_prob
    entropy = entropy + args_entropy
    action = [actions.FunctionCall(action_id[i], args[i]) for i in range(len(action_id))]

    return action, log_prob, entropy, step_dict

def inspection_update(agent, rewards, log_probs, entropies, states, done, bootstrap, trg_states): 
    old_states = torch.tensor(states).float().to(agent.device).reshape((-1,)+states.shape[2:])
    V_trg = agent.compute_n_step_V_trg(agent.n_steps, rewards, done, bootstrap, states)
    log_probs = torch.stack(log_probs).to(agent.device).transpose(1,0).reshape(-1)
    entropies = torch.stack(entropies, axis=0).to(agent.device).reshape(-1)

    values, trg_values, critic_losses = inspect_critic_loss(agent, old_states, V_trg)

    advantages, actor_losses = inspect_actor_loss(agent, log_probs, entropies, old_states, V_trg)

    update_dict = dict(values=values, 
                       trg_values=trg_values, 
                       critic_losses=critic_losses, 
                       advantages=advantages, 
                       actor_losses=actor_losses )
    return update_dict

def inspect_critic_loss(agent, old_states, V_trg):
    with torch.no_grad():
        V = agent.AC.V_critic(old_states).squeeze()
        V = V.cpu().numpy() 
        V_trg = V_trg.cpu().numpy()
        critic_losses = (V-V_trg)**2
    return V, V_trg, critic_losses

def inspect_actor_loss(agent, log_probs, entropies, old_states, V_trg):
    with torch.no_grad():
        V_pred = agent.AC.V_critic(old_states).squeeze()
        A = V_trg - V_pred
        policy_gradient = - log_probs*A
    A = A.cpu().numpy()
    pg = policy_gradient.cpu().numpy()
    return A, pg
