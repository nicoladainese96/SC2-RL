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
                        selectall_distr = [],
                        spatial_distr = [],
                        spatial_sel = [],
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
        screen_arg, screen_log_prob, screen_distr = agent.AC.sample_param(state, 'screen')
        p = screen_distr.detach().cpu().numpy().reshape(state.shape[-2:]) 
        step_dict['spatial_distr'] = p
        #step_dict['spatial_sel'] = screen_arg
    ### End inspection ###
    log_prob = log_probs[range(len(a)), a]
    action_id = np.array([agent.AC.action_dict[act] for act in a])
    args, args_log_prob, args_entropy = agent.get_arguments(state, nonspatial_features, a)
    
    if move_only:
        if a[0] != 2:
            step_dict['spatial_sel'] = [0,0]
        else:
            step_dict['spatial_sel'] = args[0][1]
    log_prob = log_prob + args_log_prob
    entropy = entropy + args_entropy
    action = [actions.FunctionCall(action_id[i], args[i]) for i in range(len(action_id))]

    return action, log_prob, entropy, step_dict
"""
def compute_actor_loss(agent, log_probs, rewards, done, bootstrap):
    n_rewards, _, _ = agent.compute_n_step_rewards(rewards, done)
    A = (n_rewards-n_rewards.mean())/(n_rewards.std()+1e-5)
    done[bootstrap] = False 
    
    done = torch.LongTensor(done.astype(int)).to(agent.device).reshape(-1)
    A = torch.tensor(A).float().to(agent.device).reshape(-1)
    log_probs = torch.stack(log_probs).to(agent.device).transpose(1,0).reshape(-1)
    
    policy_gradient = - log_probs*A
    loss = torch.mean(policy_gradient)
    return loss
"""
def inspection_update(agent, log_probs, rewards, done, bootstrap): 
    n_rewards, _, _ = agent.compute_n_step_rewards(rewards, done)
    A = (n_rewards-n_rewards.mean())/(n_rewards.std()+1e-5)
    done[bootstrap] = False 
    
    done = torch.LongTensor(done.astype(int)).to(agent.device).reshape(-1)
    A = torch.tensor(A).float().to(agent.device).reshape(-1)
    log_probs = torch.stack(log_probs).to(agent.device).transpose(1,0).reshape(-1)
    
    policy_gradient = - log_probs*A

    update_dict = dict(advantages=A.cpu().detach().numpy(), 
                       actor_losses=policy_gradient.cpu().detach().numpy())
    return update_dict
