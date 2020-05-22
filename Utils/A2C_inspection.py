import torch
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F 
from pysc2.lib import actions

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
        np.save(path+PID+"_"+str(step_idx), self.dict)
        return
    
def inspection_step(agent, state, action_mask):
    state = torch.from_numpy(state).float().to(agent.device)
    action_mask = torch.tensor(action_mask).to(agent.device)
    log_probs, spatial_features, nonspatial_features = agent.AC.pi(state, action_mask)
    probs = torch.exp(log_probs)
    a = Categorical(probs).sample()
    a = a.detach().cpu().numpy()
    ### Inspection ###
    step_dict = {}
    p = probs.detach().cpu().numpy() 
    step_dict['action_distr'] = p
    step_dict['action_sel'] = a
    with torch.no_grad():
        # select_add
        sel_arg, sel_log_prob, sel_distr = agent.AC.sample_param(nonspatial_features, 'select_add')
        p = sel_distr.detach().cpu().numpy() 
        step_dict['selectall_distr'] = p
        step_dict['selectall_sel'] = sel_arg
        # queued
        q_arg, q_log_prob, q_distr = agent.AC.sample_param(nonspatial_features, 'queued')
        p = q_distr.detach().cpu().numpy() 
        step_dict['queue_distr'] = p
        step_dict['queue_sel'] = q_arg
        # screen
        screen_arg, screen_log_prob, screen_distr = agent.AC.sample_param(spatial_features, 'screen')
        p = screen_distr.detach().cpu().numpy().reshape(16,16) 
        step_dict['spatial_distr'] = p
        step_dict['spatial_sel'] = q_arg
    ### End inspection ###
    log_prob = log_probs[range(len(a)), a]
    action_id = np.array([agent.AC.action_dict[act] for act in a])
    args, args_log_prob = agent.get_arguments(spatial_features, nonspatial_features, a)
    log_prob = log_prob + args_log_prob
    action = [actions.FunctionCall(action_id[i], args[i]) for i in range(len(action_id))]

    return action, log_prob, probs, step_dict

def inspection_update(agent, rewards, log_probs, distributions, states, done, bootstrap, trg_states): 
    ### Compute n-steps rewards, states, discount factors and done mask ###
    n_step_rewards, episode_mask, n_steps_mask_b = agent.compute_n_step_rewards(rewards, done)
    done[bootstrap] = False 
    
    # merge batch and episode dimensions
    old_states = torch.tensor(states).float().to(agent.device).reshape((-1,)+states.shape[2:])

    new_states, Gamma_V, done = agent.compute_n_step_states(states, done, episode_mask, n_steps_mask_b)
    # merge batch and episode dimensions
    new_states = torch.tensor(new_states).float().to(agent.device).reshape((-1,)+states.shape[2:])
    
    ### Wrap variables into tensors - merge batch and episode dimensions ###

    done = torch.LongTensor(done.astype(int)).to(agent.device).reshape(-1)
    log_probs = torch.stack(log_probs).to(agent.device).transpose(1,0).reshape(-1)
    distributions = torch.stack(distributions, axis=0).to(agent.device).transpose(1,0).reshape(-1, agent.n_actions)
    mask = (distributions == 0).nonzero()
    distributions[mask[:,0], mask[:,1]] = 1e-5

    n_step_rewards = torch.tensor(n_step_rewards).float().to(agent.device).reshape(-1)
    Gamma_V = torch.tensor(Gamma_V).float().to(agent.device).reshape(-1)

    ### Update critic and then actor ###
    values, trg_values, critic_losses = inspect_critic_loss(agent, n_step_rewards, new_states, old_states, done, Gamma_V)

    advantages, actor_losses = inspect_actor_loss(agent, n_step_rewards, log_probs, distributions, 
                                                   new_states, old_states, done, Gamma_V)

    update_dict = dict(values=values, 
                       trg_values=trg_values, 
                       critic_losses=critic_losses, 
                       advantages=advantages, 
                       actor_losses=actor_losses )
    return update_dict

def inspect_critic_loss(agent, n_step_rewards, new_states, old_states, done, Gamma_V):
    with torch.no_grad():
        V_trg = agent.AC.V_critic(new_states).squeeze()
        V_trg = (1-done)*Gamma_V*V_trg + n_step_rewards
        V_trg = V_trg.squeeze()
        V = agent.AC.V_critic(old_states).squeeze()
    V = V.cpu().numpy() 
    V_trg = V_trg.cpu().numpy()
    critic_losses = (V-V_trg)**2
    return V, V_trg, critic_losses

def inspect_actor_loss(agent, n_step_rewards, log_probs, distributions, new_states, old_states, done, Gamma_V):
    with torch.no_grad():
        V_pred = agent.AC.V_critic(old_states).squeeze()
        V_trg = (1-done)*Gamma_V*agent.AC.V_critic(new_states).squeeze()  + n_step_rewards
        A = V_trg - V_pred
        policy_gradient = - log_probs*A
    A = A.cpu().numpy()
    pg = policy_gradient.cpu().numpy()
    return A, pg