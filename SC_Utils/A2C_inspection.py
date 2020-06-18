import torch
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F 
from pysc2.lib import actions

class InspectionDict():
    def __init__(self, step_idx, PID, action_dict, env):
        self.step_idx = step_idx
        self.PID = PID
        self.action_dict = action_dict
        self.all_actions = env.action_spec()[0][1]
        self.all_arguments = env.action_spec()[0][0]
        self.dict = dict(
                        state_traj = [],
                        rewards = [],
                        action_distr = [],
                        action_sel = [],
                        args = [],
                        values = None,
                        trg_values = None,
                        critic_losses = None,
                        advantages = None,
                        actor_losses = None)
        self.set_unique_args(action_dict)
        
    def set_unique_args(self, action_dict):
        arg_names = []
        arg_ids = []
        for a in action_dict:
            action_id = action_dict[a]
            args = self.all_actions[action_id].args
            for arg in args:
                arg_name = str(action_id.name)+'/'+arg.name
                arg_names.append(arg_name)
                arg_ids.append(arg.id)

        unique_args, unique_idx = np.unique(arg_names, return_index=True)
        arg_ids = np.array(arg_ids)
        unique_ids = arg_ids[unique_idx]
        arg_names = np.array(arg_names)
        arg_names = arg_names[unique_idx]
        
        spatial = []
        for i, arg in enumerate(unique_args):
            size = self.all_arguments[unique_ids[i]].sizes
            if len(size) == 1:
                spatial.append(False)
            else:
                spatial.append(True)
        self.arg_names = arg_names
        for n in self.arg_names:
            self.dict[n+'_distr'] = []
        self.spatial = spatial

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
    state = torch.from_numpy(state).float().to(agent.device)
    action_mask = torch.tensor(action_mask).to(agent.device)
    log_probs, spatial_features, nonspatial_features = agent.AC.pi(state, action_mask)
    probs = torch.exp(log_probs)
    entropy = agent.compute_entropy(probs)
    a = Categorical(probs).sample()
    a = a.detach().cpu().numpy()
    #embedded_a = agent._embed_action(a)
    ### Inspection ###
    step_dict = {}
    p = probs.detach().cpu().numpy() 
    step_dict['action_distr'] = p
    step_dict['action_sel'] = a
    
    # Concatenate embedded action to spatial and nonspatial features
    #spatial_features = agent._cat_action_to_spatial(embedded_a, spatial_features)
    #nonspatial_features = agent._cat_action_to_nonspatial(embedded_a, nonspatial_features)
    
    # All this sampling is completely wrong - but distributions are ok
    with torch.no_grad():
        for i, name in enumerate(inspector.arg_names):
            if inspector.spatial[i]:
                insp_arg, insp_log_prob, insp_distr = agent.AC.sample_param(spatial_features, name)
                p = insp_distr.detach().cpu().numpy().reshape(state.shape[-2:]) 
                step_dict[name+'_distr'] = p
            else:
                insp_arg, insp_log_prob, insp_distr = agent.AC.sample_param(nonspatial_features, name)
                p = insp_distr.detach().cpu().numpy() 
                step_dict[name+'_distr'] = p
    ### End inspection ###
    log_prob = log_probs[range(len(a)), a]
    
    action_id = np.array([agent.AC.action_dict[act] for act in a])
    args, args_log_prob, args_entropy = agent.get_arguments(spatial_features, nonspatial_features, a)
    step_dict['args'] = args
    
    log_prob = log_prob + args_log_prob
    entropy = entropy + args_entropy
    action = [actions.FunctionCall(action_id[i], args[i]) for i in range(len(action_id))]

    inspector.store_step(step_dict)
    return action, log_prob, entropy

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
