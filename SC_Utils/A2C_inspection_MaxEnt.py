import torch
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F 
from pysc2.lib import actions

from SC_Utils.A2C_inspection import InspectionDict, inspection_step
            
def inspection_update(agent, rewards, log_probs, entropies, states, done, bootstrap, trg_states): 
    old_states = torch.tensor(states).float().to(agent.device).reshape((-1,)+states.shape[2:])
    entropies = torch.stack(entropies, axis=0).transpose(1,0).to(agent.device)
    V_trg = agent.compute_n_step_V_trg(agent.n_steps, rewards, done, bootstrap, states, entropies)
    log_probs = torch.stack(log_probs).to(agent.device).transpose(1,0).reshape(-1)

    values, trg_values, critic_losses = inspect_critic_loss(agent, old_states, V_trg)

    advantages, actor_losses = inspect_actor_loss(agent, log_probs, old_states, V_trg)

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
        V_trg = V_trg.detach().cpu().numpy()
        critic_losses = (V-V_trg)**2
    return V, V_trg, critic_losses

def inspect_actor_loss(agent, log_probs,old_states, V_trg):
    with torch.no_grad():
        V_pred = agent.AC.V_critic(old_states).squeeze()
        A = V_trg.detach() - V_pred
        policy_gradient = - log_probs*A
    A = A.cpu().numpy()
    pg = policy_gradient.cpu().numpy()
    return A, pg
