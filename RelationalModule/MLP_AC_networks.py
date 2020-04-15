import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from pysc2.lib import actions

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id


class Actor(nn.Module):
    """
    Network used to parametrize the policy of an agent.
    Uses 3 linear layers, the first 2 with ReLU activation,
    the third with softmax.
    """
    
    def __init__(self, action_space, observation_space, hiddens=[64,32]):
        """
        Parameters
        ----------
        observation_space: int
            Number of flattened entries of the state
        action_space: int
            Number of (discrete) possible actions to take
        discrete: bool
            If True, adds an embedding layer before the linear layers
        project_dim: int
            Dimension of the embedding space
        hiddens: list of int (default = [64,32])
            List containing the number of neurons of each linear hidden layer.
        """
        super(Actor, self).__init__()
        self.action_dict = {0:_NO_OP, 1:_SELECT_ARMY, 2:_MOVE_SCREEN}
        layers = []
        
        layers.append(nn.Linear(observation_space, hiddens[0]))
        layers.append(nn.ReLU())
            
        for i in range(0,len(hiddens)-1):
            layers.append(nn.Linear(hiddens[i], hiddens[i+1]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hiddens[-1], action_space))
        self.net = nn.Sequential(*layers)
        
    def forward(self, state, available_actions):
        logits = self.net(state)
        mask = self.get_action_mask(available_actions)
        logits[mask] = torch.tensor(np.NINF)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
    
    def get_action_mask(self, available_actions):
        action_mask = ~np.array([self.action_dict[i] in available_actions for i in self.action_dict.keys()])
        return action_mask
        
class BasicCritic(nn.Module):
    """
    Network used to parametrize the Critic of an Actor-Critic agent.
    Uses 3 linear layers, only the first 2 with ReLU activation.
    Returns the value of a state.
    """
    
    def __init__(self, observation_space, discrete=False, project_dim=4, hiddens=[64,32]):
        """
        Parameters
        ----------
        observation_space: int
            Number of flattened entries of the state
        hiddens: list of int (default = [64,32])
            List containing the number of neurons of each linear hidden layer.
        """
        super(BasicCritic, self).__init__()
        self.discrete = discrete
        
        layers = []
        
        if self.discrete:
            layers.append(nn.Embedding(observation_space, project_dim))
            layers.append(nn.Linear(project_dim, hiddens[0]))
            layers.append(nn.ReLU())
        else:
            layers.append(nn.Linear(observation_space, hiddens[0]))
            layers.append(nn.ReLU())
            
        for i in range(0,len(hiddens)-1):
            layers.append(nn.Linear(hiddens[i], hiddens[i+1]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hiddens[-1], 1)) 
        self.net = nn.Sequential(*layers)
    
    def forward(self, state):
        V = self.net(state)
        return V
    
class Critic(nn.Module):
    """Implements a generic critic, that can have 2 independent networks is twin=True. """
    def __init__(self, observation_space, discrete=False, project_dim=4, twin=False, target=False, hiddens=[64,32]):
        super(Critic, self).__init__()
        
        self.twin = twin
        self.target = target
        
        if twin:
            self.net1 = BasicCritic(observation_space, discrete, project_dim, hiddens)
            self.net2 = BasicCritic(observation_space, discrete, project_dim, hiddens)
        else:
            self.net = BasicCritic(observation_space, discrete, project_dim, hiddens)
        
    def forward(self, state):
        if self.twin:
            v1 = self.net1(state)
            v2 = self.net2(state)
            if self.target:
                v = torch.min(v1, v2) # one could also try with the mean, for a less unbiased estimate
            else:
                return v1, v2
        else:
            v = self.net(state)
            
        return v
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    