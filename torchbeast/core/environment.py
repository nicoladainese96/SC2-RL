# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The environment class for MonoBeast."""

import torch
import numpy as np
from pysc2.lib import actions

def merge_screen_and_minimap(state_dict):
    """
    Returns a tuple (state, player), where
    state = (screen+minimap channels, res, res) # no batch dim
    player = (player_features,) # no batch dim
    """
    screen = state_dict['screen_layers']
    minimap = state_dict['minimap_layers']
    player = state_dict['player_features']
    if len(minimap) > 0:
        try:
            assert screen.shape[-2:] == minimap.shape[-2:], 'different resolutions'
        except:
            print("Shape mismatch between screen and minimap. They must have the same resolution.")
            print("Screen resolution: ", screen.shape[-2:])
            print("Minimap resolution: ", minimap.shape[-2:])

        state = np.concatenate([screen, minimap])
    elif len(minimap)==0 and len(screen) >0:
        state = screen
    else:
        raise Exception("Both screen and minimap seem to have 0 layers.")
    return state, player 

def _format_state(spatial_state, player_state):
    """
    Returns
    -------
    spatial_state: tensor shape (in_channels, res, res)
    player_state: tensor shape (in_players,)
    """
    spatial_state = torch.from_numpy(spatial_state).float()
    player_state = torch.from_numpy(player_state).float()
    return spatial_state, player_state

class Environment:
    def __init__(self, env, obs_processer, random_seed=0):
        self.env = env
        self.obs_processer = obs_processer
        self.episode_return = None
        self.episode_step = None
        np.random.seed(random_seed) # used to de-synchronize parallel environments at the beginning

    def reset(self, skip_frame=True, frame_to_skip=1):
        obs = self.env.reset()
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32) # reset is counted 
        if skip_frame:
            for _ in range(frame_to_skip):
                action = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
                obs = self.env.step(actions = [action])
                self.episode_step += 1 # used internally, need to count all actions 
        return self.unpack_obs(obs) 
    
    def unpack_obs(self, obs):
        state_dict, _ = self.obs_processer.get_state(obs)  # returns (state_dict, names_dict)
        spatial_state, player_state = merge_screen_and_minimap(state_dict)
        reward = obs[0].reward
        done = obs[0].last()
        action_mask = self.obs_processer.get_action_mask(obs[0].observation.available_actions)
        return spatial_state, player_state, reward, done, action_mask
    
    def initial(self):
        initial_reward = torch.zeros(1, 1)
        #initial_done = torch.ones(1, 1, dtype=torch.uint8) # why True instead of False?
        initial_done = torch.zeros(1, 1, dtype=torch.uint8) # mine modification 
        initial_boostrap = torch.zeros(1, 1, dtype=torch.uint8)
        # skip a random number of frames in the first episode so that for all training all enviromnents are not
        # synchronized
        #spatial_state, player_state, reward, done, action_mask = self.reset(frame_to_skip=np.random.randint(240))
        spatial_state, player_state, reward, done, action_mask = self.reset()
        spatial_state, player_state = _format_state(spatial_state, player_state)
        
        return dict(
            spatial_state=spatial_state, 
            player_state=player_state,
            spatial_state_trg=spatial_state, 
            player_state_trg=player_state,
            action_mask=action_mask,
            reward=initial_reward,
            done=initial_done,
            bootstrap=initial_boostrap,
            episode_return=self.episode_return,
            episode_step=self.episode_step
        )

    def step(self, action):
        """
        Define action as [pysc2.actions.FunctionCall(action_id, [[arg1], [arg2], ...]]
        (Wrap it around a list)
        
        Notes
        -----
        - Automatically reset environment if a step is terminal
        - Everything is returned as tensors
        - spatial and player returned separately
        - All state tensors have been already processed by FullObsProcesser
        - state_trg[t+1] can be used to bootstrap the value for state[t] as 
          V(state[t]) = reward[t] + (done[t]+bootstrap[t])*discount*V(state_trg[t+1])
        - state[t] is equal to state_trg[t] if state_trg[t] wasn't terminal, otherwise
          is the state obtained by resetting the environment. Used to get current state
          value and to sample actions for next step.
        - bootstrap is used to signal steps in which done=True but only because of a time
          truncation, so we actually need to take into account the value of the last target state.
        """
        obs = self.env.step(actions=action) 
        spatial_state_trg, player_state_trg, reward, done, action_mask = self.unpack_obs(obs) 
        spatial_state_trg, player_state_trg = _format_state(spatial_state_trg, player_state_trg) 
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if done:
            if self.episode_step == 240:
                bootstrap = torch.tensor(True).view(1, 1) # time truncation
            else:
                bootstrap = torch.tensor(False).view(1, 1) # real end
            # only case in which spatial_state cannot be used for bootstrapping
            # because belongs to a new trajectory
            spatial_state, player_state, _, _, action_mask = self.reset()
            spatial_state, player_state = _format_state(spatial_state, player_state) 
        else:
            # already formatted
            bootstrap = torch.tensor(False).view(1, 1) # default case
            spatial_state = spatial_state_trg
            player_state = player_state_trg
        
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)
        return dict(
            spatial_state=spatial_state, 
            player_state=player_state,
            spatial_state_trg=spatial_state, 
            player_state_trg=player_state,
            action_mask=action_mask,
            reward=reward,
            done=done,
            bootstrap=bootstrap,
            episode_return=episode_return,
            episode_step=episode_step
        )

    def close(self):
        self.env.close()

class Environment_v2(Environment):
    
    def unpack_obs(self, obs):
        state_dict, _ = self.obs_processer.get_state(obs)  # returns (state_dict, names_dict)
        screen = state_dict['screen_layers']
        minimap = state_dict['minimap_layers']
        player = state_dict['player_features']
        reward = obs[0].reward
        done = obs[0].last()
        
        last_action = obs[0].observation['last_actions']
        if len(last_action) == 0:
            last_action = 0
        else:
            last_action = last_action[0]
        last_action_idx = np.where(self.obs_processer.action_table == last_action)[0]
        action_mask = self.obs_processer.get_action_mask(obs[0].observation.available_actions)
        return screen, minimap, player_state, last_action_idx, reward, done, action_mask
    
    def initial(self):
        initial_reward = torch.zeros(1, 1)
        initial_done = torch.zeros(1, 1, dtype=torch.uint8) # mine modification 
        initial_boostrap = torch.zeros(1, 1, dtype=torch.uint8)
        screen, minimap, player_state, last_action, reward, done, action_mask = self.reset() # action_mask is already a tensor
        screen, minimap, player, last_action = self._format_state(screen, minimap, player, last_action)
        
        return dict(
            screen_layers=screen, 
            minimap_layers=minimap,
            player_state=player_state,
            screen_layers_trg=screen, 
            minimap_layers_trg=minimap,
            player_state_trg=player_state,
            action_mask=action_mask,
            last_action=last_action,
            reward=initial_reward,
            done=initial_done,
            bootstrap=initial_boostrap,
            episode_return=self.episode_return,
            episode_step=self.episode_step
        )
    
    def step(self, action):
        """
        Define action as [pysc2.actions.FunctionCall(action_id, [[arg1], [arg2], ...]]
        (Wrap it around a list)
        
        Notes
        -----
        - Automatically reset environment if a step is terminal
        - Everything is returned as tensors
        - spatial and player returned separately
        - All state tensors have been already processed by FullObsProcesser
        - state_trg[t+1] can be used to bootstrap the value for state[t] as 
          V(state[t]) = reward[t] + (done[t]+bootstrap[t])*discount*V(state_trg[t+1])
        - state[t] is equal to state_trg[t] if state_trg[t] wasn't terminal, otherwise
          is the state obtained by resetting the environment. Used to get current state
          value and to sample actions for next step.
        - bootstrap is used to signal steps in which done=True but only because of a time
          truncation, so we actually need to take into account the value of the last target state.
        """
        obs = self.env.step(actions=action) 
        screen_trg, minimap_trg, player_state_trg, last_action, reward, done, action_mask = self.unpack_obs(obs) 
        screen_trg, minimap_trg, player_state_trg, last_action = self._format_state(screen_trg, 
                                                                                    minimap_trg, 
                                                                                    player_state_trg, 
                                                                                    last_action) 
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if done:
            if self.episode_step == 240:
                bootstrap = torch.tensor(True).view(1, 1) # time truncation
            else:
                bootstrap = torch.tensor(False).view(1, 1) # real end
            # only case in which spatial_state cannot be used for bootstrapping
            # because belongs to a new trajectory
            screen, minimap, player_state, last_action, _, _, action_mask = self.reset()
            screen, minimap, player_state, last_action = _format_state(screen, minimap, player_state, last_action) 
        else:
            # already formatted
            bootstrap = torch.tensor(False).view(1, 1) # default case
            screen = screen_trg
            minimap = minimap_trg
            player_state = player_state_trg
        
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)
        return dict(
            screen_layers=screen, 
            minimap_layers=minimap,
            player_state=player_state,
            screen_layers_trg=screen_trg, 
            minimap_layers_trg=minimap_trg,
            player_state_trg=player_state_trg,
            action_mask=action_mask,
            last_action=last_action,
            reward=reward,
            done=done,
            bootstrap=boostrap,
            episode_return=self.episode_return,
            episode_step=self.episode_step
        )
    
    @staticmethod
    def _format_state(screen, minimap, player, last_action):
        screen = torch.from_numpy(screen).float()
        minimap = torch.from_numpy(minimap).float()
        player = torch.from_numpy(player).float()
        last_action = torch.LongTensor(last_action)
        return screen, minimap, player, last_action