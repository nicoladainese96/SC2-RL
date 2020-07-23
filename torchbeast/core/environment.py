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
    def __init__(self, env, obs_processer):
        self.env = env
        self.obs_processer = obs_processer
        self.episode_return = None
        self.episode_step = None

    def reset(self, skip_first=True):
        obs = self.env.reset()
        if skip_first:
            action = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
            obs = self.env.step(actions = [action])
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
        # This supports only single-tensor actions ATM.
        self.episode_return = torch.zeros(1, 1) # (batch_dim, value) ?
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8) # why True instead of False?
        spatial_state, player_state, reward, done, action_mask = self.reset()
        spatial_state, player_state = _format_state(spatial_state, player_state)
        
        return dict(
            spatial_state=spatial_state, 
            player_state=player_state,
            action_mask=action_mask,
            reward=initial_reward,
            done=initial_done,
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
        """
        obs = self.env.step(actions=action) 
        spatial_state, player_state, reward, done, action_mask = self.unpack_obs(obs)
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if done:
            spatial_state, player_state, _, _, action_mask = self.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        spatial_state, player_state = _format_state(spatial_state, player_state) 
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)
        return dict(
            spatial_state=spatial_state, 
            player_state=player_state,
            action_mask=action_mask,
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step
        )

    def close(self):
        self.env.close()
