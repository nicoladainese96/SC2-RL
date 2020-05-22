import torch
import torch.multiprocessing as mp
import time
import numpy as np

from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.lib import actions

# Useful aliases for actions
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id

action_dict = {0:_NO_OP, 1:_SELECT_ARMY, 2:_MOVE_SCREEN} # global variable

# indexes of useful layers of the screen_features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index 
_SELECTED = features.SCREEN_FEATURES.selected.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Identifiers in player_relative feature layer
_BACKGROUND = 0
_PLAYER_FRIENDLY = 1
_PLAYER_ALLIES = 2
_PLAYER_NEUTRAL = 3
_PLAYER_HOSTILE = 4

debug=False

def get_action_mask(available_actions):
    action_mask = ~np.array([action_dict[i] in available_actions for i in action_dict.keys()])
    return action_mask

def get_coord_state(obs):

    player_relative = obs[0].observation['feature_screen'][_PLAYER_RELATIVE]

    beacon_ys, beacon_xs = (player_relative == _PLAYER_NEUTRAL).nonzero()
    player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()

    if beacon_ys.any():
        beacon_pos = [beacon_xs.mean(), beacon_ys.mean()]
    else:
        beacon_pos = [-1., -1.] # not present

    if player_y.any():
        player_pos = [player_x.mean(), player_y.mean()]
    else:
        player_pos = beacon_pos # if the two are superimposed, only beacon cells are showed

    beacon_exists = float(beacon_ys.any())

    selected = obs[0].observation['feature_screen'][_SELECTED]
    is_selected = np.any((selected==1).nonzero()[0]).astype(float) 

    state = np.concatenate([player_pos, beacon_pos, [beacon_exists, is_selected]])

    return state

def get_ohe_state(obs):
    
    player_relative = obs[0].observation['feature_screen'][_PLAYER_RELATIVE]
    selected = obs[0].observation['feature_screen'][_SELECTED].astype(float)
    
    friendly = (player_relative == _PLAYER_FRIENDLY).astype(float)
    neutral = (player_relative == _PLAYER_NEUTRAL).astype(float)
    
    state = np.zeros((3,)+player_relative.shape).astype(float)
    state[0] = friendly
    state[1] = neutral
    state[2] = selected
       
    return state

def init_game(game_params, max_steps=256, step_multiplier=8, **kwargs):

    race = sc2_env.Race(1) # 1 = terran
    agent = sc2_env.Agent(race, "Testv0") # NamedTuple [race, agent_name]
    agent_interface_format = sc2_env.parse_agent_interface_format(**game_params) #AgentInterfaceFormat instance

    game_params = dict(map_name='MoveToBeacon', # simplest minigame
                       players=[agent], # use a list even for single player
                       game_steps_per_episode = max_steps*step_multiplier,
                       agent_interface_format=[agent_interface_format] # use a list even for single player
                       )  
    env = sc2_env.SC2Env(**game_params, **kwargs)

    return env

def worker(worker_id, master_end, worker_end, game_params, max_steps):
    master_end.close()  # Forbid worker to use the master end for messaging
    np.random.seed() # sets random seed for the environment
    env = init_game(game_params, max_steps, random_seed=np.random.randint(10000))
    
    
    while True:
        cmd, data = worker_end.recv()
        if cmd == 'step':
            obs = env.step([data])
            state_trg = get_ohe_state(obs)
            reward = obs[0].reward
            done = obs[0].last()
            available_actions = obs[0].observation.available_actions
            action_mask = get_action_mask(available_actions)
            
            # Always bootstrap when episode finishes (in MoveToBeacon there is no real end)
            if done:
                bootstrap = True
            else:
                bootstrap = False
                
            # ob_trg is the state used as next state for the update
            # ob is the new state used to decide the next action 
            # (different if the episode ends and another one begins)
            if done:
                obs = env.reset()
                state = get_ohe_state(obs)
            else:
                state = state_trg
                
            worker_end.send((state, reward, done, bootstrap, state_trg, action_mask))
            
        elif cmd == 'reset':
            obs = env.reset()
            state = get_ohe_state(obs)
            available_actions = obs[0].observation.available_actions
            action_mask = get_action_mask(available_actions)
            
            worker_end.send((state, action_mask))
        elif cmd == 'close':
            worker_end.close()
            break
        else:
            raise NotImplementedError

class ParallelEnv:
    def __init__(self, n_train_processes, game_params, max_steps):
        self.nenvs = n_train_processes
        self.waiting = False
        self.closed = False
        self.workers = list()

        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)])
        self.master_ends, self.worker_ends = master_ends, worker_ends

        for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
            p = mp.Process(target=worker,
                           args=(worker_id, master_end, worker_end, game_params, max_steps))
            p.daemon = True
            p.start()
            self.workers.append(p)

        # Forbid master to use the worker end for messaging
        for worker_end in worker_ends:
            worker_end.close()

    def step_async(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [master_end.recv() for master_end in self.master_ends]
        self.waiting = False
        states, rews, dones, bootstraps, trg_states, action_mask = zip(*results)
        return np.stack(states), np.stack(rews), np.stack(dones), np.stack(bootstraps), np.stack(trg_states), np.stack(action_mask)

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None))
        results = [master_end.recv() for master_end in self.master_ends]
        states, action_mask = zip(*results)
        return np.stack(states), np.stack(action_mask)

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):  # For clean up resources
        if self.closed:
            return
        if self.waiting:
            [master_end.recv() for master_end in self.master_ends]
        for master_end in self.master_ends:
            master_end.send(('close', None))
        for worker in self.workers:
            worker.join()
            self.closed = True
