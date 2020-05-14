import torch
import torch.multiprocessing as mp
import time
import numpy as np

from pysc2.env import sc2_env
from pysc2.lib import features

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

def worker(worker_id, master_end, worker_end, game_params):
    master_end.close()  # Forbid worker to use the master end for messaging
    env = init_game(**game_params)
    np.random.seed(worker_id) # sets random seed for the environment

    while True:
        cmd, data = worker_end.recv()
        if cmd == 'step':
            ob_trg, reward, done, info = env.step(data)
            # Check if termination happened for time limit truncation or natural end
            if done and 'TimeLimit.truncated' in info:
                bootstrap = True
            else:
                bootstrap = False
            # ob_trg is the state used as next state for the update
            # ob is the new state used to decide the next action 
            # (different if the episode ends and another one begins)
            if done:
                ob = env.reset()
            else:
                ob = ob_trg
            worker_end.send((ob, reward, done, info, bootstrap, ob_trg)) # check this; bootstrap = False always?
        elif cmd == 'reset':
            ob = env.reset()
            worker_end.send(ob)
        elif cmd == 'close':
            worker_end.close()
            break
        else:
            raise NotImplementedError

class ParallelEnv:
    def __init__(self, n_train_processes, game_params):
        self.nenvs = n_train_processes
        self.waiting = False
        self.closed = False
        self.workers = list()

        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)])
        self.master_ends, self.worker_ends = master_ends, worker_ends

        for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
            p = mp.Process(target=worker,
                           args=(worker_id, master_end, worker_end, game_params))
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
        obs, rews, dones, infos, bootstraps, trg_obs = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos, np.stack(bootstraps), np.stack(trg_obs)

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None))
        return np.stack([master_end.recv() for master_end in self.master_ends])

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

# Aggiungere il test

def train_batched_A2C(agent, game_params, lr, n_train_processes, max_train_steps, unroll_length, test_interval=100):
    envs = ParallelEnv(n_train_processes, game_params)

    optimizer = optim.Adam(agent.parameters(), lr=lr)

    step_idx = 0
    s = envs.reset()
    while step_idx < max_train_steps:
        s_lst, a_lst, r_lst, done_lst, bootstrap_lst, s_trg_lst = list(), list(), list(), list(), list(), list()
        log_probs = []
        distributions = []
        for _ in range(unroll_length):

            a, log_prob, probs = agent.get_action(s)
            a_lst.append(a)
            log_probs.append(log_prob)
            distributions.append(probs)

            s_prime, r, done, info, bootstrap, s_trg = envs.step(a)
            s_lst.append(s)
            r_lst.append(r)
            done_lst.append(done)
            bootstrap_lst.append(bootstrap)
            s_trg_lst.append(s_trg)

            s = s_prime
            step_idx += n_train_processes

        ### Update time ###
        critic_loss, actor_loss, entropy = agent.compute_ac_loss(r_lst, log_probs, distributions, 
                                                                 s_lst, done_lst, bootstrap_lst, s_trg_lst)

        loss = (critic_loss + actor_loss).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ### 
        
        #if step_idx % PRINT_INTERVAL == 0:
        #    test(step_idx, model)

    envs.close()
    return #cose

def test(step_idx, agent, game_params):
    env = test_env.Sandbox(**game_params)
    score = 0.0
    done = False
    num_test = 10
    steps_to_solve = 0
    for _ in range(num_test):
        s = env.random_reset()
        rewards = []
        while not done:
            a, log_prob, probs = agent.get_action(s)
            s_prime, r, done, info = env.step(a[0])
            s = s_prime
            score += r
            rewards.append(r)
        done = False
        steps_to_solve += len(rewards)
    print(f"Step # :{step_idx}, avg score : {score/num_test:.1f}, avg steps to solve : {steps_to_solve/num_test}")
    return score/num_test, steps_to_solve/num_test

def train_batched_A2C(agent, game_params, lr, n_train_processes, max_train_steps, unroll_length, test_interval=100):
    envs = ParallelEnv(n_train_processes, game_params)

    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    score = []
    steps_to_solve = []
    critic_losses = [] 
    actor_losses = []
    entropies = []
    
    step_idx = 0
    s = envs.reset()
    while step_idx < max_train_steps:
        s_lst, a_lst, r_lst, done_lst, bootstrap_lst, s_trg_lst = list(), list(), list(), list(), list(), list()
        log_probs = []
        distributions = []
        for _ in range(unroll_length):

            a, log_prob, probs = agent.get_action(s)
            a_lst.append(a)
            log_probs.append(log_prob)
            distributions.append(probs)

            s_prime, r, done, info, bootstrap, s_trg = envs.step(a)
            s_lst.append(s)
            r_lst.append(r)
            done_lst.append(done)
            bootstrap_lst.append(bootstrap)
            s_trg_lst.append(s_trg)

            s = s_prime
            step_idx += 1 #n_train_processes

        s_lst = np.array(s_lst).transpose(1,0,2,3,4)
        r_lst = np.array(r_lst).transpose(1,0)
        done_lst = np.array(done_lst).transpose(1,0)
        bootstrap_lst = np.array(bootstrap_lst).transpose(1,0)
        s_trg_lst = np.array(s_trg_lst).transpose(1,0,2,3,4)
        
        ### Update time ###
        #print("len(r_lst): ", r_lst.shape)
        #print("len(s_lst): ", s_lst.shape)
        #print("len(done_lst): ", done_lst.shape)
        #print("len(s_trg_lst): ", s_trg_lst.shape)
        critic_loss, actor_loss, entropy = agent.compute_ac_loss(r_lst, log_probs, distributions, 
                                                                 s_lst, done_lst, bootstrap_lst, s_trg_lst)

        loss = (critic_loss + actor_loss).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #print("critic_loss: ", critic_loss)
        #print("actor_loss: ", actor_loss)
        #print("entropy: ", entropy)
        critic_losses.append(critic_loss.item())
        actor_losses.append(actor_loss.item())
        entropies.append(entropy.item())
        
        ### Test time ###
        if step_idx % test_interval == 0:
            avg_score, avg_steps = test(step_idx, agent, game_params)
            score.append(avg_score)
            steps_to_solve.append(avg_steps)
    envs.close()
    
    losses = dict(critic_losses=critic_losses, actor_losses=actor_losses, entropies=entropies)
    return score, steps_to_solve, losses, agent