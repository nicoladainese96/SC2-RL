import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import time
import numpy as np
import string
import random
import copy

import os
import sys
sys.path.insert(0, "../")
from SC_Utils.game_utils import ObsProcesser
from SC_Utils.A2C_inspection import *

from pysc2.env import sc2_env

debug=False
inspection=True

def gen_PID():
    ID = ''.join([random.choice(string.ascii_letters) for _ in range(4)])
    ID = ID.upper()
    return ID

def get_action_mask(available_actions, action_dict):
    action_mask = ~np.array([action_dict[i] in available_actions for i in action_dict.keys()])
    return action_mask

def merge_screen_and_minimap(state_dict):
    screen = state_dict['screen_layers']
    minimap = state_dict['minimap_layers']
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
    return state              

def init_game(game_params, map_name='MoveToBeacon', step_multiplier=8, **kwargs):

    race = sc2_env.Race(1) # 1 = terran
    agent = sc2_env.Agent(race, "Testv0") # NamedTuple [race, agent_name]
    agent_interface_format = sc2_env.parse_agent_interface_format(**game_params) #AgentInterfaceFormat instance

    game_params = dict(map_name=map_name, 
                       players=[agent], # use a list even for single player
                       game_steps_per_episode = 0,
                       step_mul = step_multiplier,
                       agent_interface_format=[agent_interface_format] # use a list even for single player
                       )  
    env = sc2_env.SC2Env(**game_params, **kwargs)

    return env

def worker(worker_id, master_end, worker_end, game_params, map_name, obs_proc_params, action_dict):
    master_end.close()  # Forbid worker to use the master end for messaging
    np.random.seed() # sets random seed for the environment
    env = init_game(game_params, map_name, random_seed=np.random.randint(10000))
    op = ObsProcesser(**obs_proc_params)
    
    while True:
        cmd, data = worker_end.recv()
        if cmd == 'step':
            obs = env.step([data])
            state_trg_dict, _ = op.get_state(obs)  #returns (state_dict, names_dict)
            state_trg = merge_screen_and_minimap(state_trg_dict)
            reward = obs[0].reward
            done = obs[0].last()
            
            # Always bootstrap when episode finishes (in MoveToBeacon there is no real end)
            if done:
                bootstrap = True
            else:
                bootstrap = False
                
            # state_trg is the state used as next state for the update
            # state is the new state used to decide the next action 
            # (different if the episode ends and another one begins)
            if done:
                obs = env.reset()
                state_dict, _ = op.get_state(obs)  # returns (state_dict, names_dict)
                state = merge_screen_and_minimap(state_dict)
            else:
                state = state_trg
                
            available_actions = obs[0].observation.available_actions
            action_mask = get_action_mask(available_actions, action_dict)
            worker_end.send((state, reward, done, bootstrap, state_trg, action_mask))
            
        elif cmd == 'reset':
            obs = env.reset()
            state_dict, _ = op.get_state(obs) # returns (state_dict, names_dict)
            state = merge_screen_and_minimap(state_dict)
            available_actions = obs[0].observation.available_actions
            action_mask = get_action_mask(available_actions, action_dict)
            
            worker_end.send((state, action_mask))
        elif cmd == 'close':
            worker_end.close()
            break
        else:
            raise NotImplementedError

class ParallelEnv:
    def __init__(self, n_train_processes, game_params, map_name, obs_proc_params, action_dict):
        self.nenvs = n_train_processes
        self.waiting = False
        self.closed = False
        self.workers = list()

        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)])
        self.master_ends, self.worker_ends = master_ends, worker_ends

        for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
            p = mp.Process(target=worker,
                           args=(worker_id, master_end, worker_end, game_params, 
                                 map_name, obs_proc_params, action_dict))
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
            
class H_linear_schedule():
    def __init__(self, H_init, H_end, max_train_steps):
        self.H0 = H_init
        self.m = (H_end-H_init)/max_train_steps
        self.max_steps = max_train_steps
        
    def get_H(self, step_idx):
        # clip at max_step (independent assurance from init conditions)
        if step_idx > self.max_steps:
            step_idx = self.max_steps
        return self.H0 + self.m*step_idx
    
def train_batched_A2C(agent, game_params, map_name, lr, n_train_processes, max_train_steps, 
                      unroll_length, obs_proc_params, action_dict,
                      test_interval=100, num_tests=5, inspection_interval=120000, save_path=None):
    if save_path is None:
        save_path = "../Results/"+map_name
    replay_dict = dict(save_replay_episodes=num_tests,
                       replay_dir='Replays/',
                       replay_prefix='A2C_'+map_name)
    test_env = init_game(game_params, map_name, **replay_dict) # save just test episodes
    op = ObsProcesser(**obs_proc_params)
    envs = ParallelEnv(n_train_processes, game_params, map_name, obs_proc_params, action_dict)

    optimizer = torch.optim.Adam(agent.AC.parameters(), lr=lr)
    H_schedule = H_linear_schedule(agent.H, agent.H/10, max_train_steps)
    PID = gen_PID()
    print("Process ID: ", PID)
    score = []
    critic_losses = [] 
    actor_losses = []
    entropy_losses = []
    
    step_idx = 0
    s, a_mask = envs.reset() # reset manually only at the beginning
    while step_idx < max_train_steps:
        s_lst, r_lst, done_lst, bootstrap_lst, s_trg_lst = list(), list(), list(), list(), list()
        log_probs = []
        entropies = []
        for _ in range(unroll_length):

            a, log_prob, entropy = agent.step(s, a_mask)
            # variables with gradient
            log_probs.append(log_prob)
            entropies.append(entropy)

            s_prime, r, done, bootstrap, s_trg, a_mask = envs.step(a)
            s_lst.append(s)
            r_lst.append(r)
            done_lst.append(done)
            bootstrap_lst.append(bootstrap)
            s_trg_lst.append(s_trg)

            s = s_prime
            step_idx += 1 #n_train_processes

        # all variables without gradient
        s_lst = np.array(s_lst).transpose(1,0,2,3,4)
        r_lst = np.array(r_lst).transpose(1,0)
        done_lst = np.array(done_lst).transpose(1,0)
        bootstrap_lst = np.array(bootstrap_lst).transpose(1,0)
        s_trg_lst = np.array(s_trg_lst).transpose(1,0,2,3,4)

        critic_loss, actor_loss, entropy_term = agent.compute_ac_loss(r_lst, log_probs, entropies, 
                                                                 s_lst, done_lst, bootstrap_lst, s_trg_lst)

        
        loss = (critic_loss + actor_loss).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        critic_losses.append(critic_loss.item())
        actor_losses.append(actor_loss.item())
        entropy_losses.append(entropy_term.item())
        
        H = H_schedule.get_H(step_idx)
        agent.H = H
        
        ### Test time ###
        if step_idx % test_interval == 0:
            if not os.path.isdir(save_path+'/Logging/'):
                os.system('mkdir '+save_path+'/Logging/')
            if step_idx // test_interval == 1:
                with open(save_path+'/Logging/'+PID+'.txt', 'a+') as f:
                    print("#Steps,score", file=f)
            avg_score = test(step_idx, agent, test_env, PID, op, action_dict, num_tests, save_path)
            score.append(avg_score)
        if inspection and (step_idx%inspection_interval==0):
            inspector = inspection_test(step_idx, agent, test_env, PID, op, action_dict)
            # save episode for inspection and model weights at that point
            if not os.path.isdir(save_path):
                os.system('mkdir '+save_path)
            if not os.path.isdir(save_path+'/Inspection/'):
                os.system('mkdir '+save_path+'/Inspection/')
            if not os.path.isdir(save_path+'/Checkpoints/'):
                os.system('mkdir '+save_path+'/Checkpoints/')
            inspector.save_dict(path=save_path+'/Inspection/')
            torch.save(agent.AC.state_dict(), save_path+'/Checkpoints/'+PID+'_'+str(step_idx))
    envs.close()
    
    losses = dict(critic_losses=critic_losses, actor_losses=actor_losses, entropies=entropy_losses)
    return score, losses, agent, PID

def test(step_idx, agent, test_env, process_ID, op, action_dict, num_test, save_path):
    score = 0.0
    done = False
            
    for _ in range(num_test):
        
        obs = test_env.reset()
        s_dict, _ = op.get_state(obs)
        s = merge_screen_and_minimap(s_dict)
        s = s[np.newaxis, ...] # add batch dim
        available_actions = obs[0].observation.available_actions
        a_mask = get_action_mask(available_actions, action_dict)[np.newaxis, ...] # add batch dim
        
        while not done:
            a, log_prob, probs = agent.step(s, a_mask)
            obs = test_env.step(a)
            s_prime_dict, _ = op.get_state(obs) 
            s_prime = merge_screen_and_minimap(s_prime_dict)
            s_prime = s_prime[np.newaxis, ...] # add batch dim
            reward = obs[0].reward
            done = obs[0].last()
            available_actions = obs[0].observation.available_actions
            a_mask = get_action_mask(available_actions, action_dict)[np.newaxis, ...] # add batch dim
            
            s = s_prime
            score += reward
        done = False

    with open(save_path+'/Logging/'+process_ID+'.txt', 'a+') as f:
        print(f"{step_idx},{score/num_test:.1f}", file=f)
    return score/num_test

def inspection_test(step_idx, agent, test_env, process_ID, op, action_dict):
    inspector = InspectionDict(step_idx, process_ID, action_dict, test_env)
    
    obs = test_env.reset()
    s_dict, _ = op.get_state(obs)
    s = merge_screen_and_minimap(s_dict)
    s = s[np.newaxis, ...] # add batch dim
    available_actions = obs[0].observation.available_actions
    a_mask = get_action_mask(available_actions, action_dict)[np.newaxis, ...] # add batch dim
    
    done = False
    G = 0.0
    # list used for update
    s_lst, r_lst, done_lst, bootstrap_lst, s_trg_lst = list(), list(), list(), list(), list()
    log_probs = []
    entropies = []
    while not done:
        a, log_prob, entropy = inspection_step(agent, inspector, s, a_mask)
        log_probs.append(log_prob)
        entropies.append(entropy)
        obs = test_env.step(a)
        s_prime_dict, _ = op.get_state(obs) 
        s_prime = merge_screen_and_minimap(s_prime_dict)
        s_prime = s_prime[np.newaxis, ...] # add batch dim
        reward = obs[0].reward
        done = obs[0].last()
        available_actions = obs[0].observation.available_actions
        a_mask = get_action_mask(available_actions, action_dict)[np.newaxis, ...] # add batch dim
        if done:
            bootstrap = True
        else:
            bootstrap = False
            
        inspector.dict['state_traj'].append(s)
        s_lst.append(s)
        r_lst.append(reward)
        done_lst.append(done)
        bootstrap_lst.append(bootstrap)
        s_trg_lst.append(s_prime)
            
        s = s_prime
        G += reward
        
    inspector.dict['rewards'] = r_lst
    s_lst = np.array(s_lst).transpose(1,0,2,3,4)
    r_lst = np.array(r_lst).reshape(1,-1)
    done_lst = np.array(done_lst).reshape(1,-1)
    bootstrap_lst = np.array(bootstrap_lst).reshape(1,-1)
    s_trg_lst = np.array(s_trg_lst).transpose(1,0,2,3,4)    
    update_dict = inspection_update(agent, r_lst, log_probs, entropies, s_lst, 
                                    done_lst, bootstrap_lst, s_trg_lst)
    inspector.store_update(update_dict)
    return inspector
