"""
Version used for parallel parameters sampling
Main changes:
- get_action_mask from action_table instead of action_dict
"""

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
from SC_Utils.game_utils import FullObsProcesser
from SC_Utils.A2C_inspection import *
# Various versions have different functions
from SC_Utils.train_v2 import gen_PID, init_game
from SC_Utils.train_v3 import reset_and_skip_first_frame
from SC_Utils.train_v4 import merge_screen_and_minimap # handles also player info

from pysc2.env import sc2_env
from pysc2.lib import actions # used somewhere?

debug=False
inspection=False # does not work yet

def get_action_mask(available_actions, action_table):
    """
    Creates a mask of length action_table with zeros (negated True casted to float) 
    in the positions of available actions and ones (negated False casted to float) 
    in the other positions. 
    """
    action_mask = ~np.array([a in available_actions for a in action_table])
    return action_mask

def worker(worker_id, master_end, worker_end, game_params, map_name, obs_proc_params, action_table):

    master_end.close()  # Forbid worker to use the master end for messaging
    np.random.seed() # sets random seed for the environment
    env = init_game(game_params, map_name, random_seed=np.random.randint(10000))
    op = FullObsProcesser(**obs_proc_params)
    
    while True:
        cmd, data = worker_end.recv()
        if cmd == 'step':
            obs = env.step([data])
            state_trg_dict, _ = op.get_state(obs)  #returns (state_dict, names_dict)
            state_trg = merge_screen_and_minimap(state_trg_dict) # state now is a tuple
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
                obs = reset_and_skip_first_frame(env)
                state_dict, _ = op.get_state(obs)  # returns (state_dict, names_dict)
                state = merge_screen_and_minimap(state_dict) # state now is a tuple
            else:
                state = state_trg
                
            available_actions = obs[0].observation.available_actions
            action_mask = get_action_mask(available_actions, action_table)
            worker_end.send((state, reward, done, bootstrap, state_trg, action_mask))
            
        elif cmd == 'reset':
            obs = reset_and_skip_first_frame(env)
            state_dict, _ = op.get_state(obs) # returns (state_dict, names_dict)
            state = merge_screen_and_minimap(state_dict) # state now is a tuple
            available_actions = obs[0].observation.available_actions
            action_mask = get_action_mask(available_actions, action_table)
          
            worker_end.send((state, action_mask))
        elif cmd == 'close':
            worker_end.close()
            break
        else:
            raise NotImplementedError

class ParallelEnv:
    def __init__(self, n_train_processes, game_params, map_name, obs_proc_params, action_table):
        self.nenvs = n_train_processes
        self.waiting = False
        self.closed = False
        self.workers = list()

        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)])
        self.master_ends, self.worker_ends = master_ends, worker_ends

        for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
            p = mp.Process(target=worker,
                           args=(worker_id, master_end, worker_end, game_params, 
                                 map_name, obs_proc_params, action_table))
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
        states = np.stack(states)
        states = {"spatial":np.stack(states[:,0]), "player":np.stack(states[:,1])}
        trg_states = np.stack(trg_states)
        trg_states = {"spatial":np.stack(trg_states[:,0]), "player":np.stack(trg_states[:,1])}
        return states, np.stack(rews), np.stack(dones), np.stack(bootstraps), trg_states, np.stack(action_mask)

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None))
        results = [master_end.recv() for master_end in self.master_ends]
        states, action_mask = zip(*results)
        states = np.stack(states)
        states = {"spatial":np.stack(states[:,0]), "player":np.stack(states[:,1])}
        return states, np.stack(action_mask)

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
        
def train_batched_A2C(agent, game_params, map_name, lr, n_train_processes, max_train_steps, 
                      unroll_length, obs_proc_params, 
                      test_interval=100, num_tests=5, inspection_interval=120000, save_path=None):
    if save_path is None:
        save_path = "../Results/"+map_name
    replay_dict = dict(save_replay_episodes=num_tests,
                       replay_dir='Replays/',
                       replay_prefix='A2C_'+map_name)
    action_table = agent.AC.action_table
    test_env = init_game(game_params, map_name, **replay_dict) # save just test episodes
    op = FullObsProcesser(**obs_proc_params)
    envs = ParallelEnv(n_train_processes, game_params, map_name, obs_proc_params, action_table)

    optimizer = torch.optim.Adam(agent.AC.parameters(), lr=lr)
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

        # all variables without gradient, batch first, then episode length
        #s_lst = np.array(s_lst).transpose(1,0,2,3,4) # no more an array, but a list of dictionaries of arrays
        r_lst = np.array(r_lst).transpose(1,0)
        done_lst = np.array(done_lst).transpose(1,0)
        bootstrap_lst = np.array(bootstrap_lst).transpose(1,0)
        #s_trg_lst = np.array(s_trg_lst).transpose(1,0,2,3,4) # no more an array, but a list of dictionaries of arrays

        critic_loss, actor_loss, entropy_term = agent.compute_ac_loss(r_lst, log_probs, entropies, 
                                                                 s_lst, done_lst, bootstrap_lst, s_trg_lst)

        
        loss = (critic_loss + actor_loss).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        critic_losses.append(critic_loss.item())
        actor_losses.append(actor_loss.item())
        entropy_losses.append(entropy_term.item())
        
        ### Test time ###
        if step_idx % test_interval == 0:
            if not os.path.isdir(save_path+'/Logging/'):
                os.system('mkdir '+save_path+'/Logging/')
            if step_idx // test_interval == 1:
                with open(save_path+'/Logging/'+PID+'.txt', 'a+') as f:
                    print("#Steps,score", file=f)
            avg_score = test(step_idx, agent, test_env, PID, op, action_table, num_tests, save_path)
            score.append(avg_score)
        if inspection and (step_idx%inspection_interval==0):
            inspector = inspection_test(step_idx, agent, test_env, PID, op, action_table)
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


def test(step_idx, agent, test_env, process_ID, op, action_table, num_test, save_path):
    score = 0.0
    done = False
            
    for _ in range(num_test):
        
        obs = reset_and_skip_first_frame(test_env)
        s_dict, _ = op.get_state(obs)
        spatial, player = merge_screen_and_minimap(s_dict)
        s = {"spatial":spatial, "player":player}
        for k in s.keys():
            s[k] = s[k][np.newaxis, ...] # add batch dim
        available_actions = obs[0].observation.available_actions
        a_mask = get_action_mask(available_actions, action_table)[np.newaxis, ...] # add batch dim
        
        while not done:
            a, log_prob, probs = agent.step(s, a_mask)
            obs = test_env.step(a)
            s_prime_dict, _ = op.get_state(obs) 
            spatial, player = merge_screen_and_minimap(s_prime_dict)
            s_prime = {"spatial":spatial, "player":player}
            for k in s_prime.keys():
                s_prime[k] = s_prime[k][np.newaxis, ...] # add batch dim
            reward = obs[0].reward
            done = obs[0].last()
            available_actions = obs[0].observation.available_actions
            a_mask = get_action_mask(available_actions, action_table)[np.newaxis, ...] # add batch dim
            
            s = s_prime
            score += reward
        done = False

    with open(save_path+'/Logging/'+process_ID+'.txt', 'a+') as f:
        print(f"{step_idx},{score/num_test:.1f}", file=f)
    return score/num_test

def inspection_test(step_idx, agent, test_env, process_ID, op, action_table):
    inspector = InspectionDict(step_idx, process_ID, agent)
    
    obs = reset_and_skip_first_frame(test_env)
    s_dict, _ = op.get_state(obs)
    spatial, player = merge_screen_and_minimap(s_dict)
    s = {"spatial":spatial, "player":player}
    for k in s.keys():
        s[k] = s[k][np.newaxis, ...] # add batch dim
    available_actions = obs[0].observation.available_actions
    a_mask = get_action_mask(available_actions, action_table)[np.newaxis, ...] # add batch dim
    
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
        spatial, player = merge_screen_and_minimap(s_prime_dict)
        s_prime = {"spatial":spatial, "player":player}
        for k in s_prime.keys():
            s_prime[k] = s_prime[k][np.newaxis, ...] # add batch dim
        reward = obs[0].reward
        done = obs[0].last()
        available_actions = obs[0].observation.available_actions
        a_mask = get_action_mask(available_actions, action_table)[np.newaxis, ...] # add batch dim
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
    r_lst = np.array(r_lst).reshape(1,-1)
    done_lst = np.array(done_lst).reshape(1,-1)
    bootstrap_lst = np.array(bootstrap_lst).reshape(1,-1)  
    update_dict = inspection_update(agent, r_lst, log_probs, entropies, s_lst, 
                                    done_lst, bootstrap_lst, s_trg_lst)
    inspector.store_update(update_dict)
    return inspector
