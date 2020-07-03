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
from SC_Utils.A2C_inspection_MaxEnt import *
from SC_Utils.train_v3 import *

from pysc2.env import sc2_env
from pysc2.lib import actions

debug=False
inspection=True

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
        s_lst = np.array(s_lst).transpose(1,0,2,3,4)
        r_lst = np.array(r_lst).transpose(1,0)
        done_lst = np.array(done_lst).transpose(1,0)
        bootstrap_lst = np.array(bootstrap_lst).transpose(1,0)
        s_trg_lst = np.array(s_trg_lst).transpose(1,0,2,3,4)

        critic_loss, actor_loss = agent.compute_ac_loss(r_lst, log_probs, entropies, 
                                                                 s_lst, done_lst, bootstrap_lst, s_trg_lst)

        
        loss = (critic_loss + actor_loss).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        critic_losses.append(critic_loss.item())
        actor_losses.append(actor_loss.item())
        
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
    
    losses = dict(critic_losses=critic_losses, actor_losses=actor_losses)
    return score, losses, agent, PID