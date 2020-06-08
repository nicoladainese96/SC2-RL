import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import time
import numpy as np
import string
import random
import copy

from Utils.REINFORCE_inspection import *
from Utils.train_batched_A2C import gen_PID, ParallelEnv, get_action_mask, select_army_mask, move_screen_mask, get_ohe_state, init_game

debug=False
simplified = True

def compute_actor_loss(agent, log_probs, rewards, done, bootstrap):
    n_rewards, _, _ = agent.compute_n_step_rewards(rewards, done)
    #A = (n_rewards-n_rewards.mean())/(n_rewards.std()+1e-5)
    A = n_rewards
    done[bootstrap] = False 
    
    done = torch.LongTensor(done.astype(int)).to(agent.device).reshape(-1)
    A = torch.tensor(A).float().to(agent.device).reshape(-1)
    log_probs = torch.stack(log_probs).to(agent.device).transpose(1,0).reshape(-1)
    
    policy_gradient = - log_probs*A
    loss = torch.mean(policy_gradient)
    return loss

def train_batched_A2C(agent, game_params, lr, n_train_processes, max_train_steps, 
                      unroll_length, max_episode_steps, test_interval=100, num_tests=5):
    
    replay_dict = dict(save_replay_episodes=num_tests,
                       replay_dir='Replays/',
                       replay_prefix='A2C')
    test_env = init_game(game_params, max_episode_steps, **replay_dict) # save just test episodes
    envs = ParallelEnv(n_train_processes, game_params, max_episode_steps)

    optimizer = torch.optim.Adam(agent.AC.parameters(), lr=lr)
    PID = gen_PID()
    print("Process ID: ", PID)
    score = []
    actor_losses = []
    
    step_idx = 0
    while step_idx < max_train_steps:
        s_lst, r_lst, done_lst, bootstrap_lst, s_trg_lst = list(), list(), list(), list(), list()
        log_probs = []
        s, a_mask = envs.reset()
        for _ in range(unroll_length):

            a, log_prob, entropy = agent.step(s, a_mask)
            # variables with gradient
            log_probs.append(log_prob)

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

        loss = compute_actor_loss(agent, log_probs, r_lst, done_lst, bootstrap_lst)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        actor_losses.append(loss.item())
        
        ### Test time ###
        if step_idx % test_interval == 0:
            avg_score, inspector = test(step_idx, agent, test_env, PID, num_tests)
            score.append(avg_score)
            # save episode for inspection and model weights at that point
            inspector.save_dict()
            torch.save(agent.AC.state_dict(), "Results/MoveToBeacon/Checkpoints/"+PID+"_"+str(step_idx))
    envs.close()
    
    return score, actor_losses, agent, PID

def test(step_idx, agent, test_env, process_ID, num_test=5):
    score = 0.0
    done = False
    
    ### Standard tests ###
    for _ in range(num_test-1):
        
        obs = test_env.reset()
        s = get_ohe_state(obs)[np.newaxis, ...] # add batch dim
        available_actions = obs[0].observation.available_actions
        if simplified:
            a_mask = select_army_mask()[np.newaxis, ...] # add batch dim
        else:
            a_mask = get_action_mask(available_actions)[np.newaxis, ...] # add batch dim
        
        while not done:
            a, log_prob, probs = agent.step(s, a_mask)
            obs = test_env.step(a)
            s_prime = get_ohe_state(obs)[np.newaxis, ...] # add batch dim
            reward = obs[0].reward
            done = obs[0].last()
            available_actions = obs[0].observation.available_actions
            if simplified:
                a_mask = move_screen_mask()[np.newaxis, ...] # add batch dim
            else:
                a_mask = get_action_mask(available_actions)[np.newaxis, ...] # add batch dim
            
            s = s_prime
            score += reward
        done = False
        
    ### Inspection test ###
    G, inspector = inspection_test(step_idx, agent, test_env, process_ID)
    score += G
    print(f"Step # : {step_idx}, avg score : {score/num_test:.1f}")
    return score/num_test, inspector

def inspection_test(step_idx, agent, test_env, process_ID):
    inspector = InspectionDict(step_idx, process_ID)
    
    obs = test_env.reset()
    s = get_ohe_state(obs)[np.newaxis, ...] # add batch dim
    
    available_actions = obs[0].observation.available_actions
    if simplified:
        a_mask = select_army_mask()[np.newaxis, ...] # add batch dim
    else:
        a_mask = get_action_mask(available_actions)[np.newaxis, ...] # add batch dim
    
    done = False
    G = 0.0
    # list used for update
    s_lst, r_lst, done_lst, bootstrap_lst, s_trg_lst = list(), list(), list(), list(), list()
    log_probs = []
    while not done:
        a, log_prob, entropy, step_dict = inspection_step(agent, s, a_mask)
        inspector.store_step(step_dict)
        log_probs.append(log_prob)

        obs = test_env.step(a)
        s_prime = get_ohe_state(obs)[np.newaxis, ...] # add batch dim

        reward = obs[0].reward
        done = obs[0].last()
        available_actions = obs[0].observation.available_actions
        if simplified:
            a_mask = move_screen_mask()[np.newaxis, ...] # add batch dim
        else:
            a_mask = get_action_mask(available_actions)[np.newaxis, ...] # add batch dim
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
    update_dict = inspection_update(agent, log_probs, r_lst, done_lst, bootstrap_lst)
    inspector.store_update(update_dict)
    return G, inspector
