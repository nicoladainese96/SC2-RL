import numpy as np
from pysc2.env import sc2_env
from pysc2.agents import scripted_agent

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

def play_episode(agent, env, max_steps):
    obs = env.reset()
    rewards = []
    steps = 0
    while True:
        action = agent.step(obs[0]) 
        new_obs = env.step(actions=[action])
        reward = new_obs[0].reward
        terminal = new_obs[0].last()
        rewards.append(reward)
        if terminal is True:
            break
        obs = new_obs
        steps += 1
    rewards = np.array(rewards)
    return rewards

def evaluate_scripted_agent(n_episodes = 100, max_steps=120):
    
    game_params = dict(feature_screen=16, # screen resolution in pixel
                      feature_minimap=16, # minimap resolution in pixel (smaller or equal to screen)
                      action_space="FEATURES") # either FEATURES or RGB - suggested: FEATURES

    performance = []
    env = init_game(game_params, max_steps)
    agent = scripted_agent.MoveToBeacon()
    for e in range(n_episodes):
        if (e+1)%10 == 0:
            print("Episode %d / %d"%(e+1, n_episodes))
        rewards = play_episode(agent, env, max_steps)
        performance.append(np.sum(rewards))
    performance = np.array(performance)
    return performance.mean(), performance.std(), performance