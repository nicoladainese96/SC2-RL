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

def play_episode(agent, env, max_steps, coord_state=True):

    # Start the episode
    obs = env.reset()
    
    if coord_state:
        state = get_coord_state(obs)
    else:
        state = get_ohe_state(obs)

    rewards = []
    log_probs = []
    distributions = []
    states = [state]
    done = []
    bootstrap = []
        
    steps = 0 # not used
    while True:
     
        action, log_prob, probs = agent.step(obs, return_log = True) 
        new_obs = env.step(actions=[action])
        
        if coord_state:
            new_state = get_coord_state(new_obs)
        else:
            new_state = get_ohe_state(new_obs)
            
        reward = new_obs[0].reward
        terminal = new_obs[0].last()
        
        rewards.append(reward)
        log_probs.append(log_prob)
        distributions.append(probs)
        states.append(new_state)
        done.append(terminal)
        
        # last step is always truncated, because it would be an infinite-horizon game
        if terminal or steps == max_steps:
            bootstrap.append(True)
        else:
            bootstrap.append(False) 
        
        if terminal is True:
            break
            
        obs = new_obs
        steps += 1 # not used
        
    rewards = np.array(rewards)
    states = np.array(states)
    if debug: print("states.shape: ", states.shape)
    done = np.array(done)
    bootstrap = np.array(bootstrap)

    return rewards, log_probs, distributions, np.array(states), done, bootstrap

def train_SC2_agent(agent, game_params, n_episodes = 1000, max_steps=120, return_agent=False, coord_state=True, **kwargs):

    performance = []
    time_profile = []
    critic_losses = [] 
    actor_losses = []
    entropies = []
    
    env = init_game(game_params, max_steps, **kwargs)
    
    for e in range(n_episodes):
        
        t0 = time.time()
        
        rewards, log_probs, distributions, states, done, bootstrap = play_episode(agent, env, max_steps, coord_state)
        performance.append(np.sum(rewards))

        t1 = time.time()
        
        critic_loss, actor_loss, entropy = agent.update(rewards, log_probs, distributions, states, done, bootstrap)
        critic_losses.append(critic_loss)
        actor_losses.append(actor_loss)
        entropies.append(entropy)
        
        t2 = time.time()
        
        if (e+1)%10 == 0:
            print("Episode %d - reward: %.2f "%(e+1, np.mean(performance[-10:])))
            print("\tTime playing the episode: %.2f s"%(t1-t0))
            print("\tTime updating the agent: %.2f s"%(t2-t1))
            
        time_profile.append([t1-t0, t2-t1])
                            
    performance = np.array(performance)
    time_profile = np.array(time_profile)
                            
    L = n_episodes // 6 # consider last sixth of episodes to compute agent's asymptotic performance
    losses = dict(critic_losses=critic_losses, actor_losses=actor_losses, entropies=entropies)
    
    if return_agent:
        return performance, performance[-L:].mean(), performance[-L:].std(), agent, time_profile, losses
    else:
        return performance, performance[-L:].mean(), performance[-L:].std(), losses
