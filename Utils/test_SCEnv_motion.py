import numpy as np
import torch
import matplotlib.pyplot as plt
from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib import features

# Ids of the actions that we'll use
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id

# Meaning of some arguments required by the actions
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

def init_env(res, random_seed=None):
    RESOLUTION = res
    race = sc2_env.Race(1) # 1 = terran
    agent = sc2_env.Agent(race, "Testv0") # NamedTuple [race, agent_name]

    interface_dict = dict(feature_screen=RESOLUTION, # screen resolution in pixel
                          feature_minimap=16, # minimap resolution in pixel (smaller or equal to screen)
                          action_space="FEATURES") # either FEATURES or RGB - suggested: FEATURES

    agent_interface_format = sc2_env.parse_agent_interface_format(**interface_dict) #AgentInterfaceFormat instance

    game_params = dict(map_name='MoveToBeacon', # simplest minigame
                       players=[agent], # use a list even for single player
                       agent_interface_format=[agent_interface_format], # use a list even for single player
                       game_steps_per_episode = 256*8
                       )  
    # create an envirnoment
    env = sc2_env.SC2Env(**game_params, random_seed=random_seed)
    return env
    
def select():
    return [actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])]

def move(point=None, q=False):
    if point is None:
        point = list(np.random.choice(64, 2))
    if q:
        return [actions.FunctionCall(_MOVE_SCREEN, [[1], point])]
    else:
        return [actions.FunctionCall(_MOVE_SCREEN, [[0], point])]
    
def no_op():
    return [actions.FunctionCall(_NO_OP, [])]
    
def plot_states(states, points, policy):
    for t in range(len(states)):
        if t > 0:
            print("\nLast move: "+policy[t-1])
        print("Current move: "+policy[t])
        plot_state(states, points, t)
        plt.show()

def plot_state(states, points, t):
    PLAYER_COLOR = np.array([200,10,10])
    LAST_POS_COLOR = np.array([100,10,10])
    BEACON_COLOR = np.array([10,10,100])
    BACKGROUND_COLOR = np.array([10,100,10])
    BACKGROUND_COLOR2 = np.array([0,0,0])
    CLICK_COLOR = np.array([255,255,0])
    LAST_CLICK_COLOR = np.array([255,165,0])
    
    s = states[t]
    rgb_map = np.full(s.shape[-2:]+(3,), BACKGROUND_COLOR)
    visibility_map = s[3].astype(bool)
    rgb_map[~visibility_map] = BACKGROUND_COLOR2
    
    if t>0:
        s_last = states[t-1]
        player_y, player_x = s_last[0].nonzero()
        rgb_map[player_y, player_x] = LAST_POS_COLOR
        click_y, click_x = points[t-1]
        rgb_map[click_y, click_x] = LAST_CLICK_COLOR
            
    beacon_ys, beacon_xs = s[1].nonzero()
    player_y, player_x = s[0].nonzero()
    click_y, click_x = points[t]
    
    rgb_map[beacon_ys, beacon_xs] = BEACON_COLOR
    rgb_map[player_y, player_x] = PLAYER_COLOR
    rgb_map[click_y, click_x] = CLICK_COLOR
        
    plt.imshow(rgb_map)
    plt.xticks([])
    plt.yticks([])