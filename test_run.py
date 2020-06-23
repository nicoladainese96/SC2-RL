from absl import flags
import argparse
from pysc2.env import sc2_env
import sys

parser = argparse.ArgumentParser(description='A2C for StarCraftII minigames')
# Game arguments
parser.add_argument('--res', type=int, help='Screen and minimap resolution', default=32)
parser.add_argument('--map_name', type=str, help='Name of the minigame', default='MoveToBeacon')

args, unknown_flags = parser.parse_known_args()  # Let argparse parse known flags from sys.argv.
flags.FLAGS(sys.argv[:1] + unknown_flags)  # Let absl.flags parse the rest.

def init_game(game_params, map_name='MoveToBeacon', max_steps=256, step_multiplier=8, **kwargs):

    race = sc2_env.Race(1) # 1 = terran
    agent = sc2_env.Agent(race, "_") # NamedTuple [race, agent_name]
    agent_interface_format = sc2_env.parse_agent_interface_format(**game_params) #AgentInterfaceFormat instance

    game_params = dict(map_name=map_name,
                       players=[agent],
                       game_steps_per_episode = max_steps*step_multiplier,
                       agent_interface_format=[agent_interface_format]
                       )  
    env = sc2_env.SC2Env(**game_params, **kwargs)

    return env

def main():
    # Environment parameters
    RESOLUTION = args.res
    game_params = dict(feature_screen=RESOLUTION, feature_minimap=RESOLUTION, action_space="FEATURES") 
    game_names = ['MoveToBeacon','CollectMineralShards','DefeatRoaches','FindAndDefeatZerglings',
                  'DefeatZerglingsAndBanelings','CollectMineralsAndGas','BuildMarines']
    map_name = args.map_name
    if map_name not in game_names:
        raise Exception("map name "+map_name+" not recognized.")
    env = init_game(game_params, map_name) 
    print("Game started")
    env.close()
    return

if __name__=="__main__":
    main()
