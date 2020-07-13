from Utils import utils
from SC_Utils.game_utils import FullObsProcesser, get_action_dict
from SC_Utils.train_v4 import *
from AC_modules.BatchedA2C import GeneralA2C
import AC_modules.Networks as net
import torch
import argparse 
import os
import sys
from absl import flags

parser = argparse.ArgumentParser(description='A2C for StarCraftII minigames')
# Game arguments
parser.add_argument('--res', type=int, help='Screen and minimap resolution', default=32)
parser.add_argument('--map_name', type=str, help='Name of the minigame', default='MoveToBeacon')
parser.add_argument('--select_all_layers', type=bool, help='If True, selects all useful layers of screen and minimap', default=True)
parser.add_argument('--screen_names', type=str, nargs='*', help='List of strings containing screen layers names to use. \
                    Overridden by select_all_layers=True', 
                    default=['visibility_map', 'player_relative', 'selected', 'unit_density', 'unit_density_aa'])
parser.add_argument('--minimap_names', type=str, nargs='*', help='List of strings containing minimap layers names to use. \
                    Overridden by select_all_layers=True', 
                    default=['visibility_map', 'camera'])
# Agent arguments
parser.add_argument('--conv_channels', type=int, help='Number of convolutional channels for screen+minimap output', default=32)
parser.add_argument('--player_features', type=int, help='Number of features for the player features output', default=16)
parser.add_argument('--n_features', type=int, help='Number of features of the non-spatial features', default=256)
parser.add_argument('--n_steps', type=int, help='Number of steps used in the TD update', default=20)
# Training arguments
parser.add_argument('--lr', type=float, help='Learning rate', default=7e-4)
parser.add_argument('--H', type=float, help='Entropy weight', default=1e-2)
parser.add_argument('--traj_length', type=int, help='Number of steps taken in every environment before an optimizer step', default=60)
parser.add_argument('--n_train_processes', type=int, help='Number of parallel environments', default=3) # num of CPU cores - 1
parser.add_argument('--max_train_steps', type=int, help='Number of env steps used for the training', default=120000)
parser.add_argument('--test_interval', type=int, help='Number of steps after which a test episode is executed', 
                    default=60*100) # express this as a multiple of traj_length
parser.add_argument('--inspection_interval', type=int, help='Number of steps after which an in-depth inspection is executed',
                    default=55000)
# Paths
parser.add_argument('--save_dir', type=str, help='Path to save directory', default='Results/')


args, unknown_flags = parser.parse_known_args()  # Let argparse parse known flags from sys.argv.
flags.FLAGS(sys.argv[:1] + unknown_flags)  # Let absl.flags parse the rest.

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
    
    # Action and state space params
    if args.select_all_layers:
        obs_proc_params = {'select_all':True}
    else:
        obs_proc_params = {'screen_names':args.screen_names, 'minimap_names':args.minimap_names}
    op = FullObsProcesser(**obs_proc_params)
    screen_channels, minimap_channels, in_player = op.get_n_channels()
    in_channels = screen_channels + minimap_channels 
    
    # A2C params
    spatial_model = net.FullyConvPlayerAndSpatial
    nonspatial_model = net.FullyConvNonSpatial
    # Internal features, passed inside a dictionary
    conv_channels = args.conv_channels #32
    player_features = args.player_features #16
    # Exposed features, passed outside of a dictionary
    n_channels = conv_channels + player_features #48
    n_features = args.n_features #256

    spatial_dict = {"in_channels":in_channels, 'in_player':in_player, 
                    'conv_channels':conv_channels, 'player_features':player_features}
    nonspatial_dict = {'resolution':RESOLUTION, 'kernel_size':3, 'stride':2, 'n_channels':n_channels}

    HPs = dict(gamma=0.99, n_steps=20, H=args.H, 
           spatial_model=spatial_model, nonspatial_model=nonspatial_model,
           n_features=n_features, n_channels=n_channels, 
           spatial_dict=spatial_dict, nonspatial_dict=nonspatial_dict)

    if torch.cuda.is_available():
        HPs['device'] = 'cuda'
    else:
        HPs['device'] = 'cpu'
    print("Using device "+HPs['device'])
    agent = GeneralA2C(env=env, **HPs)
    env.close()
    
    # Training args
    train_dict = dict(n_train_processes = args.n_train_processes,
                      max_train_steps = args.max_train_steps,
                      unroll_length = args.traj_length,
                      test_interval = args.test_interval,
                      inspection_interval = args.inspection_interval
                      )

    # Creating paths if not existing
    if not os.path.isdir(args.save_dir):
        os.system("mkdir "+args.save_dir)
    if not os.path.isdir(args.save_dir+map_name):
        os.system("mkdir "+args.save_dir+map_name)
    # Actual training
    results = train_batched_A2C(agent, game_params, map_name, args.lr, 
                            obs_proc_params=obs_proc_params, 
                            save_path=args.save_dir+map_name, **train_dict)
    score, losses, trained_agent, PID = results
    
    # Save results
    save = True
    keywords = [map_name,
                'lr-'+str(args.lr),
                str(args.n_steps)+'-steps', 
                str(args.res)+'-res',
                str(args.max_train_steps)+"-env-steps",
                str(args.traj_length)+"-unroll-len",
                str(in_channels)+'-in-channels'] 

    if save:
        save_dir = args.save_dir+map_name+"/"
        os.system('mkdir '+save_dir)
        keywords.append(PID)
        filename = '_'.join(keywords)
        filename = 'S_'+filename
        print("Save at "+save_dir+filename)
        train_session_dict = dict(game_params=game_params, HPs=HPs, score=score, n_epochs=len(score), keywords=keywords, losses=losses)
        np.save(save_dir+filename, train_session_dict)
        torch.save(trained_agent, save_dir+"agent_"+PID)
    else:
        print("Nothing saved")
        pass
    
if __name__ == "__main__":
    main()
