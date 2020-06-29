from Utils import utils
from SC_Utils.game_utils import ObsProcesser, get_action_dict
from SC_Utils.train_v2 import *
from AC_modules.BatchedA2C import SpatialA2C, SpatialA2C_v1, SpatialA2C_v2, SpatialA2C_v3
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
parser.add_argument('--action_names', '-a_n', type=str, nargs='*', help='List of strings containing action names to use.', 
                    default=['select_army', 'Attack_screen', 'Move_screen', 'select_point', 'select_rect',
                'move_camera','Stop_quick','Move_minimap','Attack_minimap','HoldPosition_quick'])
# Agent arguments
parser.add_argument('--n_channels', type=int, help='Number of channels of the spatial features', default=32)
parser.add_argument('--n_features', type=int, help='Number of features of the non-spatial features', default=256)
parser.add_argument('--n_steps', type=int, help='Number of steps used in the TD update', default=20)
parser.add_argument('--A2C_version', type=int, help='Version of the A2C to use', default=2)
parser.add_argument('--embed_dim', type=int, help='Embedding dimension for actions', default=8)
# Training arguments
parser.add_argument('--lr', type=float, help='Learning rate', default=7e-4)
parser.add_argument('--traj_length', type=int, help='Number of steps taken in every environment before an optimizer step', default=60)
parser.add_argument('--n_train_processes', type=int, help='Number of parallel environments', default=3) # num of CPU cores - 1
parser.add_argument('--max_train_steps', type=int, help='Number of env steps used for the training', default=120000)
parser.add_argument('--test_interval', type=int, help='Number of steps after which a test episode is executed', 
                    default=60*100) # express this as a multiple of traj_length
parser.add_argument('--inspection_interval', type=int, help='Number of steps after which an in-depth inspection is executed',
                    default=60000)
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
    op = ObsProcesser(**obs_proc_params)
    screen_channels, minimap_channels = op.get_n_channels()
    in_channels = screen_channels + minimap_channels 
    action_dict = get_action_dict(args.action_names)
    print(action_dict)
    action_space = len(action_dict)
    
    # A2C params
    spatial_model = net.FullyConvSpatial
    nonspatial_model = net.FullyConvNonSpatial
    embed_dim = args.embed_dim
    spatial_dict = {"in_channels":in_channels}
    nonspatial_dict = {'resolution':RESOLUTION, 'kernel_size':3, 'stride':2}
    HPs = dict(action_space=action_space, n_steps=args.n_steps, H=1e-2, 
           spatial_model=spatial_model, nonspatial_model=nonspatial_model,
           n_features=args.n_features, n_channels=args.n_channels, 
           spatial_dict=spatial_dict, nonspatial_dict=nonspatial_dict, 
           action_dict=action_dict)

    if torch.cuda.is_available():
        HPs['device'] = 'cuda'
    else:
        HPs['device'] = 'cpu'

    print("Using device "+HPs['device'])
    version = args.A2C_version
    if version == 1:
        HPs = {**HPs, 'embed_dim':embed_dim}
        agent = SpatialA2C_v1(env=env, **HPs)
    elif version == 2:
        # no action embedding
        agent = SpatialA2C_v2(env=env, **HPs)
    elif version == 3:
        agent = SpatialA2C_v3(env=env, **HPs)
    else:
        raise Exception("Version not implemented.")
        
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
                            obs_proc_params=obs_proc_params, action_dict=action_dict,
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
