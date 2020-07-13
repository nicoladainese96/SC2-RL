import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import math
import torch

from SC_Utils.inspection_plots import plot_update_curves, _plot_screen_distr
from SC_Utils.render import plot_screen, plot_minimap, plot_screen_layers_grid, plot_minimap_layers_grid

def plot_logging(logging, map_name):
    plt.figure(figsize=(8,6))
    plt.plot(logging[:,0], logging[:,1])
    plt.xlabel("Parallel env steps", fontsize=16)
    plt.ylabel("Reward per episode", fontsize=16)
    plt.title("Training curve of "+map_name, fontsize=16)
    plt.show()

def print_action_info(inspector, insp_dict, t):
    print("\nStep %d"%t)
    for i, a in enumerate(insp_dict['top_5_actions'][t]):
        print("Action "+inspector.all_actions[a].name, '- prob : %.3f'%insp_dict['action_distr'][t][0][a])
    print("-"*35)
    print("Action chosen: ", inspector.all_actions[insp_dict['action_sel'][t][0]].name)

def plot_screen_and_decision(inspector, d, layer_names, t):
    fig = plt.figure(figsize=(14,6))
    plt.subplot(121)
    plot_state(inspector, d, layer_names, t)
    plt.subplot(122)
    plot_minimap_state(inspector, d, layer_names, t)
    plt.show()
    
    fig = plt.figure(figsize=(14,6))
    plot_screen_distr(d, t)
    plt.show()
    
def plot_state(inspector, d, layer_names, t):
    current_state = d['state_traj'][t]['spatial'][0]
    a = d['action_sel'][t][0]
    args = d['args'][t][0]
    
    click = None
    
    arg_names = inspector.act_to_arg_names[a]
    real_arg_names = [arg.split('/')[1] for arg in arg_names]
    
    if 'screen' in real_arg_names:        
        for arg in args:
            if len(arg) == 2:
                click = arg
                break
            else:
                continue

    plot_screen(current_state, layer_names, click=click)
    
def plot_minimap_state(inspector, d, layer_names, t):
    current_state = d['state_traj'][t]['spatial'][0]
    a = d['action_sel'][t][0]
    args = d['args'][t][0]
    
    click = None
    
    arg_names = inspector.act_to_arg_names[a]
    real_arg_names = [arg.split('/')[1] for arg in arg_names]
    
    if 'minimap' in real_arg_names:        
        for arg in args:
            if len(arg) == 2:
                click = arg
                break
            else:
                continue

    plot_minimap(current_state, layer_names, click=click)
    
def plot_screen_distr(d, t, alpha=0.7):
    actions = d['top_5_actions'][t]
    spatial_distr = []
    spatial_names = []
    for a in actions:
        arg_names = list(d['top_5_action_distr'][t][a].keys())
        spatial_arg_names = [name for name in arg_names if ('screen_distr' in name) or ('minimap_distr' in name)]
        for sp_arg in spatial_arg_names:
            spatial_distr.append(d['top_5_action_distr'][t][a][sp_arg])
            spatial_names.append(sp_arg)

    num_spatial_args = len(spatial_names)
    grid = (2, math.ceil(num_spatial_args/2))
    gs1 = gridspec.GridSpec(*grid)
    
    for i in range(num_spatial_args):
        ax1 = plt.subplot(gs1[i])
        ax1.set_title(spatial_names[i])
        probs = np.exp(spatial_distr[i])
        _plot_screen_distr(probs)
        plt.axis('on')

    plt.tight_layout()
    
def plot_screen_layers(d, layer_names, t, n_cols=5):
    current_state = d['state_traj'][t]['spatial'][0]
    n_screen_layers = len(layer_names['screen_names'])
    plt.figure(figsize=(3*n_cols, 3*math.ceil(n_screen_layers/n_cols)))
    plot_screen_layers_grid(current_state, layer_names)
    plt.show()
    
def plot_minimap_layers(d, layer_names, t, n_cols=5):
    current_state = d['state_traj'][t]['spatial'][0]
    n_minimap_layers = len(layer_names['minimap_names'])
    plt.figure(figsize=(3*n_cols, 3*math.ceil(n_minimap_layers/n_cols)))
    plot_minimap_layers_grid(current_state, layer_names)
    plt.show()
    