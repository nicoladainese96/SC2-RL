import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import math
import torch
from SC_Utils.render import plot_screen, plot_minimap, plot_screen_layers_grid, plot_minimap_layers_grid

### Scalar variable plots ###
def plot_rewards(rewards):
    r = np.array(rewards).astype(bool)
    spikes = np.arange(len(r))[r]
    flag = True
    for s in spikes:
        if flag:
            plt.axvline(s, alpha=0.7, c='g', label='rewards')
            flag = False
        else:
            plt.axvline(s, alpha=0.7, c='g')
            
def plot_V(d, t_min=0, t_max=-1, show_rewards=False):
    fig = plt.figure(figsize=(14,6))
    
    grid = (2,4)
    ax1 = plt.subplot2grid(grid, (0, 0), colspan=2, rowspan=2)
    timesteps = np.arange(len(d['values'][t_min:t_max]))
    ax1.plot(timesteps, d['values'][t_min:t_max], label='critic prediciton')
    ax1.plot(timesteps, d['trg_values'][t_min:t_max], label='critic target')
    if show_rewards:
        plot_rewards(d['rewards'][t_min:t_max])
    ax1.legend(fontsize=13)
    ax1.set_xlabel('Timesteps', fontsize=16)
    ax1.set_ylabel('State value', fontsize=16)
    
    v_min = min([np.min(d['values'][t_min:t_max]), np.min(d['trg_values'][t_min:t_max])])
    v_max = max([np.max(d['values'][t_min:t_max]), np.max(d['trg_values'][t_min:t_max])])
    
    ax2 = plt.subplot2grid(grid, (0, 2), colspan=2, rowspan=1)
    n, bins, _ = ax2.hist(d['values'][t_min:t_max], bins=50, range=(v_min, v_max))
    ax2.set_xlabel('State value', fontsize=13)
    ax2.set_ylabel('Occurrencies', fontsize=13)
    ax2.set_title("Critic's value distribution", fontsize=13)
              
    ax3 = plt.subplot2grid(grid, (1, 2), colspan=2, rowspan=1)
    n1, bins, _ = ax3.hist(d['trg_values'][t_min:t_max], bins=50, range=(v_min, v_max))
    ax3.set_xlabel('State value', fontsize=13)
    ax3.set_ylabel('Occurrencies', fontsize=13)
    ax3.set_title("Target's value distribution", fontsize=13)
    
    plt.tight_layout()
    plt.show()
    
def plot_critic_loss(d, t_min=0, t_max=-1, show_rewards=False):
    plt.figure(figsize=(14,6))
    
    plt.subplot(121)
    timesteps = np.arange(len(d['values'][t_min:t_max]))
    plt.plot(timesteps, d['critic_losses'][t_min:t_max])
    if show_rewards:
        plot_rewards(d['rewards'][t_min:t_max])
    plt.xlabel('Timesteps', fontsize=16)
    plt.ylabel('Critic loss', fontsize=16)
    
    plt.subplot(122)
    plt.hist(d['critic_losses'][t_min:t_max], bins=50)
    plt.xlabel('Loss values', fontsize=16)
    plt.ylabel('Occurrencies', fontsize=16)
    
    plt.tight_layout()
    plt.show()
    
def plot_actor_loss(d, t_min=0, t_max=-1, show_rewards=False):
    fig = plt.figure(figsize=(14,6))
    
    grid = (2,4)
    ax1 = plt.subplot2grid(grid, (0, 0), colspan=2, rowspan=2)
    timesteps = np.arange(len(d['advantages'][t_min:t_max]))
    ax1.plot(timesteps, d['advantages'][t_min:t_max], label='estimated advantages')
    ax1.plot(timesteps, d['actor_losses'][t_min:t_max], label='actor losses')
    if show_rewards:
        plot_rewards(d['rewards'][t_min:t_max])
    ax1.legend(fontsize=13)
    ax1.set_xlabel('Timesteps', fontsize=16)
    ax1.set_ylabel('Actor advantages and losses', fontsize=16)
    
    ax2 = plt.subplot2grid(grid, (0, 2), colspan=2, rowspan=1)
    n, bins, _ = ax2.hist(d['advantages'][t_min:t_max], bins=50)
    ax2.set_xlabel('Advantages', fontsize=13)
    ax2.set_ylabel('Occurrencies', fontsize=13)
    ax2.set_title("Advantages distribution", fontsize=13)
              
    ax3 = plt.subplot2grid(grid, (1, 2), colspan=2, rowspan=1)
    n1, bins, _ = ax3.hist(d['actor_losses'][t_min:t_max], bins=50)
    ax3.set_xlabel('Actor losses', fontsize=13)
    ax3.set_ylabel('Occurrencies', fontsize=13)
    ax3.set_title("Actor losses distribution", fontsize=13)
    
    plt.tight_layout()
    
    plt.show()
    
def plot_update_curves(d, t_min=0, t_max=-1):
    plot_V(d, t_min, t_max)
    plot_critic_loss(d, t_min, t_max)
    plot_actor_loss(d, t_min, t_max)
    
def print_action_info(inspector, d, t):
<<<<<<< HEAD
    print("\nStep %d - top 5 actions"%t)
=======
    print("\nStep %d"%t)
>>>>>>> a0bc346a83e651c0cbb872d50a7442c0b12fdf33
    for i in inspector.action_dict.keys():
        print("Action "+inspector.action_dict[i].name, '- prob: %.3f'%d['action_distr'][t][0,i])
    print("-"*35)
    print("Action chosen: ", inspector.action_dict[d['action_sel'][t][0]].name)
    
def plot_state(inspector, d, layer_names, t):
    current_state = d['state_traj'][t][0]
    click = None
    # get all arguments' names that the action selected requires
    a = d['action_sel'][t][0]
    a_name = inspector.action_dict[a].name
    #print("action name: ", a_name)
    arg_names = [arg_name for arg_name in list(d.keys()) if a_name in arg_name]
    #print("arguments names: ", arg_names)
    # check if any of those arguments is screen_distr (at most 1)
    screen_args = [arg_name for arg_name in arg_names if "screen_distr" in arg_name]
    #print("screen_args ", screen_args)
    if len(screen_args)==1:
        args = d['args'][t][0]
        # this assumes that screen is the first spatial argument, true but risky
        for arg in args:
            if len(arg) > 1:
                click = arg
                break

    last_click = None
    # FIXME
    #if t > 0:
    #    args = d['args'][t-1][0]
    #    for arg in args:
    #        if len(arg) > 1:
    #            last_click = arg
    #            break
    #print("Click: ", click)
    #print("Last click: ", last_click)
    # FIXME
    #if t > 0:
    #    last_state = d['state_traj'][t-1][0]
    #    plot_screen(current_state, layer_names, last_state, click, last_click)
    #else:
    plot_screen(current_state, layer_names, click=click)
    
def plot_minimap_state(inspector, d, layer_names, t):
    
    current_state = d['state_traj'][t][0]
    click = None
    # get all arguments' names that the action selected requires
    a = d['action_sel'][t][0]
    a_name = inspector.action_dict[a].name
    #print("action name: ", a_name)
    arg_names = [arg_name for arg_name in list(d.keys()) if a_name in arg_name]
    #print("arguments names: ", arg_names)
    # check if any of those arguments is minimap_distr (at most 1)
    minimap_args = [arg_name for arg_name in arg_names if "minimap_distr" in arg_name]
    #print("minimap_args ", minimap_args)
    if len(minimap_args)==1:
        args = d['args'][t][0]
        # this assumes that minimap is the first spatial argument, true but risky
        for arg in args:
            if len(arg) > 1:
                click = arg
                break

        
    last_click = None
    # FIXME
    #if t > 0:
    #    args = d['args'][t-1][0]
    #    for arg in args:
    #        if len(arg) > 1:
    #            last_click = arg
    #            break
    #print("Click: ", click)
    #print("Last click: ", last_click)
    # FIXME
    #if t > 0:
    #    last_state = d['state_traj'][t-1][0]
    #    plot_screen(current_state, layer_names, last_state, click, last_click)
    #else:
    plot_minimap(current_state, layer_names, click=click)
        
def plot_screen_distr(d, t, alpha=0.7):
    spatial_args = get_spatial_args(d)
    num_spatial_args = len(spatial_args)
    grid = (2, math.ceil(num_spatial_args/2))
    gs1 = gridspec.GridSpec(*grid)
    #gs1.update(wspace=0, hspace=0)

    for i in range(num_spatial_args):
        ax1 = plt.subplot(gs1[i])
        ax1.set_title(spatial_args[i])
        probs = np.exp(d[spatial_args[i]][t])
        _plot_screen_distr(probs)
        plt.axis('on')

    plt.tight_layout()
    
def _plot_screen_distr(probs, alpha=0.7):
    M = probs.max()
    m = probs.min()
    norm_probs = (probs-m)/(M-m)
    color_probs = plt.cm.plasma(norm_probs)
    #print(color_probs.min(), color_probs.max())
    #mask = np.ones(beacon_layer.shape)*(alpha-1e-3) + beacon_layer*(1-alpha)
    #color_probs[:,:,3] = mask
    plt.imshow(color_probs)
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='plasma'), cax=cax, ticks=[0, 0.5, 1])
    pmin = m
    pmax = M
    pmean = 0.5*(M-m)+m
    cbar.ax.set_yticklabels(["{:0.4f}".format(pmin), "{:0.4f}".format(pmean), "{:0.4f}".format(pmax)])
    
def plot_screen_and_decision(inspector, d, layer_names, t, show_minimap=True):
    if show_minimap:
        fig = plt.figure(figsize=(14,6))
        plt.subplot(121)
        plot_state(inspector, d, layer_names, t)
        plt.subplot(122)
        plot_minimap_state(inspector, d, layer_names, t)
    else:
        fig = plt.figure(figsize=(8,6))
        plot_state(inspector, d, layer_names, t)
    plt.show()
    
    fig = plt.figure(figsize=(14,6))
    plot_screen_distr(d, t)
    plt.show()
        
def get_spatial_args(insp_dict):
    spatial_names = ['screen_distr','minimap_distr','screen2_distr']
    spatial_args = []
    for k in insp_dict.keys():
        T = False
        for s in spatial_names:
            if s in k:
                T = True
        if T:
            spatial_args.append(k)
    return spatial_args

def plot_screen_layers(d, layer_names, t, n_cols=5):
    current_state = d['state_traj'][t][0]
    n_screen_layers = len(layer_names['screen_names'])
    plt.figure(figsize=(3*n_cols, 3*math.ceil(n_screen_layers/n_cols)))
    plot_screen_layers_grid(current_state, layer_names)
    plt.show()
    
def plot_minimap_layers(d, layer_names, t, n_cols=5):
    current_state = d['state_traj'][t][0]
    n_minimap_layers = len(layer_names['minimap_names'])
    plt.figure(figsize=(3*n_cols, 3*math.ceil(n_minimap_layers/n_cols)))
    plot_minimap_layers_grid(current_state, layer_names)
    plt.show()
    
    