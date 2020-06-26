import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_screen_layers_grid(state, names, cols=4):
    """
    Use merge_screen_and_minimap before calling this. 
    Keep names separated instead, i.e. names = {'screen_names':[...],'layer_names':[...]}
    """
    screen_names = names['screen_names']
    num_plots = len(screen_names)
    grid = (math.ceil(num_plots/cols), cols)
    gs1 = gridspec.GridSpec(*grid)
    s = state[:num_plots]
    for i in range(num_plots):
        ax1 = plt.subplot(gs1[i])
        ax1.set_title(screen_names[i])
        plt.imshow(s[i], cmap='Greys')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(cax=cax)
        plt.axis('on')
        
    plt.suptitle("Screen layers ", y=0.99, fontsize=20)
    plt.tight_layout()
    
def plot_minimap_layers_grid(state, names, cols=4):
    """
    Use merge_screen_and_minimap before calling this. 
    Keep names separated instead, i.e. names = {'screen_names':[...],'layer_names':[...]}
    """
    minimap_names = names['minimap_names']
    num_plots = len(minimap_names)
    num_screen = len(names['screen_names'])
    grid = (math.ceil(num_plots/cols), cols)
    gs1 = gridspec.GridSpec(*grid)
    s = state[num_screen:]
    for i in range(num_plots):
        ax1 = plt.subplot(gs1[i])
        ax1.set_title(minimap_names[i])
        plt.imshow(s[i], cmap='Greys')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(cax=cax)
        plt.axis('on')

    plt.suptitle("Minimap layers ", y=0.99, fontsize=20)
    plt.tight_layout()
    
def plot_screen(current_state, names, last_state=None, click=None, last_click=None):
    """
    Use merge_screen_and_minimap before calling this
    Keep names separated instead, i.e. names = {'screen_names':[...],'layer_names':[...]}
    """
    PLAYER_COLOR = np.array([255,255,0]) # yellow
    LAST_POS_COLOR = np.array([255,165,0]) # orange
    NEUTRAL_COLOR = np.array([10,10,100]) # blue
    BACKGROUND_COLOR = np.array([10,100,10]) # green
    BACKGROUND_COLOR2 = np.array([0,0,0]) # black
    BACKGROUND_COLOR3 = np.array([127,153,127]) # green + grey - explored but with fog
    ENEMY_COLOR = np.array([200,10,10]) # red
    CLICK_COLOR = np.array([250,250,250]) # white
    LAST_CLICK_COLOR = np.array([102,255,0]) # light green
    s = current_state
    screen_names = names['screen_names'] 
    rgb_map = np.full(s.shape[-2:]+(3,), BACKGROUND_COLOR)
    if 'visibility_map_2' in screen_names:
        vm_idx = np.where(screen_names == 'visibility_map_2')[0] # only first element of the tuple for 1d arrays
        vm = s[vm_idx].astype(bool).reshape(s.shape[-2:])
        rgb_map[~vm] = BACKGROUND_COLOR2
    if 'visibility_map_1' in screen_names:
        vm_idx = np.where(screen_names == 'visibility_map_1')[0] # only first element of the tuple for 1d arrays
        vm = s[vm_idx].astype(bool).reshape(s.shape[-2:])
        rgb_map[vm] = BACKGROUND_COLOR3
        
    friendly_idx = np.where(screen_names == 'player_relative_1')[0]
    neutral_idx = np.where(screen_names == 'player_relative_3')[0]
    enemy_idx = np.where(screen_names == 'player_relative_4')[0]
    
    ### Plot old position if possible ###
    if last_state is not None:
        ls = last_state
        player_ys, player_xs = ls[friendly_idx].reshape(s.shape[-2:]).nonzero()
        rgb_map[player_ys, player_xs] = LAST_POS_COLOR
    
        
    player_ys, player_xs = s[friendly_idx].reshape(s.shape[-2:]).nonzero()
    neutral_ys, neutral_xs = s[neutral_idx].reshape(s.shape[-2:]).nonzero() 
    enemy_ys, enemy_xs = s[enemy_idx].reshape(s.shape[-2:]).nonzero()
    
    rgb_map[neutral_ys, neutral_xs] = NEUTRAL_COLOR
    rgb_map[player_ys, player_xs] = PLAYER_COLOR
    rgb_map[enemy_ys, enemy_xs] = ENEMY_COLOR
        
    if click is not None:
        rgb_map[click[1],click[0]] = CLICK_COLOR
    if last_click is not None:
        rgb_map[last_click[1],last_click[0]] = LAST_CLICK_COLOR
        
    plt.imshow(rgb_map)
    plt.xticks([])
    plt.yticks([])
    
def plot_minimap(current_state, names, last_state=None, click=None, last_click=None):
    """
    Use merge_screen_and_minimap before calling this.
    Keep names separated instead, i.e. names = {'screen_names':[...],'layer_names':[...]}
    """
    PLAYER_COLOR = np.array([255,255,0]) # yellow
    LAST_POS_COLOR = np.array([255,165,0]) # orange
    NEUTRAL_COLOR = np.array([10,10,100]) # blue
    BACKGROUND_COLOR = np.array([10,100,10]) # green
    BACKGROUND_COLOR2 = np.array([0,0,0]) # black
    BACKGROUND_COLOR3 = np.array([127,153,127]) # green + grey - explored but with fog
    ENEMY_COLOR = np.array([200,10,10]) # red
    CLICK_COLOR = np.array([250,250,250]) # white
    LAST_CLICK_COLOR = np.array([102,255,0]) # light green
    minimap_names = names['minimap_names'] 
    minimap_start_idx = len(names['screen_names']) # offset for the state
    s = current_state[minimap_start_idx:]
    rgb_map = np.full(s.shape[-2:]+(3,), BACKGROUND_COLOR)
    if 'visibility_map_2' in minimap_names:
        vm_idx = np.where(minimap_names == 'visibility_map_2')[0] # only first element of the tuple for 1d arrays
        vm = s[vm_idx].astype(bool).reshape(s.shape[-2:])
        rgb_map[~vm] = BACKGROUND_COLOR2
    if 'visibility_map_1' in minimap_names:
        vm_idx = np.where(minimap_names == 'visibility_map_1')[0] # only first element of the tuple for 1d arrays
        vm = s[vm_idx].astype(bool).reshape(s.shape[-2:])
        rgb_map[vm] = BACKGROUND_COLOR3
        
    friendly_idx = np.where(minimap_names == 'player_relative_1')[0]
    neutral_idx = np.where(minimap_names == 'player_relative_3')[0]
    enemy_idx = np.where(minimap_names == 'player_relative_4')[0]
    ### Plot old position if possible ###
    if last_state is not None:
        ls = last_state[minimap_start_idx:]
        player_ys, player_xs = ls[friendly_idx].reshape(s.shape[-2:]).nonzero()
        rgb_map[player_ys, player_xs] = LAST_POS_COLOR
    
        
    player_ys, player_xs = s[friendly_idx].reshape(s.shape[-2:]).nonzero()
    neutral_ys, neutral_xs = s[neutral_idx].reshape(s.shape[-2:]).nonzero() 
    enemy_ys, enemy_xs = s[enemy_idx].reshape(s.shape[-2:]).nonzero()
    
    rgb_map[neutral_ys, neutral_xs] = NEUTRAL_COLOR
    rgb_map[player_ys, player_xs] = PLAYER_COLOR
    rgb_map[enemy_ys, enemy_xs] = ENEMY_COLOR
        
    if click is not None:
        rgb_map[click[1],click[0]] = CLICK_COLOR
    if last_click is not None:
        rgb_map[last_click[1],last_click[0]] = LAST_CLICK_COLOR
        
    plt.imshow(rgb_map)
    plt.xticks([])
    plt.yticks([])