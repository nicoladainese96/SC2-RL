import matplotlib.pyplot as plt
import numpy as np

def plot_screen(current_state, names, last_state=None):
    PLAYER_COLOR = np.array([255,255,0]) # yellow
    LAST_POS_COLOR = np.array([255,165,0]) # orange
    NEUTRAL_COLOR = np.array([10,10,100]) # blue
    BACKGROUND_COLOR = np.array([10,100,10]) # green
    BACKGROUND_COLOR2 = np.array([0,0,0]) # black
    ENEMY_COLOR = np.array([200,10,10]) # red
    #CLICK_COLOR = np.array([255,255,0])
    #LAST_CLICK_COLOR = np.array([255,165,0])

    s = current_state['screen_layers']
    screen_names = names['screen_names']
    rgb_map = np.full(s.shape[-2:]+(3,), BACKGROUND_COLOR)
    if 'visibility_map_2' in screen_names:
        vm_idx = np.where(screen_names == 'visibility_map_2')[0] # only first element of the tuple for 1d arrays
        vm = s[vm_idx].astype(bool).reshape(s.shape[-2:])
        rgb_map[~vm] = BACKGROUND_COLOR2
    
    friendly_idx = np.where(screen_names == 'player_relative_1')[0]
    neutral_idx = np.where(screen_names == 'player_relative_3')[0]
    enemy_idx = np.where(screen_names == 'player_relative_4')[0]
    
    ### Plot old position if possible ###
    if last_state is not None:
        ls = last_state['screen_layers']
        player_ys, player_xs = ls[friendly_idx].reshape(s.shape[-2:]).nonzero()
        rgb_map[player_ys, player_xs] = LAST_POS_COLOR
       
    player_ys, player_xs = s[friendly_idx].reshape(s.shape[-2:]).nonzero()
    neutral_ys, neutral_xs = s[neutral_idx].reshape(s.shape[-2:]).nonzero() 
    enemy_ys, enemy_xs = s[enemy_idx].reshape(s.shape[-2:]).nonzero()
    
    rgb_map[neutral_ys, neutral_xs] = NEUTRAL_COLOR
    rgb_map[player_ys, player_xs] = PLAYER_COLOR
    rgb_map[enemy_ys, enemy_xs] = ENEMY_COLOR
        
    plt.imshow(rgb_map)
    plt.xticks([])
    plt.yticks([])