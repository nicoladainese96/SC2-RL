import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import math
import torch

# see if this is needed
#from SC_Utils.render import plot_screen, plot_minimap, plot_screen_layers_grid, plot_minimap_layers_grid
from SC_Utils.inspection_plots import plot_update_curves

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
