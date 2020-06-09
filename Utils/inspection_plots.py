import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch

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
            
def plot_V(d, t_min=0, t_max=-1):
    fig = plt.figure(figsize=(14,6))
    
    grid = (2,4)
    ax1 = plt.subplot2grid(grid, (0, 0), colspan=2, rowspan=2)
    timesteps = np.arange(len(d['values'][t_min:t_max]))
    ax1.plot(timesteps, d['values'][t_min:t_max], label='critic prediciton')
    ax1.plot(timesteps, d['trg_values'][t_min:t_max], label='critic target')
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
    
def plot_critic_loss(d, t_min=0, t_max=-1):
    plt.figure(figsize=(14,6))
    
    plt.subplot(121)
    timesteps = np.arange(len(d['values'][t_min:t_max]))
    plt.plot(timesteps, d['critic_losses'][t_min:t_max])
    plot_rewards(d['rewards'][t_min:t_max])
    plt.xlabel('Timesteps', fontsize=16)
    plt.ylabel('Critic loss', fontsize=16)
    
    plt.subplot(122)
    plt.hist(d['critic_losses'][t_min:t_max], bins=50)
    plt.xlabel('Loss values', fontsize=16)
    plt.ylabel('Occurrencies', fontsize=16)
    
    plt.tight_layout()
    plt.show()
    
def plot_actor_loss(d, t_min=0, t_max=-1):
    fig = plt.figure(figsize=(14,6))
    
    grid = (2,4)
    ax1 = plt.subplot2grid(grid, (0, 0), colspan=2, rowspan=2)
    timesteps = np.arange(len(d['advantages'][t_min:t_max]))
    ax1.plot(timesteps, d['advantages'][t_min:t_max], label='estimated advantages')
    ax1.plot(timesteps, d['actor_losses'][t_min:t_max], label='actor losses')
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
    
### Trajectory visualization ###

def plot_state_old(d, t, mask_queue=False):
    PLAYER_COLOR = np.array([200,10,10])
    LAST_POS_COLOR = np.array([100,10,10])
    BEACON_COLOR = np.array([10,10,100])
    BACKGROUND_COLOR = np.array([10,100,10])
    BACKGROUND_COLOR2 = np.array([0,0,0])
    CLICK_COLOR = np.array([255,255,0])
    LAST_CLICK_COLOR = np.array([255,165,0])
    
    action = d['action_sel'][t][0]
    s = d['state_traj'][t][0]
    rgb_map = np.full(s.shape[-2:]+(3,), BACKGROUND_COLOR)
    visibility_map = s[3].astype(bool)
    rgb_map[~visibility_map] = BACKGROUND_COLOR2
    
    if t>0:
        s_last = d['state_traj'][t-1][0]
        player_y, player_x = s_last[0].nonzero()
        rgb_map[player_y, player_x] = LAST_POS_COLOR
        click_x, click_y = d['spatial_sel'][t-1]
        rgb_map[click_y, click_x] = LAST_CLICK_COLOR
        
    beacon_ys, beacon_xs = s[1].nonzero()
    player_y, player_x = s[0].nonzero()
    point_x, point_y = d['spatial_sel'][t]
    
    rgb_map[beacon_ys, beacon_xs] = BEACON_COLOR
    rgb_map[player_y, player_x] = PLAYER_COLOR
    rgb_map[point_y, point_x] = CLICK_COLOR
        
    plt.imshow(rgb_map)
    plt.xticks([])
    plt.yticks([])
    
def plot_state(d, t, mask_queue=False):
    PLAYER_COLOR = np.array([200,10,10])
    LAST_POS_COLOR = np.array([100,10,10])
    BEACON_COLOR = np.array([10,10,100])
    BACKGROUND_COLOR = np.array([10,100,10])
    BACKGROUND_COLOR2 = np.array([0,0,0])
    CLICK_COLOR = np.array([255,255,0])
    LAST_CLICK_COLOR = np.array([255,165,0])
    
    action = d['action_sel'][t][0]
    s = d['state_traj'][t][0]
    rgb_map = np.full(s.shape[-2:]+(3,), BACKGROUND_COLOR)
    #visibility_map = s[3].astype(bool)
    #rgb_map[~visibility_map] = BACKGROUND_COLOR2
    
    if t>0:
        s_last = d['state_traj'][t-1][0]
        last_action = d['action_sel'][t-1][0]
        player_y, player_x = s_last[0].nonzero()
        rgb_map[player_y, player_x] = LAST_POS_COLOR
        click_x, click_y = d['spatial_sel'][t-1]
        if last_action==[2]:
            rgb_map[click_y, click_x] = LAST_CLICK_COLOR
        
    beacon_ys, beacon_xs = s[1].nonzero()
    player_y, player_x = s[0].nonzero()
    point_x, point_y = d['spatial_sel'][t]
    
    rgb_map[beacon_ys, beacon_xs] = BEACON_COLOR
    rgb_map[player_y, player_x] = PLAYER_COLOR
    if action == [2]:
        rgb_map[point_y, point_x] = CLICK_COLOR
        
    plt.imshow(rgb_map)
    plt.xticks([])
    plt.yticks([])
    
def plot_screen_distr(d, t, alpha=0.7):
    probs = d['spatial_distr'][t]
    beacon_layer = d['state_traj'][t][0][1]
    _plot_screen_distr(probs, beacon_layer, alpha)
    
def _plot_screen_distr(probs, beacon_layer, alpha=0.7):
    M = probs.max()
    m = probs.min()
    norm_probs = (probs-m)/(M-m)
    color_probs = plt.cm.plasma(norm_probs)
    #print(color_probs.min(), color_probs.max())
    mask = np.ones(beacon_layer.shape)*(alpha-1e-3) + beacon_layer*(1-alpha)
    color_probs[:,:,3] = mask
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
    
def plot_screen_and_decision(d, t, mask_queue=False):
    fig = plt.figure(figsize=(14,6))
    
    plt.subplot(121)
    plot_state(d, t, mask_queue)
    
    plt.subplot(122)
    plot_screen_distr(d, t)
    plt.show()

### Synthetic state ###

def plot_synt_state(s):
    PLAYER_COLOR = np.array([200,10,10])
    BEACON_COLOR = np.array([10,10,100])
    BACKGROUND_COLOR = np.array([10,100,10])
    rgb_map = np.full(s.shape[-2:]+(3,), BACKGROUND_COLOR)
    beacon_ys, beacon_xs = s[1].nonzero()
    player_y, player_x = s[0].nonzero()
    rgb_map[beacon_ys, beacon_xs] = BEACON_COLOR
    rgb_map[player_y, player_x] = PLAYER_COLOR
    plt.imshow(rgb_map)
    plt.xticks([])
    plt.yticks([])
    
def gen_synt_state(agent_pos, beacon_pos, n_channels, res, selected=True):
    if n_channels==6 and res==16:
        return gen_6channels_16res_state(agent_pos, beacon_pos, selected)
    elif n_channels==6 and res==32:
        return gen_6channels_32res_state(agent_pos, beacon_pos, selected)
    elif n_channels==2 and res==16:
        return gen_2channels_16res_state(agent_pos, beacon_pos)
    elif n_channels==2 and res==32:
        return gen_2channels_32res_state(agent_pos, beacon_pos)
    elif n_channels==3 and res==16:
        raise Exception("Resolution of 16 not implemented yet for 3 channels")
    elif n_channels==3 and res==32:
        return gen_3channels_32res_state(agent_pos, beacon_pos, selected)
    else:
        raise Exception("Either the resolution of the number of channels are not supported")
    
def gen_3channels_32res_state(agent_pos, beacon_pos, selected=True):
    res = 32
    state = np.zeros((3,res,res))

    for x in range(beacon_pos[0]-2, beacon_pos[0]+3):
        for y in range(beacon_pos[1]-2, beacon_pos[1]+3):
            cond1 = (x == beacon_pos[0]-2) or (x == beacon_pos[0]+2)
            cond2 = (y == beacon_pos[1]-2) or (y == beacon_pos[1]+2)
            if cond1 and cond2:
                pass
            else:
                state[1, y, x] = 1


    if state[1, agent_pos[1], agent_pos[0]] != 1:
        state[0, agent_pos[1], agent_pos[0]] = 1
        if selected:
            state[2, agent_pos[1], agent_pos[0]] = 1
        
    state = state.astype(int)
    return state

def gen_2channels_32res_state(agent_pos, beacon_pos):
    res = 32
    state = np.zeros((2,res,res))

    for x in range(beacon_pos[0]-2, beacon_pos[0]+3):
        for y in range(beacon_pos[1]-2, beacon_pos[1]+3):
            cond1 = (x == beacon_pos[0]-2) or (x == beacon_pos[0]+2)
            cond2 = (y == beacon_pos[1]-2) or (y == beacon_pos[1]+2)
            if cond1 and cond2:
                pass
            else:
                state[1, y, x] = 1
                
    if state[1, agent_pos[1], agent_pos[0]] != 1:
        state[0, agent_pos[1], agent_pos[0]] = 1
        
    state = state.astype(int)
    return state

def gen_2channels_16res_state(agent_pos, beacon_pos):
    res = 16
    state = np.zeros((2,res,res))
    for x in range(beacon_pos[0]-1, beacon_pos[0]+2):
        for y in range(beacon_pos[1]-1, beacon_pos[1]+2):
            state[1, y, x] = 1
        
    if state[1, agent_pos[1], agent_pos[0]] != 1:
        state[0, agent_pos[1], agent_pos[0]] = 1
        
    state = state.astype(int)
    return state

def gen_6channels_16res_state(agent_pos, beacon_pos, selected=True):
    """
    Layers:
    0. player pos
    1. beacon pos
    2. agent selected
    3. visibility map
    4. unit density
    5. unit density anti-aliasing
    """
    
    res = 16
    state = np.zeros((6,res,res)).astype(float)
    for x in range(beacon_pos[0]-1, beacon_pos[0]+2):
        for y in range(beacon_pos[1]-1, beacon_pos[1]+2):
            state[1, y, x] = 1
        
    if state[1, agent_pos[1], agent_pos[0]] != 1:
        state[0, agent_pos[1], agent_pos[0]] = 1
        if selected:
            state[2, agent_pos[1], agent_pos[0]] = 1
                           
    # Visibility map
    for x in range(1,res-1):
        for y in range(1,11):
            state[3,y,x] = 2
    
    # Unit density = sum of beacon and player layers
    state[4] = state[0] + state[1]
    
    #Unit density aa -> crude approximation
    state[5, agent_pos[1], agent_pos[0]] += 4 # agent density all in one cell
    for x in range(beacon_pos[0]-1, beacon_pos[0]+2):
        for y in range(beacon_pos[1]-1, beacon_pos[1]+2):
            cond1 = (x == beacon_pos[0]-1) or (x == beacon_pos[0]+1)
            cond2 = (y == beacon_pos[1]-1) or (y == beacon_pos[1]+1)
            if (x == beacon_pos[0]) and (y == beacon_pos[1]):
                state[5, y, x] += 16
            elif cond1 and cond2:
                state[5, y, x] += 3
            else:
                state[5, y, x] += 12
    return state              

def gen_6channels_32res_state(agent_pos, beacon_pos, selected=True):
    """
    Layers:
    0. player pos
    1. beacon pos
    2. agent selected
    3. visibility map
    4. unit density
    5. unit density anti-aliasing
    """
    
    res = 32
    state = np.zeros((6,res,res)).astype(float)
    for x in range(beacon_pos[0]-1, beacon_pos[0]+2):
        for y in range(beacon_pos[1]-1, beacon_pos[1]+2):
            state[1, y, x] = 1
        
    if state[1, agent_pos[1], agent_pos[0]] != 1:
        state[0, agent_pos[1], agent_pos[0]] = 1
        if selected:
            state[2, agent_pos[1], agent_pos[0]] = 1
                           
    # Visibility map
    for x in range(1,res-1):
        for y in range(1,23):
            state[3,y,x] = 2
    
    # Unit density = sum of beacon and player layers
    state[4] = state[0] + state[1]
    
    #Unit density aa -> crude approximation
    state[5, agent_pos[1], agent_pos[0]] += 16 # agent density all in one cell
    beacon_d_aa = np.array([[0,13,16,13,0],
                            [13,16,16,16,13],
                            [16,16,16,16,16],
                            [13,16,16,16,13],
                            [0,13,16,13,0]])
    for x in range(beacon_pos[0]-2, beacon_pos[0]+3):
        for y in range(beacon_pos[1]-2, beacon_pos[1]+3):
            x_offset = x - (beacon_pos[0]-2)
            y_offset = y - (beacon_pos[1]-2)
            state[5, y, x] += beacon_d_aa[y_offset, x_offset]
    return state           

def compute_value_map(agent, beacon_pos, n_channels, res):
    v_map = np.zeros((res,res))
    for x in range(res):
        for y in range(res):
            s = gen_synt_state([x,y], beacon_pos, n_channels, res)
            s = torch.from_numpy(s).float().to(agent.device).unsqueeze(0)
            with torch.no_grad():
                V = agent.AC.V_critic(s).squeeze()
            v_map[y,x] = V.cpu().numpy()
    return v_map

def compute_value_map_6channels_16res(agent, beacon_pos):
    res = 16
    v_map = np.zeros((res,res))
    for x in range(res):
        for y in range(res):
            s = gen_6channels_16res_state([x,y], beacon_pos)
            s = torch.from_numpy(s).float().to(agent.device).unsqueeze(0)
            with torch.no_grad():
                V = agent.AC.V_critic(s).squeeze()
            v_map[y,x] = V.cpu().numpy()
    return v_map

def plot_value_map(agent, beacon_pos, n_channels, res):
    v_map = compute_value_map(agent, beacon_pos, n_channels, res)
    
    fig = plt.figure(figsize=(14,6))
    
    plt.subplot(121)
    plt.title("Beacon position", fontsize = 16)
    plot_synt_state(gen_synt_state(beacon_pos, beacon_pos, n_channels, res))
    
    plt.subplot(122)
    plt.imshow(v_map, cmap='plasma')
    plt.title("Value map", fontsize = 16)
    plt.xticks([])
    plt.yticks([])
    
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax)
    return
 
def plot_value_map_at_step(agent, step_idx, PID, beacon_pos, n_channels, res):
    agent.AC.load_state_dict(torch.load("Results/MoveToBeacon/Checkpoints/"+PID+"_"+str(step_idx)))
    agent.AC.to(agent.device) 
    v_map = compute_value_map(agent, beacon_pos, n_channels, res)
    
    fig = plt.figure(figsize=(14,6))
    
    plt.subplot(121)
    plt.imshow(v_map, cmap='plasma')
    plt.title("Value map - step %d"%step_idx, fontsize = 16)
    plt.xticks([])
    plt.yticks([])
    
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax)
    
    plt.subplot(122)
    plt.hist(v_map.flatten(), bins=50)
    plt.xlabel("State values", fontsize = 16)
    plt.ylabel("Occurrencies", fontsize = 16)
    plt.title("State values distribution", fontsize = 16)
    
    plt.tight_layout()
    plt.show()
    return
    
def plot_value_maps(agent, PID, init_step, step_jump, n_jumps, beacon_pos, n_channels, res):
    fig = plt.figure(figsize=(8,6))
    
    plt.title("Beacon position", fontsize = 16)
    state = gen_synt_state(beacon_pos, beacon_pos, n_channels, res)
    plot_synt_state(state)
    
    for n in range(1,n_jumps+1):
        #fig = plt.figure(figsize=(8,6))
        
        step_idx = init_step + step_jump*n
        plot_value_map_at_step(agent, step_idx, PID, beacon_pos, n_channels, res)
    return
        
def plot_decision_maps(agent, PID, init_step, step_jump, n_jumps, agent_pos_lst=[[2,2],[11,11],[16,16]], 
                       beacon_pos = [16,16], selected=True, n_channels=6, res=16):
    
    fig = plt.figure(figsize=(14,6))
    
    states = []
    N = len(agent_pos_lst)
    for i in range(N):
        pos = agent_pos_lst[i]
        plt.subplot(1,3,i+1)
        if i == 1:
            plt.title("States considered", fontsize=16)
        s = gen_synt_state(pos, beacon_pos, n_channels, res)
        plot_synt_state(s)
        states.append(s)
    plt.show()
    
    state = torch.from_numpy(np.array(states)).float().to(agent.device)
    
    for n in range(1,n_jumps+1):
        fig = plt.figure(figsize=(14,6))
        
        step_idx = init_step + step_jump*n
        plot_map_at_step(agent, step_idx, PID, state, n_channels, res)
    
def plot_map_at_step(agent, step_idx, PID, state, n_channels, res):
    
    agent.AC.load_state_dict(torch.load("Results/MoveToBeacon/Checkpoints/"+PID+"_"+str(step_idx)))
    agent.AC.to(agent.device) 
    
    with torch.no_grad():
        #print('state.shape: ', state.shape)
        spatial_features = agent.AC.spatial_features_net(state)
        #print("spatial_features.shape: ", spatial_features.shape)
        screen_arg, screen_log_prob, screen_distr = agent.AC.sample_param(spatial_features, 'screen')
        #screen_arg, screen_log_prob, screen_distr = agent.AC.sample_param(state, 'screen')
    N = screen_distr.shape[0]
    screen_distr = screen_distr.cpu().numpy().reshape(N, res, res)
    M = screen_distr.max()
    m = screen_distr.min()
    
    for i in range(N):
        plt.subplot(1,3,i+1)
        if i == 1:
            plt.title("Step %d"%step_idx, fontsize = 16)
        beacon_layer = state.cpu().numpy()[i][1]
        _plot_screen_distr(screen_distr[i], beacon_layer, alpha=0.7)    
    plt.show()   
    
def print_action_info(d, t):
    print("\nStep %d"%t)
    a_distr = d['action_distr'][t][0]
    select_distr = d['selectall_distr'][t][0]
    queue_distr = d['queue_distr'][t][0]
    point_sel = d['spatial_sel'][t]
    adv = d['advantages'][t]
    V = d['values'][t]
    print("_NO_OP: \t%.2f"%(a_distr[0]*100))
    print("_SELECT_ARMY:  %.2f - _SELECT_ALL: %.2f"%(a_distr[1]*100, select_distr[0]*100))
    print("_MOVE_SCREEN:   %.2f - _NOT_QUEUED: %.2f - POINT: (x,y)=(%d,%d)"%(a_distr[2]*100, queue_distr[0]*100, point_sel[0], point_sel[1]))
    print("Action selected: ", d['action_sel'][t])
    print("State value: %.4f"%V) 
    print("Move advantage: %.4f"%adv)

def print_reinf_action_info(d, t):
    print("\nStep %d"%t)
    a_distr = d['action_distr'][t][0]
    select_distr = d['selectall_distr'][t][0]
    queue_distr = d['queue_distr'][t][0]
    point_sel = d['spatial_sel'][t]
    adv = d['advantages'][t]
    print("_NO_OP: \t%.2f"%(a_distr[0]*100))
    print("_SELECT_ARMY:  %.2f - _SELECT_ALL: %.2f"%(a_distr[1]*100, select_distr[0]*100))
    print("_MOVE_SCREEN:   %.2f - _NOT_QUEUED: %.2f - POINT: (x,y)=(%d,%d)"%(a_distr[2]*100, queue_distr[0]*100, point_sel[0], point_sel[1]))
    print("Action selected: ", d['action_sel'][t])
    print("Move advantage: %.4f"%adv)
    
### Notebook 9 ###

def plot_trajectory_policy1(states, point=[1,1]):
    PLAYER_COLOR = np.array([200,10,10])
    LAST_POS_COLOR = np.array([100,10,10])
    BEACON_COLOR = np.array([10,10,100])
    BACKGROUND_COLOR = np.array([10,100,10])
    CLICK_COLOUR = np.array([255,255,0])
    
    for i, s in enumerate(states):
        print(s)
        rgb_map = np.full(s.shape[-2:]+(3,), BACKGROUND_COLOR)
        
        if i > 0:
            s_prev = states[i-1]
            player_y, player_x = s_prev[0].nonzero()
            rgb_map[player_y, player_x] = LAST_POS_COLOR
            
            rgb_map[point[1],point[0]] = CLICK_COLOUR
            
        beacon_ys, beacon_xs = s[1].nonzero()
        player_y, player_x = s[0].nonzero()
        
        rgb_map[beacon_ys, beacon_xs] = BEACON_COLOR
        rgb_map[player_y, player_x] = PLAYER_COLOR
        
        plt.imshow(rgb_map)
        plt.title("Step %d - policy 1"%i)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        
def plot_trajectory_policy2(states, points, last_actions, alpha = 0.5):
    PLAYER_COLOR = np.array([200,10,10])
    LAST_POS_COLOR = np.array([100,10,10])
    BEACON_COLOR = np.array([10,10,100])
    BACKGROUND_COLOR = np.array([10,100,10])
    CLICK_COLOUR = np.array([255,255,0])
    CLICK_COLOUR2 = np.array([255,0,255])
    CLICK_COLOUR3 = np.array([0,255,255])
    
    for i, s in enumerate(states):
        rgb_map = np.full(s.shape[-2:]+(3,), BACKGROUND_COLOR)
        #if i > 0:
        #    s_prev = states[i-1]
        #    player_y, player_x = s_prev[0].nonzero()
        #    rgb_map[player_y, player_x] = LAST_POS_COLOR
          
        
        rgb_map[points[0][1],points[0][0]] = CLICK_COLOUR
        rgb_map[points[1][1],points[1][0]] = CLICK_COLOUR2
        if i >= 2:
            rgb_map[points[i][1],points[i][0]] = CLICK_COLOUR3
        beacon_ys, beacon_xs = s[1].nonzero()
        player_y, player_x = s[0].nonzero()
        
        rgb_map[beacon_ys, beacon_xs] = BEACON_COLOR
        rgb_map[player_y, player_x] = PLAYER_COLOR
        
        if i > 1:
            rgb_map = alpha*rgb_map + (1-alpha)*rgb_map_old
            rgb_map = rgb_map.astype(int)
        rgb_map_old = rgb_map
        
        if i > 0:
            print("Last action: ", last_actions[i-1])
        fig = plt.figure(figsize = (6,6))
        plt.imshow(rgb_map)
        plt.title("Step %d"%i, fontsize=16)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        
def plot_trajectory_policy3(states, points):
    PLAYER_COLOR = np.array([200,10,10])
    LAST_POS_COLOR = np.array([100,10,10])
    BEACON_COLOR = np.array([10,10,100])
    BACKGROUND_COLOR = np.array([10,100,10])
    CLICK_COLOUR = np.array([255,255,0])
    CLICK_COLOUR2 = np.array([255,0,255])
    CLICK_COLOUR3 = np.array([0,255,255])
    
    for i, s in enumerate(states):
        rgb_map = np.full(s.shape[-2:]+(3,), BACKGROUND_COLOR)
        if i > 0:
            s_prev = states[i-1]
            player_y, player_x = s_prev[0].nonzero()
            rgb_map[player_y, player_x] = LAST_POS_COLOR
          
        
        rgb_map[points[0][1],points[0][0]] = CLICK_COLOUR
        if i > 0:
            rgb_map[points[1][1],points[1][0]] = CLICK_COLOUR2
        if i > 1:
            rgb_map[points[2][1],points[2][0]] = CLICK_COLOUR3
        
        beacon_ys, beacon_xs = s[1].nonzero()
        player_y, player_x = s[0].nonzero()
        
        rgb_map[beacon_ys, beacon_xs] = BEACON_COLOR
        rgb_map[player_y, player_x] = PLAYER_COLOR
        
        fig = plt.figure(figsize = (6,6))
        plt.imshow(rgb_map)
        plt.title("Step %d"%i, fontsize=16)
        plt.xticks([])
        plt.yticks([])
        plt.show()