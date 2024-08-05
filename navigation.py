import numpy as np
from chmm_actions import CHMM, forwardE, datagen_structured_obs_room
import matplotlib.pyplot as plt
import igraph
from matplotlib import cm, colors
import sys, os, pickle

retrain_models = False

custom_colors = (
    np.array(
        [
            [214, 214, 214],
            [253, 252, 144],
            [239, 142, 192],
            [140, 194, 250],
            [214, 134, 48],
            [85, 35, 157],
            [114, 245, 144],
            [151, 38, 20],
            [72, 160, 162],
        ]
    )
    / 256
)

simple_granular_room = np.array(
    [[4, 2, 4, 0],
    [3, 0, 0, 2],
    [4, 1, 3, 0],
    [3, 3, 2, 0]]
)

def input_act():
    act_map = {'w':2, 'a':0, 's':3, 'd':1}
    act_list = [0, 1, 2, 3]  # 0: left, 1: right, 2: up, 3: down
    got = False
    while not got:
        x = input()
        if x in act_map.keys():
            return act_map[x]
        try:
            x = int(x)
            if x in act_list:
                got = True
        except:
            pass
    return x

def plot_room(room, pos=None):
    room = room.copy()
    fig, ax = plt.subplots()
    text = None
    n_emissions = np.max(room) + 1
    if pos is not None:

        # c = np.zeros((n_emissions+1, 3))
        # c[:n_emissions] = custom_colors[:n_emissions]
        # cmap = colors.ListedColormap(c)
        cmap = colors.ListedColormap(custom_colors[:n_emissions])

        r, c = pos
        # room[r, c] = n_emissions

        ax.matshow(room, cmap=cmap)
        ax.set_title(f'current position: ({r},{c})')
        ax.set_xlabel('0: left, 1: right, 2: up, 3: down')
        ASCII_person = "O\n/|\\\n/ \\"
        text = ax.text(c, r, ASCII_person, va='center', ha='center', color='black')
    else:
        cmap = colors.ListedColormap(custom_colors[:n_emissions])
        ax.matshow(room, cmap=cmap)
        
    return ax, text
        
def redraw_room(ax, pos, old_text=None):

    r, c = pos
    # room[r, c] = n_emissions
    if old_text is not None:
        old_text.remove()
    ASCII_person = "O\n/|\\\n/ \\"
    text = ax.text(c, r, ASCII_person, va='center', ha='center', color='black')
    
    plt.draw()
    return text

def plot_current_obs(room, pos):
    n_emissions = np.max(room) + 1

    fig, ax = plt.subplots()
    (r,c) = pos
    e = room[r, c]

    one_square = np.zeros((1, 1))
    one_square[0, 0] = e
    cmap = colors.ListedColormap(custom_colors[e])

    axim = ax.matshow(one_square, cmap=cmap)
    ax.set_title(f'current observation')
    return axim

def redraw_current_obs(axim, room, pos):
    (r, c) = pos
    e = room[r, c]
    one_square = np.zeros((1, 1))
    one_square[0, 0] = e

    axim.set_data(one_square)
    new_cmap = colors.ListedColormap(custom_colors[e])
    axim.set_cmap(new_cmap)
    plt.draw()    
    

def navigate(room, start_pos=None, display_mode=2):
    """
    display_mode: 
        0: just current obs
        1: current obs and map of room
        2: current obs and map of room with position
    """
    H, W = room.shape
    if start_pos == None:
        r = np.random.randint(0, H)
        c = np.random.randint(0, W)
    else:
        r, c = start_pos
    plt.ion()
    if display_mode == 2:
        ax, text = plot_room(room, pos=(r, c))
    elif display_mode == 1:
        plot_room(room, pos=None)
    obs_axim = plot_current_obs(room, pos=(r,c))
    while True:
        a = input_act()
        if a == 0 and 0 < c:
            c -= 1
        elif a == 1 and c < W - 1:
            c += 1
        elif a == 2 and 0 < r:
            r -= 1
        elif a == 3 and r < H - 1:
            r += 1
        if display_mode==2:
            text = redraw_room(ax, (r, c), old_text=text)
        redraw_current_obs(obs_axim, room, pos=(r, c))
 
navigate(simple_granular_room, display_mode=2)