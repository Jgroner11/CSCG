import numpy as np
import matplotlib.pyplot as plt
import igraph
from matplotlib import cm, colors
import sys, os, pickle

from chmm_actions import CHMM, forwardE, datagen_structured_obs_room
from CSCG_helpers import Plotting

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
        fig, ax, text = Plotting.plot_room(room, pos=(r, c))
    elif display_mode == 1:
        Plotting.plot_room(room, pos=None)
    obs_axim = Plotting.plot_current_obs(room, pos=(r,c))
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
            text = Plotting.redraw_room(fig, ax, (r, c), old_text=text)
        Plotting.redraw_current_obs(obs_axim, room, pos=(r, c))
 

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

Plotting.custom_colors = custom_colors

simple_granular_room = np.array(
    [[4, 2, 4, 0],
    [3, 0, 0, 2],
    [4, 1, 3, 0],
    [3, 3, 2, 0]]
)

navigate(simple_granular_room, display_mode=1)