import math
import os
import pickle
import sys

import igraph
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

from agent import Navigator
from chmm_actions import CHMM, datagen_structured_obs_room, forwardE
from CSCG_helpers import Plotting, Reasoning
from rooms import GRANULAR_ROOM


def input_act(agent, o):
    act_map = {'w':2, 'a':0, 's':3, 'd':1}
    act_list = [-1, 0, 1, 2, 3]  # 0: left, 1: right, 2: up, 3: down
    got = False
    while not got:
        x = agent.step(o)
        if x in act_map.keys():
            return act_map[x]
        try:
            x = int(x)
            if x in act_list:
                got = True
        except:
            pass
    return x


    

def navigate(room, agent, start_pos=None, display_mode=2):
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
        input()
        print('o', room[r, c])
        if room[r, c] == 1:
            pass
        a = input_act(agent, room[r, c])
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

room = GRANULAR_ROOM
name = 'navigation-granular_room'

n_emissions = np.max(room) + 1
c = np.zeros((n_emissions+1, 3))
c[:n_emissions] = Plotting.custom_colors[:n_emissions]

a, x, rc = datagen_structured_obs_room(room, length=5000)

n_clones = np.ones(n_emissions, dtype=np.int64) * 25

file = os.path.join("models", f"{name}.pkl")
if os.path.isfile(file) and not retrain_models:
    with open(file, 'rb') as f:
        (chmm, progression) = pickle.load(f)
else:
    chmm = CHMM(n_clones=n_clones, pseudocount=2e-3, x=x, a=a, seed=42)  # Initialize the model
    progression = chmm.learn_em_T(x, a, n_iter=1000)  # Training
    chmm.pseudocount = 0.0
    chmm.learn_viterbi_T(x, a, n_iter=100)
    with open(file, 'wb') as f: # open a text file
        pickle.dump((chmm, progression), protocol=5, file=f) # Serializes model object

# Plot the layout of the room
cmap = colors.ListedColormap(c[:n_emissions])

# Plot the learned graph
file = os.path.join("figures", f"{name}.png")
graph = Plotting.plot_graph(
    chmm, x, a, output_file=file, cmap=cmap, rotation=3
)

image = mpimg.imread(file)
fig, ax = plt.subplots()
ax.axis('off')
ax.imshow(image)

agent = Navigator(chmm, [42], [])

navigate(room, agent, (0, 0), display_mode=2)
