import sys, os

import numpy as np
from chmm_actions import CHMM, forwardE, datagen_structured_obs_room
import matplotlib.pyplot as plt
import igraph
from matplotlib import cm, colors
import os
from igraph.drawing import plot

custom_colors = (
    np.array(
        [
            [214, 214, 214],
            [85, 35, 157],
            [253, 252, 144],
            [114, 245, 144],
            [151, 38, 20],
            [239, 142, 192],
            [214, 134, 48],
            [140, 194, 250],
            [72, 160, 162],
        ]
    )
    / 256
)

if not os.path.exists("figures"):
    os.makedirs("figures")

def plot_graph(
    chmm, x, a, output_file, cmap=cm.Spectral, multiple_episodes=False, vertex_size=30
):
    states = chmm.decode(x, a)[1]

    v = np.unique(states)
    if multiple_episodes:
        T = chmm.C[:, v][:, :, v][:-1, 1:, 1:]
        v = v[1:]
    else:
        T = chmm.C[:, v][:, :, v]
    A = T.sum(0)
    A /= A.sum(1, keepdims=True)

    g = igraph.Graph.Adjacency((A > 0).tolist())
    node_labels = np.arange(x.max() + 1).repeat(n_clones)[v]
    if multiple_episodes:
        node_labels -= 1
    colors = [cmap(nl)[:3] for nl in node_labels / node_labels.max()]
    out = igraph.plot(
        g,
        output_file,
        layout=g.layout("kamada_kawai"),
        vertex_color=colors,
        vertex_label=v,
        vertex_size=vertex_size,
        margin=50,
    )

    return out

def datagen_structured_obs_linear_track(
    linear_track,
    start_r=None,
    length=10000,
    seed=42,
):
    """linear_track is a 2d numpy array. Each row represents one possible
    variation of the track. The rat can only perform one action (right) to move
    forward along the track. After reaching the end of the track, the right
    action takes the rat to the start of a random (uniformly selected) track.

    Returns:
      a: a list of action states, every element will be 1, the action corresponding to right
      x: list of observations
      rc: the row, col that the rat is at during the observation

    """
    np.random.seed(seed)
    n_r, n_c = linear_track.shape
    if start_r is None:
        start_r = np.random.randint(n_r)
    start_c = 0

    actions = np.ones(length, int) * 1 #1 is right
    x = np.zeros(length, int)  # observations
    rc = np.zeros((length, 2), int)  # actual r&c

    r, c = start_r, start_c
    x[0] = linear_track[r, c]
    rc[0] = r, c

    count = 0
    while count < length - 1:

        if c < n_c - 1:
            c += 1
        else:
          c = 0
          r = np.random.randint(n_r)


        x[count + 1] = linear_track[r, c]
        rc[count + 1] = r, c
        count += 1

    return actions, x, rc


room = np.array([
    [0, 0, 0, 1, 1, 0, 0, 3, 3, 0, 0, 4, 4, 0, 0],
    [0, 0, 0, 2, 2, 0, 0, 4, 4, 0, 0, 3, 3, 0, 0]
])


n_emissions = room.max() + 1
print('n_emissions', n_emissions)

# Plot the layout of the room
cmap = colors.ListedColormap(custom_colors[:n_emissions])
plt.matshow(room, cmap=cmap)
plt.title('Figure 1: Room Layout')
plt.savefig("figures/rectangular_room_layout.pdf")

a, x, rc = datagen_structured_obs_linear_track(room, length=30)     #Use length=50000 for bigger room

print('a', a)
print('x', x)
print('rc', rc)

# Session 1

print('Session 1')
n_clones = np.ones(n_emissions, dtype=np.int64) * 25
print('n_clones', n_clones)
chmm = CHMM(n_clones=n_clones, pseudocount=2e-3, x=x, a=a, seed=42)  # Initialize the model
progression = chmm.learn_em_T(x, a, n_iter=1)  # Training   use n_iter=1000 for better training

# Consolidate learning. Takes a few seconds
chmm.pseudocount = 0.0
chmm.learn_viterbi_T(x, a, n_iter=100)

graph = plot_graph(
    chmm, x, a, output_file="figures/rectangular_room_graph.pdf", cmap=cmap
)
plot(graph)


# Look for the correspondence between the graph and the original layout of the rooom in Figure 1
# Node colors correspond to the observations from the room. Node numbers are the clone/neuron numbers.

# Session 2
print('Session 2')
a, x, rc = datagen_structured_obs_linear_track(room, length=100)     #Use length=50000 for bigger room
print('n_clones', n_clones)
chmm = CHMM(n_clones=n_clones, pseudocount=2e-3, x=x, a=a, seed=42)  # Initialize the model
progression = chmm.learn_em_T(x, a, n_iter=5)  # Training   use n_iter=1000 for better training
chmm.pseudocount = 0.0
chmm.learn_viterbi_T(x, a, n_iter=100)
graph = plot_graph(chmm, x, a, output_file="figures/rectangular_room_graph.pdf", cmap=cmap)
plot(graph)


# Session 3

print('Session 3')
a, x, rc = datagen_structured_obs_linear_track(room, length=1000)     #Use length=50000 for bigger room
print('n_clones', n_clones)
chmm = CHMM(n_clones=n_clones, pseudocount=2e-3, x=x, a=a, seed=42)  # Initialize the model
progression = chmm.learn_em_T(x, a, n_iter=100)  # Training   use n_iter=1000 for better training
chmm.pseudocount = 0.0
chmm.learn_viterbi_T(x, a, n_iter=100)
graph = plot_graph(chmm, x, a, output_file="figures/rectangular_room_graph.pdf", cmap=cmap)
plot(graph)
plt.show()

# Session 4
# print('Session 4')
# a, x, rc = datagen_structured_obs_linear_track(room, length=50000)     #Use length=50000 for bigger room
# print('n_clones', n_clones)
# chmm = CHMM(n_clones=n_clones, pseudocount=2e-3, x=x, a=a, seed=42)  # Initialize the model
# progression = chmm.learn_em_T(x, a, n_iter=1000)  # Training   use n_iter=1000 for better training
# chmm.pseudocount = 0.0
# chmm.learn_viterbi_T(x, a, n_iter=100)
# graph = plot_graph(chmm, x, a, output_file="figures/rectangular_room_graph.pdf", cmap=cmap)
# graph


# # Session 5
# a, x, rc = datagen_structured_obs_linear_track(room, length=50000)     #Use length=50000 for bigger room
# n_clones = np.ones(n_emissions, dtype=np.int64) * 75
# print('n_clones', n_clones)
# chmm = CHMM(n_clones=n_clones, pseudocount=2e-3, x=x, a=a, seed=42)  # Initialize the model
# progression = chmm.learn_em_T(x, a, n_iter=1000)  # Training   use n_iter=1000 for better training
# chmm.pseudocount = 0.0
# chmm.learn_viterbi_T(x, a, n_iter=100)
# graph = plot_graph(chmm, x, a, output_file="figures/rectangular_room_graph.pdf", cmap=cmap)
# graph

