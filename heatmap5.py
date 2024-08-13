import numpy as np
import matplotlib.pyplot as plt
import igraph
import matplotlib
from matplotlib import cm, colors
import matplotlib.image as mpimg
import sys, os, pickle
import math

from chmm_actions import CHMM, forwardE, datagen_structured_obs_room
from CSCG_helpers import Plotting, Reasoning

def plot_heat_map(
    chmm, x, a, V, T_, output_file, multiple_episodes=False, vertex_size=30, rotation = 0
):
    # States is a list of which latent node (ie state) is most active at each time step
    states = chmm.decode(x, a)[1]

    v = np.unique(states)
    if multiple_episodes:
        T = chmm.C[:, v][:, :, v][:-1, 1:, 1:]
        v = v[1:]
    else:
        T = chmm.C[:, v][:, :, v]
    A = T.sum(0)
    A /= A.sum(1, keepdims=True)
    # A is a transition matrix of only the latent nodes (states) that get activated during walk of path

    # V_displayed represents the activity of all the nodes that are present in the A matrix/graph based on the inputted V activity for all the nodes
    V_displayed_nodes = np.zeros(v.shape)
    for i, id in enumerate(v):
        V_displayed_nodes[i] = V[id]

    # print('sum V:', sum(V), 'sum Vdisp:', sum(V_displayed_nodes))

    V_disp_norm = V_displayed_nodes
    if (np.max(V_displayed_nodes) - np.min(V_displayed_nodes)) > 0:
        V_disp_norm = (V_displayed_nodes - np.min(V_displayed_nodes)) / (np.max(V_displayed_nodes) - np.min(V_displayed_nodes))
        

    # colormap = cm.get_cmap('viridis')
    colormap = matplotlib.colormaps['viridis']
    colors = colormap(V_disp_norm)
    colors = [tuple(c) for c in colors]

    g = igraph.Graph.Adjacency((A > 0).tolist())

    A_ = T_.sum(0)
    edge_labels = ["" for _ in g.es]
    for index, edge in enumerate(g.es):
        i = v[edge.source]
        j = v[edge.target]
        if i != j:        
            edge_labels[index] = str(round(A_[j, i], 2))
            # edge_labels[index] = str((int(j), int(i)))


    out = igraph.plot(
        g,
        output_file,
        layout=[Plotting.rotate(x, y, 90 * rotation) for x, y in g.layout("kamada_kawai")],
        vertex_color=colors,
        vertex_label=V_displayed_nodes,
        # vertex_label=v,
        vertex_size=vertex_size,
        edge_label=edge_labels,
        margin=50,
    )

    return out

def plot_reasoning(targets):
    V_init = np.zeros(sum(chmm.n_clones))
    for i in targets:
        V_init[i] = 1.0
    T_init = chmm.T

    img_path = "figures\\reasoning_fig.png"
    graph = plot_heat_map(
        chmm, x, a, V_init, T_init, output_file=img_path
    )
    image = mpimg.imread(img_path)
    fig, ax = plt.subplots()
    ax.axis('off')
    t = 0
    ax.set_title(f't={t}')
    img_display = ax.imshow(image, cmap='viridis')
    cbar = plt.colorbar(img_display, ax=ax, orientation='vertical')

    V = V_init
    T = T_init
    def update_image(event):
        """Updates the plot with the next image when the specified key is pressed."""
        nonlocal V, T, t
        if event.key == 'n':  # 'n' key for next image
            V, T = Reasoning.STP(V, T)
            t += 1
            ax.set_title(f't={t}')
            print(sum(V_init), sum(V))
            graph = plot_heat_map(
                chmm, x, a, V, T, output_file=img_path
            )
            new_image = mpimg.imread(img_path)
            img_display.set_data(new_image)
            fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', update_image)
    plt.show()
    return T

def plot_planning(start, T):
    V_init = np.zeros(sum(chmm.n_clones))
    V_init[start] = 1.0

    img_path = "figures\\reasoning_fig.png"
    graph = plot_heat_map(
        chmm, x, a, V_init, T, output_file=img_path
    )
    image = mpimg.imread(img_path)
    fig, ax = plt.subplots()
    ax.axis('off')
    t = 0
    ax.set_title(f't={t}')
    img_display = ax.imshow(image, cmap='viridis')
    cbar = plt.colorbar(img_display, ax=ax, orientation='vertical')

    V = V_init
    def update_image(event):
        """Updates the plot with the next image when the specified key is pressed."""
        nonlocal V, t
        if event.key == 'n':  # 'n' key for next image
            V = Reasoning.forward(V, T, V_init)
            t += 1
            ax.set_title(f't={t}')
            print(sum(V_init), sum(V))
            graph = plot_heat_map(
                chmm, x, a, V, T, output_file=img_path
            )
            new_image = mpimg.imread(img_path)
            img_display.set_data(new_image)
            fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', update_image)
    plt.show()
    return T

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
granular_room = np.array(
    [
        [4, 2, 3, 0, 3, 4, 4, 4],
        [4, 4, 3, 2, 3, 2, 3, 4],
        [4, 4, 2, 0, 4, 2, 4, 0],
        [0, 2, 4, 4, 3, 0, 0, 2],
        [3, 3, 4, 0, 4, 1, 3, 0],
        [2, 4, 2, 3, 3, 3, 2, 0],
    ]
)

room = granular_room
name = 'navigation-granular_room'

n_emissions = np.max(room) + 1
c = np.zeros((n_emissions+1, 3))
c[:n_emissions] = custom_colors[:n_emissions]


a, x, rc = datagen_structured_obs_room(room, length=5000)

n_clones = np.ones(n_emissions, dtype=np.int64) * 25

file = os.path.join("models", f"{name}.pkl")
if os.path.isfile(file) and not retrain_models:
    with open(file, 'rb') as f:
        (chmm, progression) = pickle.load(f)
else:
    chmm = CHMM(n_clones=n_clones, pseudocount=2e-3, x=x, a=a, seed=42)  # Initialize the model
    progression = chmm.learn_em_T(x, a, n_iter=1000)  # Training
    with open(file, 'wb') as f: # open a text file
        pickle.dump((chmm, progression), protocol=5, file=f) # Serializes model object

chmm.pseudocount = 0.0
chmm.learn_viterbi_T(x, a, n_iter=100)

# Plot the layout of the room
cmap = colors.ListedColormap(c[:n_emissions])
plt.matshow(room, cmap=cmap)
plt.title('Figure 1: Room Layout')
plt.savefig("figures/granular_room.pdf")

file = os.path.join("figures", f"{name}.png")
graph = Plotting.plot_graph(
    chmm, x, a, output_file=file, cmap=cmap
)

image = mpimg.imread(file)
fig, ax = plt.subplots()
ax.axis('off')
ax.imshow(image)

start = 5
targets = [101, 83]

T = plot_reasoning(targets)
T = plot_planning(start, T)

