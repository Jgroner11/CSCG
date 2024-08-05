import numpy as np
from chmm_actions import CHMM, forwardE, datagen_structured_obs_room
import matplotlib.pyplot as plt
import igraph
import matplotlib
from matplotlib import cm, colors
import matplotlib.image as mpimg
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

def plot_graph(
    chmm, x, a, output_file, cmap=cm.Spectral, multiple_episodes=False, vertex_size=30
):
    global n_clones
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

def get_mess_fwd(chmm, x, pseudocount=0.0, pseudocount_E=0.0):
    n_clones = chmm.n_clones
    E = np.zeros((n_clones.sum(), len(n_clones)))
    last = 0
    for c in range(len(n_clones)):
        E[last : last + n_clones[c], c] = 1
        last += n_clones[c]
    E += pseudocount_E
    norm = E.sum(1, keepdims=True)
    norm[norm == 0] = 1
    E /= norm
    T = chmm.C + pseudocount
    norm = T.sum(2, keepdims=True)
    norm[norm == 0] = 1
    T /= norm
    T = T.mean(0, keepdims=True)
    log2_lik, mess_fwd = forwardE(
        T.transpose(0, 2, 1), E, chmm.Pi_x, chmm.n_clones, x, x * 0, store_messages=True
    )
    return mess_fwd

def plot_heat_map(
    chmm, x, a, V, output_file, multiple_episodes=False, vertex_size=30
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

    # Rn skip normalizing step
    V_disp_norm = (V_displayed_nodes - np.min(V_displayed_nodes)) / (np.max(V_displayed_nodes) - np.min(V_displayed_nodes))
    # V_disp_norm = V_displayed_nodes

    colormap = matplotlib.colormaps['viridis']
    colors = colormap(V_disp_norm)
    colors = [tuple(c) for c in colors]

    g = igraph.Graph.Adjacency((A > 0).tolist())

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

simple_granular_room = np.array(
    [[4, 2, 4, 0],
    [3, 0, 0, 2],
    [4, 1, 3, 0],
    [3, 3, 2, 0]]
)
n_emissions = np.max(simple_granular_room) + 1
c = np.zeros((n_emissions+1, 3))
c[:n_emissions] = custom_colors[:n_emissions]


a, x, rc = datagen_structured_obs_room(simple_granular_room, length=5000)

n_clones = np.ones(n_emissions, dtype=np.int64) * 25

name = 'navigation-simple_granular_room'
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
plt.matshow(simple_granular_room, cmap=cmap)
plt.title('Figure 1: Room Layout')
plt.savefig("figures/granular_room.pdf")

file = os.path.join("figures", f"{name}.png")
graph = plot_graph(
    chmm, x, a, output_file=file, cmap=cmap
)

image = mpimg.imread(file)
fig, ax = plt.subplots()
ax.axis('off')
ax.imshow(image)

def updateV(V, V_init, T):
        s = V_init
        for i in range(T.shape[0]):
            s += V @ T[i]
        return s

def plot_reasoning(seq):
    mess_fwd = get_mess_fwd(chmm, seq, pseudocount_E=0.1)
    V_init = mess_fwd[-1]
    img_path = "figures\\reasoning_fig.png"
    graph = plot_heat_map(
        chmm, x, a, V_init, output_file=img_path
    )
    image = mpimg.imread(img_path)
    fig, ax = plt.subplots()
    ax.axis('off')
    img_display = ax.imshow(image, cmap='viridis')
    cbar = plt.colorbar(img_display, ax=ax, orientation='vertical')

    V = V_init
    def update_image(event):
        """Updates the plot with the next image when the specified key is pressed."""
        nonlocal V
        if event.key == 'n':  # 'n' key for next image
            V = updateV(V, V_init, chmm.T)
            print(max(V))
            graph = plot_heat_map(
                chmm, x, a, V, output_file=img_path
            )
            new_image = mpimg.imread(img_path)
            img_display.set_data(new_image)
            fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', update_image)
    plt.show()


plot_reasoning(x[200:300])

def forward_search(chmm, x, n_iters=10):
    mess_fwd = get_mess_fwd(chmm, x, pseudocount_E=0.1)
    V_init = mess_fwd[-1]
    V = V_init
    for i in range(n_iters):
        V = updateV(V, V_init, chmm.C)


# sequential colormaps:
# viridis
# plasma
# inferno
# magma
# cividis
# Blues
# BuGn
# BuPu
# GnBu
# Greens
# Greys
# Oranges
# OrRd
# PuBu
# PuBuGn
# PuRd
# Purples
# RdPu
# Reds
# YlGn
# YlGnBu
# YlOrBr
# YlOrRd