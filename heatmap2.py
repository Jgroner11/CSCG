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

def plot_reasoning(seq):
    mess_fwd = Reasoning.get_mess_fwd(chmm, seq, pseudocount_E=0.1)
    V_init = mess_fwd[-1]
    img_path = "figures\\reasoning_fig.png"
    graph = Plotting.plot_heat_map(
        chmm, x, a, V_init, output_file=img_path
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
            V = Reasoning.forwardV(V, V_init, chmm.T)
            t += 1
            ax.set_title(f't={t}')
            print(sum(V_init), sum(V))
            graph = Plotting.plot_heat_map(
                chmm, x, a, V, output_file=img_path
            )
            new_image = mpimg.imread(img_path)
            img_display.set_data(new_image)
            fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', update_image)
    plt.show()

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

plot_reasoning(x[1690:1700])

