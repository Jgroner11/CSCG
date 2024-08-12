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

def save_image(t, file, rotation=0):
    V = mess_fwd[t]
    graph = Plotting.plot_heat_map(
        chmm, x, a, V, output_file=file, rotation=rotation
    )

def plot_path(start = None, plot_location=True, rotation = 0):
    if start is None:
        start = 0
    end = len(mess_fwd)
    img_path = "figures\\plot_fig.png"
    save_image(start, img_path, rotation=rotation)
    image = mpimg.imread(img_path)
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.set_title(f'mess_fwd activity at t={start}')
    img_display = ax.imshow(image, cmap='viridis')
    cbar = plt.colorbar(img_display, ax=ax, orientation='vertical')
    if plot_location:
        location_fig, location_ax, text = Plotting.plot_room(room, pos=(rc[start, 0], rc[start, 1]), t=start)
    t = start
    def update_image(event):
        """Updates the plot with the next image when the specified key is pressed."""
        nonlocal t, text, location_ax
        if event.key == 'n' or event.key == 'b':  # 'n' key for next time step, 'b' key to back one time step 
            if event.key == 'n' and t < end:
                t += 1
            elif event.key == 'b' and t > 0:
                t -= 1
            save_image(t, img_path, rotation=rotation)
            new_image = mpimg.imread(img_path)
            img_display.set_data(new_image)
            ax.set_title(f'mess_fwd activity at t={t}')
            fig.canvas.draw()
            if plot_location:
                text = Plotting.redraw_room(location_fig, location_ax, (rc[t, 0], rc[t, 1]), old_text=text, t=t)

    fig.canvas.mpl_connect('key_press_event', update_image)
    location_fig.canvas.mpl_connect('key_press_event', update_image)
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
room = simple_granular_room
name = 'navigation-simple_granular_room'

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

# Plot the learned graph
file = os.path.join("figures", f"{name}.png")
graph = Plotting.plot_graph(
    chmm, x, a, output_file=file, cmap=cmap, rotation=3
)

image = mpimg.imread(file)
fig, ax = plt.subplots()
ax.axis('off')
ax.imshow(image)


mess_fwd = Reasoning.get_mess_fwd(chmm, x, pseudocount_E=0.1)
plot_path(0, rotation = 3)

