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

def plot_reasoning(seq, target):
    t = 0

    # Forward
    mess_fwd = Reasoning.get_mess_fwd(chmm, seq, pseudocount_E=0.1)
    forward_V_init = mess_fwd[-1]
    forward_img_path = "figures\\forward_reasoning_fig.png"
    forward_graph = Plotting.plot_heat_map(
        chmm, x, a, forward_V_init, output_file=forward_img_path
    )
    forward_image = mpimg.imread(forward_img_path)
    forward_fig, forward_ax = plt.subplots()
    forward_ax.axis('off')
    forward_ax.set_title(f'forward, t={t}')
    forward_img_display = forward_ax.imshow(forward_image, cmap='viridis')
    forward_cbar = plt.colorbar(forward_img_display, ax=forward_ax, orientation='vertical')
    forward_V = forward_V_init
    
    # Backward
    backward_V_init = np.zeros(forward_V_init.shape)
    backward_V_init[target] = 1.0
    
    backward_img_path = "figures\\backward_reasoning_fig.png"
    backward_graph = Plotting.plot_heat_map(
        chmm, x, a, backward_V_init, output_file=backward_img_path
    )
    backward_image = mpimg.imread(backward_img_path)
    backward_fig, backward_ax = plt.subplots()
    backward_ax.axis('off')
    backward_ax.set_title(f'backward, t={t}')
    backward_img_display = backward_ax.imshow(backward_image, cmap='viridis')
    backward_cbar = plt.colorbar(backward_img_display, ax=backward_ax, orientation='vertical')
    backward_V = backward_V_init

    # Sum
    sum_V_init = forward_V_init + backward_V_init
    sum_img_path = "figures\\sum_reasoning_fig.png"
    sum_graph = Plotting.plot_heat_map(
        chmm, x, a, sum_V_init, output_file=sum_img_path
    )
    sum_image = mpimg.imread(sum_img_path)
    sum_fig, sum_ax = plt.subplots()
    sum_ax.axis('off')
    sum_ax.set_title(f'sum, t={t}')
    sum_img_display = sum_ax.imshow(sum_image, cmap='viridis')
    sum_cbar = plt.colorbar(sum_img_display, ax=sum_ax, orientation='vertical')
    sum_V = sum_V_init

    # Product
    prod_V_init = forward_V_init * backward_V_init
    prod_img_path = "figures\\prod_reasoning_fig.png"
    prod_graph = Plotting.plot_heat_map(
        chmm, x, a, prod_V_init, output_file=prod_img_path
    )
    prod_image = mpimg.imread(prod_img_path)
    prod_fig, prod_ax = plt.subplots()
    prod_ax.axis('off')
    prod_ax.set_title(f'product, t={t}')
    prod_img_display = prod_ax.imshow(prod_image, cmap='viridis')
    prod_cbar = plt.colorbar(prod_img_display, ax=prod_ax, orientation='vertical')
    prod_V = prod_V_init
 
    def update_image(event):
        """Updates the plot with the next image when the specified key is pressed."""
        nonlocal forward_V, backward_V, sum_V, prod_V, t
        if event.key == 'n':  # 'n' key for next image
            t += 1

            # Forward
            forward_V = Reasoning.forwardV(forward_V, forward_V_init, chmm.T)
            print(max(forward_V), 'forward')
            forward_graph = Plotting.plot_heat_map(
                chmm, x, a, forward_V, output_file=forward_img_path
            )
            forward_new_image = mpimg.imread(forward_img_path)
            forward_img_display.set_data(forward_new_image)
            forward_ax.set_title(f'forward, t={t}')
            forward_fig.canvas.draw()

            # Backward
            backward_V = Reasoning.backwardV(backward_V, backward_V_init, chmm.T)
            print(max(backward_V), 'backward')
            backward_graph = Plotting.plot_heat_map(
                chmm, x, a, backward_V, output_file=backward_img_path
            )
            backward_new_image = mpimg.imread(backward_img_path)
            backward_img_display.set_data(backward_new_image)
            backward_ax.set_title(f'backward, t={t}')
            backward_fig.canvas.draw()

            # Sum
            sum_V = forward_V + backward_V
            print(max(sum_V), 'sum')
            sum_graph = Plotting.plot_heat_map(
                chmm, x, a, sum_V, output_file=sum_img_path
            )
            sum_new_image = mpimg.imread(sum_img_path)
            sum_img_display.set_data(sum_new_image)
            sum_ax.set_title(f'sum, t={t}')
            sum_fig.canvas.draw()

            # Product
            prod_V = forward_V * backward_V
            print(max(prod_V), 'prod')
            prod_graph = Plotting.plot_heat_map(
                chmm, x, a, prod_V, output_file=prod_img_path
            )
            prod_new_image = mpimg.imread(prod_img_path)
            prod_img_display.set_data(prod_new_image)
            prod_ax.set_title(f'product, t={t}')
            prod_fig.canvas.draw()

            print('\n')
            

    forward_fig.canvas.mpl_connect('key_press_event', update_image)
    backward_fig.canvas.mpl_connect('key_press_event', update_image)
    sum_fig.canvas.mpl_connect('key_press_event', update_image)
    prod_fig.canvas.mpl_connect('key_press_event', update_image)
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

plot_reasoning(x[1690:1700], 42)

