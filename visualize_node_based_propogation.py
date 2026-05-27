import os
import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from chmm_actions import CHMM, datagen_structured_obs_room
from CSCG_helpers import Plotting, Reasoning
from rooms import GRANULAR_ROOM


def add_key_legend(fig, text):
    fig.subplots_adjust(bottom=0.12)
    fig.text(
        0.5,
        0.03,
        text,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#222222",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
    )


def setup_and_train(use_model_cache=True):
    """Load or train the granular-room navigation model."""
    room = GRANULAR_ROOM
    name = "navigation-granular_room"

    n_emissions = np.max(room) + 1
    c = np.zeros((n_emissions + 1, 3))
    c[:n_emissions] = Plotting.custom_colors[:n_emissions]

    a, x, rc = datagen_structured_obs_room(room, length=5000)
    n_clones = np.ones(n_emissions, dtype=np.int64) * 25

    model_file = os.path.join("models", f"{name}.pkl")
    if os.path.isfile(model_file) and use_model_cache:
        with open(model_file, "rb") as f:
            chmm, progression = pickle.load(f)
    else:
        chmm = CHMM(n_clones=n_clones, pseudocount=2e-3, x=x, a=a, seed=42)
        progression = chmm.learn_em_T(x, a, n_iter=1000)
        chmm.pseudocount = 0.0
        chmm.learn_viterbi_T(x, a, n_iter=100)
        with open(model_file, "wb") as f:
            pickle.dump((chmm, progression), protocol=5, file=f)

    cmap = colors.ListedColormap(c[:n_emissions])
    return chmm, x, a, room, cmap


def plot_node_based_propogation(
    seq=None,
    seq_range=None,
    start_node=None,
    target_node=None,
    panels=("forward",),
    flip=None,
    rotation=0,
    use_model_cache=True,
):
    """
    Visualize node-based activity propagation on the learned CHMM graph.

    A plan is prepared by producting activity differences in node values:
    activity is placed on start/target latent nodes, propagated through the
    transition graph, and combined with strategies like sum or product. The
    idea is that neurons representating locations along the path from current
    position to goal will have high firing frequency. This strategy was
    created before I realized that the plan could be stored in the weights
    instead of the nodes.

    Common configurations:
    - include "room" to show the original room layout
    - include "graph" to show the learned latent graph
    - include "forward" to show forward activity spreading
    - include "backward", "sum", and "product" to compare update strategies
    """
    if seq is None and seq_range is None and start_node is None:
        raise ValueError("Provide seq, seq_range, or start_node.")

    valid_panels = {"room", "graph", "forward", "backward", "sum", "product"}
    invalid_panels = set(panels) - valid_panels
    if invalid_panels:
        raise ValueError(f"Unknown panels: {sorted(invalid_panels)}")

    chmm, x, a, room, cmap = setup_and_train(use_model_cache=use_model_cache)
    if seq is None and seq_range is not None:
        start, stop = seq_range
        seq = x[start:stop]

    support_figures = []

    if "room" in panels:
        # Render the original room layout.
        room_fig = plt.figure()
        plt.matshow(room, cmap=cmap, fignum=room_fig.number)
        plt.title("Figure 1: Room Layout")
        add_key_legend(room_fig, "Controls: n - step activity panels | q - quit")
        support_figures.append(room_fig)
        plt.savefig("figures/granular_room.pdf")

    if "graph" in panels:
        # Render the learned graph once.
        file = os.path.join("figures", "navigation-granular_room.png")
        Plotting.plot_graph(chmm, x, a, output_file=file, cmap=cmap, flip=flip, rotation=rotation)

        image = mpimg.imread(file)
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.imshow(image)
        add_key_legend(fig, "Controls: n - step activity panels | q - quit")
        support_figures.append(fig)

    if start_node is None:
        # Return the final forward-message belief after observing a sequence.
        mess_fwd = Reasoning.get_mess_fwd(chmm, seq, pseudocount_E=0.1)
        forward_v_init = mess_fwd[-1]
    else:
        # Return a one-hot activity vector for one latent node.
        forward_v_init = np.zeros(sum(chmm.n_clones))
        forward_v_init[start_node] = 1.0

    backward_v_init = None
    if target_node is not None:
        # Return a one-hot activity vector for one latent node.
        backward_v_init = np.zeros(sum(chmm.n_clones))
        backward_v_init[target_node] = 1.0

    activity_panels = tuple(panel for panel in panels if panel in ("forward", "backward", "sum", "product"))

    if any(panel in activity_panels for panel in ("backward", "sum", "product")) and backward_v_init is None:
        raise ValueError("target_node is required for backward, sum, or product panels.")

    image_paths = {
        "forward": "figures\\forward_reasoning_fig.png",
        "backward": "figures\\backward_reasoning_fig.png",
        "sum": "figures\\sum_reasoning_fig.png",
        "product": "figures\\prod_reasoning_fig.png",
    }

    t = 0
    forward_v = forward_v_init
    backward_v = backward_v_init
    sum_v = forward_v_init + backward_v_init if backward_v_init is not None else None
    product_v = forward_v_init * backward_v_init if backward_v_init is not None else None

    panel_state = {}
    initial_values = {
        "forward": forward_v,
        "backward": backward_v,
        "sum": sum_v,
        "product": product_v,
    }

    def make_panel(activity, image_path, title):
        Plotting.plot_heat_map(
            chmm,
            x,
            a,
            activity,
            output_file=image_path,
            flip=flip,
            rotation=rotation,
        )
        image = mpimg.imread(image_path)
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.set_title(title)
        img_display = ax.imshow(image, cmap="viridis")
        plt.colorbar(img_display, ax=ax, orientation="vertical")
        add_key_legend(fig, "Controls: n - step | q - quit")
        return fig, ax, img_display

    for panel in activity_panels:
        panel_state[panel] = make_panel(
            initial_values[panel],
            image_paths[panel],
            f"{panel}, t={t}",
        )

    def redraw_panel(panel, activity, title):
        fig, ax, img_display = panel_state[panel]
        Plotting.plot_heat_map(
            chmm,
            x,
            a,
            activity,
            output_file=image_paths[panel],
            flip=flip,
            rotation=rotation,
        )
        img_display.set_data(mpimg.imread(image_paths[panel]))
        ax.set_title(title)
        fig.canvas.draw()

    def update_image(event):
        nonlocal t, forward_v, backward_v, sum_v, product_v
        if event.key == "q":
            plt.close("all")
            return
        if event.key != "n":
            return

        t += 1

        if "forward" in activity_panels or "sum" in activity_panels or "product" in activity_panels:
            forward_v = Reasoning.forwardV(forward_v, forward_v_init, chmm.T)
            print(max(forward_v), "forward")

        if backward_v is not None and (
            "backward" in activity_panels or "sum" in activity_panels or "product" in activity_panels
        ):
            backward_v = Reasoning.backwardV(backward_v, backward_v_init, chmm.T)
            print(max(backward_v), "backward")

        if "forward" in activity_panels:
            redraw_panel("forward", forward_v, f"forward, t={t}")

        if "backward" in activity_panels:
            redraw_panel("backward", backward_v, f"backward, t={t}")

        if "sum" in activity_panels:
            sum_v = forward_v + backward_v
            print(max(sum_v), "sum")
            redraw_panel("sum", sum_v, f"sum, t={t}")

        if "product" in activity_panels:
            product_v = forward_v * backward_v
            print(max(product_v), "prod")
            redraw_panel("product", product_v, f"product, t={t}")

        print()

    for fig, _, _ in panel_state.values():
        fig.canvas.mpl_connect("key_press_event", update_image)
    for fig in support_figures:
        fig.canvas.mpl_connect("key_press_event", update_image)

    plt.show()


if __name__ == "__main__":
    # test forward v which propogates a single burst of v_init through the network
    # plot_node_based_propogation(seq_range=(1690, 1700), panels=("room", "graph", "forward"), flip=True, rotation=.83)

    # test 4 different strategies (for changing node values)
    # plot_node_based_propogation(seq_range=(1690, 1700), target_node=42, panels=("room", "graph", "forward", "backward", "sum", "product"))

    # test 4 different strategies starting from a fixed latent node instead of decoded behavior
    plot_node_based_propogation(start_node=52, target_node=42, panels=("room", "graph", "forward", "backward", "sum", "product"))
    pass
