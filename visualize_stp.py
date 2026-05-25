import os
import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from chmm_actions import CHMM, datagen_structured_obs_room
from CSCG_helpers import Plotting, Reasoning


CUSTOM_COLORS = (
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

SIMPLE_GRANULAR_ROOM = np.array(
    [
        [4, 2, 4, 0],
        [3, 0, 0, 2],
        [4, 1, 3, 0],
        [3, 3, 2, 0],
    ]
)

GRANULAR_ROOM = np.array(
    [
        [4, 2, 3, 0, 3, 4, 4, 4],
        [4, 4, 3, 2, 3, 2, 3, 4],
        [4, 4, 2, 0, 4, 2, 4, 0],
        [0, 2, 4, 4, 3, 0, 0, 2],
        [3, 3, 4, 0, 4, 1, 3, 0],
        [2, 4, 2, 3, 3, 3, 2, 0],
    ]
)

DEFAULT_MODEL_NAME = "navigation-granular_room"
DEFAULT_IMAGE_PATH = "figures\\reasoning_fig.png"

chmm = None
x = None
a = None
room = None
cmap = None


def setup_navigation_model(
    selected_room=None,
    name=DEFAULT_MODEL_NAME,
    retrain_models=False,
    length=5000,
    clone_count=25,
    seed=42,
):
    Plotting.custom_colors = CUSTOM_COLORS

    if selected_room is None:
        selected_room = GRANULAR_ROOM

    n_emissions = np.max(selected_room) + 1
    color_values = np.zeros((n_emissions + 1, 3))
    color_values[:n_emissions] = CUSTOM_COLORS[:n_emissions]

    actions, observations, rc = datagen_structured_obs_room(selected_room, length=length)
    n_clones = np.ones(n_emissions, dtype=np.int64) * clone_count

    model_file = os.path.join("models", f"{name}.pkl")
    if os.path.isfile(model_file) and not retrain_models:
        with open(model_file, "rb") as f:
            model, progression = pickle.load(f)
    else:
        model = CHMM(n_clones=n_clones, pseudocount=2e-3, x=observations, a=actions, seed=seed)
        progression = model.learn_em_T(observations, actions, n_iter=1000)
        with open(model_file, "wb") as f:
            pickle.dump((model, progression), protocol=5, file=f)

    model.pseudocount = 0.0
    model.learn_viterbi_T(observations, actions, n_iter=100)

    room_cmap = colors.ListedColormap(color_values[:n_emissions])
    return model, observations, actions, rc, selected_room, room_cmap


def load_default_context(retrain_models=False):
    global chmm, x, a, room, cmap
    chmm, x, a, _rc, room, cmap = setup_navigation_model(retrain_models=retrain_models)
    return chmm, x, a, room, cmap


def _context(model=None, observations=None, actions=None):
    model = chmm if model is None else model
    observations = x if observations is None else observations
    actions = a if actions is None else actions
    if model is None or observations is None or actions is None:
        model, observations, actions, _room, _cmap = load_default_context()
    return model, observations, actions


def _one_hot(nodes, size, value=1.0):
    vector = np.zeros(size)
    for node in nodes:
        vector[node] = value
    return vector


def _make_activity_figure(model, observations, actions, values, transition_weights, title, image_path, flip, rotation):
    Plotting.plot_heat_map(
        model,
        observations,
        actions,
        values,
        output_file=image_path,
        flip=flip,
        rotation=rotation,
        transition_weights=transition_weights,
        edge_label_mode="int",
        vertex_label_mode="value",
    )
    image = mpimg.imread(image_path)
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.set_title(title)
    img_display = ax.imshow(image, cmap="viridis")
    plt.colorbar(img_display, ax=ax, orientation="vertical")
    return fig, ax, img_display


def _redraw_activity(model, observations, actions, values, transition_weights, image_path, img_display, ax, title, flip, rotation):
    Plotting.plot_heat_map(
        model,
        observations,
        actions,
        values,
        output_file=image_path,
        flip=flip,
        rotation=rotation,
        transition_weights=transition_weights,
        edge_label_mode="int",
        vertex_label_mode="value",
    )
    img_display.set_data(mpimg.imread(image_path))
    ax.set_title(title)
    ax.figure.canvas.draw()


def plot_reasoning(
    targets,
    model=None,
    observations=None,
    actions=None,
    image_path=DEFAULT_IMAGE_PATH,
    flip=True,
    rotation=0.9,
):
    model, observations, actions = _context(model, observations, actions)
    values = _one_hot(targets, sum(model.n_clones))
    transition_weights = model.T

    fig, ax, img_display = _make_activity_figure(
        model,
        observations,
        actions,
        values,
        transition_weights,
        "t=0",
        image_path,
        flip,
        rotation,
    )

    t = 0

    def update_image(event):
        nonlocal values, transition_weights, t
        if event.key != "n":
            return
        values, transition_weights = Reasoning.STP(values, transition_weights)
        t += 1
        _redraw_activity(
            model,
            observations,
            actions,
            values,
            transition_weights,
            image_path,
            img_display,
            ax,
            f"t={t}",
            flip,
            rotation,
        )

    fig.canvas.mpl_connect("key_press_event", update_image)
    plt.show()
    return transition_weights


def plot_planning(
    starts,
    transition_weights,
    model=None,
    observations=None,
    actions=None,
    image_path=DEFAULT_IMAGE_PATH,
    flip=True,
    rotation=0.9,
):
    model, observations, actions = _context(model, observations, actions)
    initial_values = _one_hot(starts, sum(model.n_clones))
    values = initial_values

    fig, ax, img_display = _make_activity_figure(
        model,
        observations,
        actions,
        values,
        transition_weights,
        "t=0",
        image_path,
        flip,
        rotation,
    )

    print("chosen action", Reasoning.select_action(initial_values, transition_weights), "\n")
    t = 0

    def update_image(event):
        nonlocal values, t
        if event.key != "n":
            return
        values = Reasoning.propogate(values, transition_weights, initial_values)
        t += 1
        _redraw_activity(
            model,
            observations,
            actions,
            values,
            transition_weights,
            image_path,
            img_display,
            ax,
            f"t={t}",
            flip,
            rotation,
        )
        print("chosen action", Reasoning.select_action(values, transition_weights), "\n")

    fig.canvas.mpl_connect("key_press_event", update_image)
    plt.show()


def plot_reasoning_then_planning(
    targets,
    starts,
    model=None,
    observations=None,
    actions=None,
    image_path=DEFAULT_IMAGE_PATH,
    flip=True,
    rotation=0.9,
):
    model, observations, actions = _context(model, observations, actions)
    initial_values = _one_hot(targets, sum(model.n_clones))
    values = initial_values
    transition_weights = model.T
    mode = "Wavefront"
    t = 0

    fig, ax, img_display = _make_activity_figure(
        model,
        observations,
        actions,
        values,
        transition_weights,
        f"{mode}: t={t}",
        image_path,
        flip,
        rotation,
    )

    def update_image(event):
        nonlocal mode, t, values, transition_weights, initial_values
        if event.key == "n":
            t += 1
            if mode == "Wavefront":
                values, transition_weights = Reasoning.STP(values, transition_weights)
            else:
                values = Reasoning.propogate(values, transition_weights, initial_values)
                print("chosen action", Reasoning.select_action(values, transition_weights), "\n")
            _redraw_activity(
                model,
                observations,
                actions,
                values,
                transition_weights,
                image_path,
                img_display,
                ax,
                f"{mode}: t={t}",
                flip,
                rotation,
            )
        elif mode == "Wavefront" and event.key == "m":
            mode = "Planning"
            t = 0
            initial_values = _one_hot(starts, sum(model.n_clones))
            values = initial_values
            print("chosen action", Reasoning.select_action(initial_values, transition_weights), "\n")
            _redraw_activity(
                model,
                observations,
                actions,
                values,
                transition_weights,
                image_path,
                img_display,
                ax,
                f"{mode}: t={t}",
                flip,
                rotation,
            )

    fig.canvas.mpl_connect("key_press_event", update_image)
    plt.show()
    return transition_weights


def show_graph_and_plan(starts, transition_weights, name=DEFAULT_MODEL_NAME, flip=True, rotation=0.9):
    model, observations, actions = _context()
    output_file = os.path.join("figures", f"{name}.png")
    Plotting.plot_graph(model, observations, actions, output_file=output_file, cmap=cmap, flip=flip, rotation=rotation)

    image = mpimg.imread(output_file)
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.imshow(image)

    state_seq, obs_seq, action_seq = Reasoning.plan_path(starts[0], transition_weights, model.n_clones)
    print("states", state_seq)
    print("obs", obs_seq)
    print("actions", action_seq)
    plt.show()


if __name__ == "__main__":
    load_default_context(retrain_models=False)
    targets = [42]
    starts = [52]
    transition_weights = plot_reasoning_then_planning(targets, starts)
    show_graph_and_plan(starts, transition_weights)


