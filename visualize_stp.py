import os
import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from chmm_actions import CHMM, datagen_structured_obs_room
from CSCG_helpers import Plotting, Reasoning
from rooms import GRANULAR_ROOM


DEFAULT_MODEL_NAME = "navigation-granular_room"
DEFAULT_IMAGE_PATH = "figures\\reasoning_fig.png"

chmm = None
x = None
a = None
room = None
cmap = None
_decoded_states_cache = {}


def setup_navigation_model(
    selected_room=None,
    name=DEFAULT_MODEL_NAME,
    retrain_models=False,
    seq_length=5000,
    n_clones=25,
    seed=42,
):
    if selected_room is None:
        selected_room = GRANULAR_ROOM

    n_emissions = np.max(selected_room) + 1
    color_values = np.zeros((n_emissions + 1, 3))
    color_values[:n_emissions] = Plotting.custom_colors[:n_emissions]

    actions, observations, rc = datagen_structured_obs_room(selected_room, length=seq_length)
    clone_counts = np.ones(n_emissions, dtype=np.int64) * n_clones

    model_file = os.path.join("models", f"{name}.pkl")
    if os.path.isfile(model_file) and not retrain_models:
        with open(model_file, "rb") as f:
            model, progression = pickle.load(f)
    else:
        model = CHMM(n_clones=clone_counts, pseudocount=2e-3, x=observations, a=actions, seed=seed)
        progression = model.learn_em_T(observations, actions, n_iter=1000)
        model.pseudocount = 0.0
        model.learn_viterbi_T(observations, actions, n_iter=100)
        with open(model_file, "wb") as f:
            pickle.dump((model, progression), protocol=5, file=f)

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


def _decoded_states(model, observations, actions):
    key = (id(model), id(observations), id(actions))
    if key not in _decoded_states_cache:
        _decoded_states_cache[key] = model.decode(observations, actions)[1]
    return _decoded_states_cache[key]


def _one_hot(nodes, size, value=1.0):
    vector = np.zeros(size)
    for node in nodes:
        vector[node] = value
    return vector


def _add_key_legend(fig, text):
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


def _plot_to_file(model, observations, actions, values, transition_weights, image_path, flip, rotation, states):
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
        states=states,
    )


def _make_activity_figure(
    model, observations, actions, values, transition_weights,
    title, image_path, flip, rotation, key_legend=None, states=None,
):
    _plot_to_file(model, observations, actions, values, transition_weights, image_path, flip, rotation, states)
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.set_title(title)
    img_display = ax.imshow(mpimg.imread(image_path), cmap="viridis")
    plt.colorbar(img_display, ax=ax, orientation="vertical")
    if key_legend is not None:
        _add_key_legend(fig, key_legend)
    return fig, ax, img_display


def _redraw_activity(
    model, observations, actions, values, transition_weights,
    image_path, img_display, ax, title, flip, rotation, states=None,
):
    _plot_to_file(model, observations, actions, values, transition_weights, image_path, flip, rotation, states)
    img_display.set_data(mpimg.imread(image_path))
    ax.set_title(title)
    ax.figure.canvas.draw()


def plot_reasoning(
    targets,
    model=None,
    observations=None,
    actions=None,
    decoded_states=None,
    image_path=DEFAULT_IMAGE_PATH,
    flip=True,
    rotation=0.9,
):
    model, observations, actions = _context(model, observations, actions)
    states = decoded_states if decoded_states is not None else _decoded_states(model, observations, actions)
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
        "Controls: n - step wavefront | q - quit",
        states=states,
    )

    t = 0

    def update_image(event):
        nonlocal values, transition_weights, t
        if event.key == "q":
            plt.close(event.canvas.figure)
            return
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
            states=states,
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
    decoded_states=None,
    image_path=DEFAULT_IMAGE_PATH,
    flip=True,
    rotation=0.9,
):
    model, observations, actions = _context(model, observations, actions)
    states = decoded_states if decoded_states is not None else _decoded_states(model, observations, actions)
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
        "Controls: n - propagate | q - quit",
        states=states,
    )

    print("chosen action", Reasoning.select_action(initial_values, transition_weights), "\n")
    t = 0

    def update_image(event):
        nonlocal values, t
        if event.key == "q":
            plt.close(event.canvas.figure)
            return
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
            states=states,
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
    decoded_states=None,
    image_path=DEFAULT_IMAGE_PATH,
    flip=True,
    rotation=0.9,
    plan_method=Reasoning.STP,
):
    model, observations, actions = _context(model, observations, actions)
    states = decoded_states if decoded_states is not None else _decoded_states(model, observations, actions)
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
        "Controls: n - step | m - switch to planning | q - quit",
        states=states,
    )

    def update_image(event):
        nonlocal mode, t, values, transition_weights, initial_values
        if event.key == "q":
            plt.close(event.canvas.figure)
            return
        if event.key == "n":
            t += 1
            if mode == "Wavefront":
                values, transition_weights = plan_method(values, transition_weights)
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
                states=states,
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
                states=states,
            )

    fig.canvas.mpl_connect("key_press_event", update_image)
    plt.show()
    return transition_weights


def plot_reasoning_then_planning_per_action(
    targets,
    starts,
    model=None,
    observations=None,
    actions=None,
    decoded_states=None,
    image_path=DEFAULT_IMAGE_PATH,
    flip=True,
    rotation=0.9,
):
    ACTION_NAMES = {0: "left", 1: "right", 2: "up", 3: "down"}

    model, observations, actions = _context(model, observations, actions)
    states = decoded_states if decoded_states is not None else _decoded_states(model, observations, actions)
    initial_values = _one_hot(targets, sum(model.n_clones))
    values = initial_values.copy()
    transition_weights = model.T.copy()
    mode = "Wavefront"
    t = 0
    n_actions = transition_weights.shape[0]

    # Compute shared node layout once from the full graph so all panels line up
    base_image = image_path.replace(".png", "-base.png")
    _, shared_layout = Plotting.plot_heat_map(
        model, observations, actions, values,
        output_file=base_image,
        flip=flip, rotation=rotation,
        transition_weights=transition_weights,
        edge_label_mode="int",
        vertex_label_mode="value",
        states=states,
    )

    fig, axes_2d = plt.subplots(2, 2, figsize=(9.6, 10), squeeze=False)
    img_displays = []

    def render_all():
        for a_idx in range(n_actions):
            action_image = image_path.replace(".png", f"-action{a_idx}.png")
            Plotting.plot_heat_map(
                model, observations, actions, values,
                output_file=action_image,
                transition_weights=transition_weights,
                edge_label_mode="round",
                vertex_label_mode="value",
                states=states,
                fixed_layout=shared_layout,
                action=a_idx,
            )
            ax = axes_2d[a_idx // 2, a_idx % 2]
            img_data = mpimg.imread(action_image)
            if a_idx < len(img_displays):
                img_displays[a_idx].set_data(img_data)
            else:
                ax.axis("off")
                img_displays.append(ax.imshow(img_data, cmap="viridis"))
            direction = ACTION_NAMES.get(a_idx, "")
            ax.set_title(f"Action {a_idx} ({direction})\n{mode}: t={t}")
        fig.canvas.draw_idle()

    render_all()
    fig.suptitle("Per-Action View", fontsize=14)
    _add_key_legend(fig, "Controls: n - step | m - switch to planning | q - quit")

    def update_image(event):
        nonlocal mode, t, values, transition_weights, initial_values
        if event.key == "q":
            plt.close(event.canvas.figure)
            return
        if event.key == "n":
            t += 1
            if mode == "Wavefront":
                values, transition_weights = Reasoning.STP(values, transition_weights)
            else:
                values = Reasoning.propogate(values, transition_weights, initial_values)
                print("chosen action", Reasoning.select_action(values, transition_weights), "\n")
            render_all()
        elif mode == "Wavefront" and event.key == "m":
            mode = "Planning"
            t = 0
            initial_values = _one_hot(starts, sum(model.n_clones))
            values = initial_values.copy()
            print("chosen action", Reasoning.select_action(initial_values, transition_weights), "\n")
            render_all()

    fig.canvas.mpl_connect("key_press_event", update_image)
    plt.show()
    return transition_weights


def show_graph_and_plan(starts, transition_weights, name=DEFAULT_MODEL_NAME, flip=True, rotation=0.9):
    model, observations, actions = _context()
    states = _decoded_states(model, observations, actions)
    output_file = os.path.join("figures", f"{name}.png")
    Plotting.plot_graph(
        model,
        observations,
        actions,
        output_file=output_file,
        cmap=cmap,
        flip=flip,
        rotation=rotation,
        states=states,
    )

    image = mpimg.imread(output_file)
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.imshow(image)
    _add_key_legend(fig, "Controls: q - quit")

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
