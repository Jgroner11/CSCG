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

from chmm_actions import CHMM, datagen_structured_obs_room, forwardE


class Plotting:
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

    @staticmethod
    def rotate(x, y, deg):
        rad = math.radians(deg)
        cos_rad = math.cos(rad)
        sin_rad = math.sin(rad)
        new_x = x * cos_rad - y * sin_rad
        new_y = x * sin_rad + y * cos_rad
        return new_x, new_y
    
    @staticmethod
    def flip(x, y, axis=0):
        if axis == 0:
            return -x, y
        elif axis == 1:
            return x, -y
        else:
            return x, y
    
    @staticmethod
    def plot_graph(
        chmm, x, a, output_file, cmap=cm.Spectral, multiple_episodes=False, vertex_size=30, flip=None, rotation = 0., states=None
    ):
        n_clones = chmm.n_clones
        if states is None:
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

        layout = [Plotting.flip(x, y, flip) for x, y in g.layout("kamada_kawai")]
        layout = [Plotting.rotate(x, y, 90 * rotation) for x, y in layout]

        out = igraph.plot(
            g,
            output_file,
            layout=layout,
            vertex_color=colors,
            vertex_label=v,
            vertex_size=vertex_size,
            margin=50,
        )

        return out

    @staticmethod
    def plot_heat_map(
        chmm,
        x,
        a,
        V,
        output_file,
        multiple_episodes=False,
        vertex_size=30,
        flip=None,
        rotation=0.0,
        transition_weights=None,
        edge_label_mode="none",
        vertex_label_mode="state",
        states=None,
    ):
        # States is a list of which latent node (ie state) is most active at each time step
        if states is None:
            states = chmm.decode(x, a)[1]

        v = np.unique(states)
        if multiple_episodes:
            T = chmm.C[:, v][:, :, v][:-1, 1:, 1:]
            v = v[1:]
        else:
            T = chmm.C[:, v][:, :, v]
        A = T.sum(0)
        norm = A.sum(1, keepdims=True)
        norm[norm == 0] = 1
        A /= norm
        # A is a transition matrix of only the latent nodes (states) that get activated during walk of path

        # V_displayed represents the activity of all the nodes that are present in the A matrix/graph based on the inputted V activity for all the nodes
        V_displayed_nodes = np.zeros(v.shape)
        for i, id in enumerate(v):
            V_displayed_nodes[i] = V[id]

        # print('sum V:', sum(V), 'sum Vdisp:', sum(V_displayed_nodes))

        V_disp_norm = V_displayed_nodes
        value_range = np.max(V_displayed_nodes) - np.min(V_displayed_nodes)
        if value_range > 0:
            V_disp_norm = (V_displayed_nodes - np.min(V_displayed_nodes)) / value_range

        # colormap = cm.get_cmap('viridis')
        colormap = matplotlib.colormaps['viridis']
        vertex_colors = colormap(V_disp_norm)
        vertex_colors = [tuple(c) for c in vertex_colors]

        g = igraph.Graph.Adjacency((A > 0).tolist())

        edge_labels = None
        if transition_weights is not None and edge_label_mode != "none":
            transition_sum = transition_weights.sum(0)
            edge_labels = ["" for _ in g.es]
            for index, edge in enumerate(g.es):
                i = v[edge.source]
                j = v[edge.target]
                if i == j:
                    continue
                if edge_label_mode == "round":
                    edge_labels[index] = str(round(transition_sum[j, i], 2))
                elif edge_label_mode == "int":
                    edge_labels[index] = f" {int(transition_sum[j, i])} "
                else:
                    raise ValueError(f"Unknown edge_label_mode: {edge_label_mode}")

        if vertex_label_mode == "state":
            vertex_labels = v
        elif vertex_label_mode == "value":
            vertex_labels = V_displayed_nodes
        elif vertex_label_mode == "none":
            vertex_labels = None
        else:
            raise ValueError(f"Unknown vertex_label_mode: {vertex_label_mode}")

        layout = [Plotting.flip(x, y, flip) for x, y in g.layout("kamada_kawai")]
        layout = [Plotting.rotate(x, y, 90 * rotation) for x, y in layout]

        out = igraph.plot(
            g,
            output_file,
            layout=layout,
            vertex_color=vertex_colors,
            vertex_label=vertex_labels,
            vertex_size=vertex_size,
            edge_label=edge_labels,
            margin=50,
        )

        return out
    
    @staticmethod
    def plot_room(room, pos=None, t=None):
        room = room.copy()
        fig, ax = plt.subplots()
        text = None
        n_emissions = np.max(room) + 1
        if pos is not None:
            cmap = colors.ListedColormap(Plotting.custom_colors[:n_emissions])
            r, c = pos
            ax.matshow(room, cmap=cmap)
            if t is None:
                ax.set_title(f'current position: ({r},{c})')
            else:
                ax.set_title(f'position at t={t}: ({r},{c})')
            ax.set_xlabel('0: left, 1: right, 2: up, 3: down')
            ASCII_person = "O\n/|\\\n/ \\"
            text = ax.text(c, r, ASCII_person, va='center', ha='center', color='black')
        else:
            cmap = colors.ListedColormap(Plotting.custom_colors[:n_emissions])
            ax.matshow(room, cmap=cmap)
            
        return fig, ax, text

    @staticmethod
    def redraw_room(fig, ax, pos, old_text=None, t=None):

        r, c = pos
        # room[r, c] = n_emissions
        if old_text is not None:
            old_text.remove()
        ASCII_person = "O\n/|\\\n/ \\"
        text = ax.text(c, r, ASCII_person, va='center', ha='center', color='black')
        if t is None:
            ax.set_title(f'current position: ({r},{c})')
        else:
            ax.set_title(f'position at t={t}: ({r},{c})')

        fig.canvas.draw()
        return text
    
    def plot_current_obs(room, pos):
        n_emissions = np.max(room) + 1

        fig, ax = plt.subplots()
        (r,c) = pos
        e = room[r, c]

        one_square = np.zeros((1, 1))
        one_square[0, 0] = e
        cmap = colors.ListedColormap(Plotting.custom_colors[e])

        axim = ax.matshow(one_square, cmap=cmap)
        ax.set_title(f'current observation')
        return axim

    def redraw_current_obs(axim, room, pos):
        (r, c) = pos
        e = room[r, c]
        one_square = np.zeros((1, 1))
        one_square[0, 0] = e

        axim.set_data(one_square)
        new_cmap = colors.ListedColormap(Plotting.custom_colors[e])
        axim.set_cmap(new_cmap)
        plt.draw()      
    
class Reasoning:

    @staticmethod
    def get_mess_fwd(chmm, x, pseudocount=0.0, pseudocount_E=0.0):
        """Compute normalized forward messages for an observation sequence using an explicit emission matrix."""
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
    
    @staticmethod
    def forwardV(V, V_init, T):
        """Propagate activity forward, re-add the seed activity, and normalize total activity growth."""
        s = sum(V)
        v_new = np.zeros(V.shape)
        for i in range(T.shape[0]):
            v_new += V @ T[i]
        r = v_new + V_init
        return (s+1) * r / sum(r)
    
    @staticmethod
    def backwardV(V, V_init, T):
        """Propagate activity backward through transposed transitions, re-add the seed activity, and normalize growth."""
        s = sum(V)
        v_new = np.zeros(V.shape)
        for i in range(T.shape[0]):
            v_new += V @ T[i].T
        r = v_new
        return (s+1) * r / sum(r) + V_init

    @staticmethod 
    def forward_search(chmm, x, n_iters=10):
        """Initialize activity from forward messages and repeatedly propagate it for a fixed number of iterations."""
        mess_fwd = Reasoning.get_mess_fwd(chmm, x, pseudocount_E=0.1)
        V_init = mess_fwd[-1]
        V = V_init
        for i in range(n_iters):
            V = Reasoning.updateV(V, V_init, chmm.C)

    @staticmethod
    def sigmoid(x):
        """Apply the logistic sigmoid transform elementwise."""
        return 1 / (1 + np.exp(-x))

    
    @staticmethod
    def STP(v, T):
        """Propagate activity while depressing traversed transition weights to encode a wavefront."""

        v_ = np.zeros(v.shape)
        for i in range(T.shape[0]):
            v_ += T[i] @ v
        v_ = np.minimum(np.maximum(v_, 0), 1)

        ve = np.tile(v, (len(v), 1)).T

        T_ = np.zeros(T.shape)
        for i in range(T.shape[0]):
            T_[i] = T[i] - ve * T[i].T
            # T_[i][T_[i] < 0] = 0

        return v_, T_

    
    @staticmethod
    def propogate(v, T, v_init):
        """Propagate activity forward through all action transitions and clip values to the unit interval."""
        v_ = np.zeros(v.shape)
        for i in range(T.shape[0]):
            v_ += v @ T[i]
        v_ = np.minimum(np.maximum(v_, 0), 1)

        return v_
    
    @staticmethod
    def select_action(v, T):
        """Choose the action whose transition matrix sends the most positive activity forward."""
        num_actions = T.shape[0]
        action_vector = np.zeros(num_actions)
        for i in range(num_actions):
            action_vector[i] = sum(np.maximum(v @ T[i], 0))
        print('action values:', action_vector)
        return np.argmax(action_vector)
    
    @staticmethod
    def get_obs(x, n_clones):
        """Return the observation index that owns a latent clone state."""
        lower = 0
        
        for i, amt in enumerate(n_clones):
            upper = lower + amt
            if x in range(lower, upper):
                return i
            lower = upper
        return None

    @staticmethod
    def plan_path(x, T, n_clones, termination_threshold = .01, max_depth=50):
        """Greedily follow the strongest transition path from a latent state until the transition strength falls below threshold."""
        state_seq = []
        obs_seq = []
        action_seq = []
        num_actions = T.shape[0]

        T_sum = np.sum(T, axis=0)

        found_target = False
        while not found_target and len(action_seq) < max_depth:
            row = T_sum[x]
            id = np.argmax(row) 
            val = row[id]

            action_vals = np.zeros(num_actions)
            for i in range(num_actions):
                action_vals[i] = T[i][x][id]

            action = int(np.argmax(action_vals))
            x = int(id)

            if val <= termination_threshold:
                found_target = True
            else:
                state_seq.append(x)
                obs_seq.append(Reasoning.get_obs(x, n_clones))
                action_seq.append(action)
   
        return state_seq, obs_seq, action_seq





