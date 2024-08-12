import numpy as np
from chmm_actions import CHMM, forwardE, datagen_structured_obs_room
import matplotlib.pyplot as plt
import igraph
import matplotlib
from matplotlib import cm, colors
import matplotlib.image as mpimg
import sys, os, pickle
import math
from scipy.special import softmax

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
    def plot_graph(
        chmm, x, a, output_file, cmap=cm.Spectral, multiple_episodes=False, vertex_size=30, rotation = 0
    ):
        n_clones = chmm.n_clones
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
            layout=[Plotting.rotate(x, y, 90 * rotation) for x, y in g.layout("kamada_kawai")],
            vertex_color=colors,
            vertex_label=v,
            vertex_size=vertex_size,
            margin=50,
        )

        return out

    @staticmethod
    def plot_heat_map(
        chmm, x, a, V, output_file, multiple_episodes=False, vertex_size=30, rotation = 0
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

        V_disp_norm = (V_displayed_nodes - np.min(V_displayed_nodes)) / (np.max(V_displayed_nodes) - np.min(V_displayed_nodes))

        # colormap = cm.get_cmap('viridis')
        colormap = matplotlib.colormaps['viridis']
        colors = colormap(V_disp_norm)
        colors = [tuple(c) for c in colors]

        g = igraph.Graph.Adjacency((A > 0).tolist())

        out = igraph.plot(
            g,
            output_file,
            layout=[Plotting.rotate(x, y, 90 * rotation) for x, y in g.layout("kamada_kawai")],
            vertex_color=colors,
            vertex_label=v,
            vertex_size=vertex_size,
            margin=50,
        )

        return out
    
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
        s = sum(V)
        v_new = np.zeros(V.shape)
        for i in range(T.shape[0]):
            v_new += V @ T[i]
        r = v_new + V_init
        return (s+1) * r / sum(r)
    
    @staticmethod
    def backwardV(V, V_init, T):
        s = sum(V)
        v_new = np.zeros(V.shape)
        for i in range(T.shape[0]):
            v_new += V @ T[i].T
        r = v_new
        return (s+1) * r / sum(r) + V_init

    @staticmethod 
    def forward_search(chmm, x, n_iters=10):
        mess_fwd = Reasoning.get_mess_fwd(chmm, x, pseudocount_E=0.1)
        V_init = mess_fwd[-1]
        V = V_init
        for i in range(n_iters):
            V = Reasoning.updateV(V, V_init, chmm.C)
    


