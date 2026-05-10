import math
import os
import pickle
import sys
from queue import Queue

import igraph
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

from chmm_actions import CHMM, datagen_structured_obs_room, forwardE
from CSCG_helpers import Plotting, Reasoning


class Navigator():
    def __init__(self, chmm, target_nodes = [], target_obs = []):
        if not target_nodes and not target_obs:
            raise Exception('Need to specify at least one target node or observation')

        self.chmm = chmm
        self.n_clones = chmm.n_clones
        self.total_clones = sum(self.n_clones)

        v = np.zeros(self.total_clones)
        for target_node in target_nodes:
            v[target_nodes] = 1
        for o in target_obs:
            start_ = sum(self.n_clones[:o])
            end_ = start_ + self.n_clones[o]
            v[start_:end_] = 1

        self.target_nodes = target_nodes
        self.target_obs = target_obs
        
        T = chmm.T
        while sum(v) > .01:
            v, T = Reasoning.STP(v, T)
        self.T = T

        self.x = np.array([], dtype=np.int64)
        self.a = np.array([], dtype=np.int64)

        self.planned_states = Queue()
        self.planned_obs = Queue()
        self.planned_actions = Queue()

        self.found_target = False
        self.on_path = False

    @staticmethod
    def enque_list(lst, q):
        for e in lst:
            q.put(e)

    @staticmethod
    def empty_queue(q):
        while not q.empty():
            q.get()

    def step(self, o, threshold=.95):
        if self.found_target:
            return -1

        print('on path', self.on_path)
        self.x = np.append(self.x, int(o))

        # termination check
        if o in self.target_obs:
            self.found_target = True
            return -1

        if self.on_path:
            if not self.planned_obs.empty():
                expected_o = self.planned_obs.get()
                if o == expected_o:
                    self.planned_states.get()
                    if not self.planned_actions.empty():
                        action = self.planned_actions.get()
                        self.a = np.append(self.a, int(action))
                        return action
                    else:
                        self.found_target = True
                        return -1
                else:
                    Navigator.empty_queue(self.planned_states)
                    Navigator.empty_queue(self.planned_obs)
                    Navigator.empty_queue(self.planned_actions)
                    self.on_path = False
            else:
                self.found_target = True
                return -1

        if not self.on_path:
            lik, states, decode_mess_fwd = self.chmm.jacob_decode(self.x, np.append(self.a, int(0)))

            state = states[-1]

            len_clones = self.n_clones[o]
            probs = decode_mess_fwd[-len_clones:]

            v = np.zeros(self.total_clones)
            start_ = sum(self.n_clones[:o])
            end_ = start_ + len_clones
            v[start_:end_] = probs

            v_norm = v / sum(v)

            if v_norm[state] > threshold:
                # termination check
                if state in self.target_nodes:
                    self.found_target = True
                    return -1

                # Start planning
                state_seq, obs_seq, action_seq = Reasoning.plan_path(state, self.T, self.n_clones)

                Navigator.enque_list(state_seq, self.planned_states)
                Navigator.enque_list(obs_seq, self.planned_obs)
                Navigator.enque_list(action_seq, self.planned_actions)
                
                self.on_path = True
                

                action = self.planned_actions.get()
                self.a = np.append(self.a, int(action))
                return action
            else:
                action = Reasoning.select_action(v, self.T)
                self.a = np.append(self.a, int(action))
                return action
        





