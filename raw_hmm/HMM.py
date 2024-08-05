import numpy as np
import igraph
import os, pickle
from matplotlib import cm, colors, pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def choice(p):
    """choose an element based on a probability distribution"""
    return np.random.choice(len(p), p=p)

def rand_dist(n):
    """create a uniform probaility distribution of size n"""
    p = np.random.rand(n+1)
    p.sort()
    p[0] = 0
    p[n] = 1
    return np.array([p[i+1]-p[i] for i in range(n)])

class MM:
    def __init__(self, n, A, pi=None):
        self.n = n
        if pi is None:
            self.pi = np.ones(n) / n
        else:
            self.pi = pi
        if type(A) is tuple:
            (s, d, allHaveSameDegree) = A
            if s == "uniform"[:len(s)]:
                self.A = MM.uniform_A(n, d, allHaveSameDegree)
            elif s == "random"[:len(s)]:
                self.A = MM.rand_A(n, d, allHaveSameDegree)
            else:
                raise Exception("Invalid input for A")
        else:
            self.A = A
        self.sf = self.compute_state_frequency(10000)

    def state_sequence(self, steps, start = None, include_start = True):
        if start is None:
            q = choice(self.pi)
        elif include_start:
            q = start
        else:
            q = choice(self.A[start])
        
        states = np.zeros(steps, dtype=np.uint8)
        for t in range(steps):
            states[t] = q
            q = choice(self.A[q])
        return states
    
    def compute_state_frequency(self, n_iters):
        sf = np.zeros(self.n)
        for i in self.state_sequence(n_iters):
            sf[i] += 1
        self.sf = sf / n_iters
        return self.sf
    
    def save_model(self, name):
        path = os.path.join('models', f'{name}.pkl')
        with open(path, 'wb') as f: # open a text file
            pickle.dump(self, protocol=5, file=f) # Serializes model object
    
    @staticmethod
    def delete_model(name):
        path = os.path.join('models', f'{name}.pkl')
        os.remove(path)

    @staticmethod
    def load_model(name):
        path = os.path.join('models', f'{name}.pkl')
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def uniform_A(n, degree=None, allHaveSameDegree=True):
        if degree == None:
            return np.ones((n,n)) * (1/n)
        else:
            A = np.zeros((n, n))
            degree = np.clip(degree, 1, n)
            d = degree
            for i in range(n):
                if not allHaveSameDegree:
                    d = np.clip(np.random.randint(2 * degree), 1, n)
                map = np.random.choice(n, d, replace=False)
                if d == 1 and map[0] == i: # Makes sure that if there is only one outgoing edge it is not a self edge
                    new_map = np.random.choice(n, 2, replace=False)
                    if new_map[0] != i:
                        map = [int(new_map[0])]
                    else: 
                        map = [int(new_map[1])]
                for j in range(d):
                    A[i][map[j]] = 1/d
            return A
        
    @staticmethod
    def rand_A(n, degree=None, allHaveSameDegree=True):
        A = np.zeros((n, n))
        if degree is None:
            for i in range(n):
                A[i] = rand_dist(n)
        else:
            degree = np.clip(degree, 1, n)
            d = degree
            for i in range(n):
                if not allHaveSameDegree:
                    d = np.clip(np.random.randint(2 * degree), 1, n)
                p = rand_dist(d)
                map = np.random.choice(n, d, replace=False)
                if d == 1 and map[0] == i: # Makes sure that if there is only one outgoing edge it is not a self edge
                    new_map = np.random.choice(n, 2, replace=False)
                    if new_map[0] != i:
                        map = [int(new_map[0])]
                    else: 
                        map = [int(new_map[1])]
                for j in range(d):
                    A[i][map[j]] = p[j]
        return A

    def save_image(self, img_path="mm.png", label_edges=False, cur_node=None):

        g = igraph.Graph.Adjacency((self.A > 0).tolist())

        if label_edges:
            g.es['label'] = ["" if e.source == e.target else str(round(self.A[e.source, e.target], 2))[1:] for e in g.es]

        g.es['width'] = [4 * self.A[e.source, e.target] for e in g.es]

        node_colors = ['gray' for _ in range(self.n)]

        if cur_node is not None:
            node_colors[cur_node] = 'cyan'

        g.vs['color'] = node_colors

        igraph.plot(
            g,
            target=img_path,
            layout=g.layout("kamada_kawai"),
            vertex_label=np.arange(self.n),
            vertex_size=20,

            margin = 50
        )

    def plot(self, ax=None, label_edges=False, cur_node=None):
        img_path = "imgs\\mm.png"
        self.save_image(img_path, label_edges, cur_node)
        img = Image.open(img_path)
        img_array = np.array(img)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            ax.clear()
        ax.axis('off')
        ax.imshow(img_array)
        plt.show()


    def plot_path(self, n_steps=None):
        batch_size = 10
        img_path = "imgs\\mm.png"
        path = self.state_sequence(batch_size)
        self.save_image(img_path, True, path[0])
        image = mpimg.imread(img_path)
        fig, ax = plt.subplots()
        ax.axis('off')
        img_display = ax.imshow(image)
        i = 1
        def update_image(event):
            """Updates the plot with the next image when the specified key is pressed."""
            nonlocal path, i
            if event.key == 'n':  # 'n' key for next image
                x = path[i % batch_size]
                self.save_image(img_path, True, x)
                new_image = mpimg.imread(img_path)
                img_display.set_data(new_image)
                fig.canvas.draw()
                i += 1
                if i % batch_size == 0:
                    path = self.state_sequence(batch_size, start=x, include_start=False)
                if n_steps is not None and i > n_steps:
                    plt.close(fig)     
        fig.canvas.mpl_connect('key_press_event', update_image)
        plt.show()

class HMM(MM):
    def __init__(self, n, m, A, B, pi=None):
        super().__init__(n, A, pi)
        self.m = m
        if type(B) is tuple:
            (s, d, allHaveSameDegree) = B
            if s == "uniform"[:len(s)]:
                self.B = HMM.uniform_B(n, m, d, allHaveSameDegree)
            elif s == "random"[:len(s)]:
                self.B = HMM.rand_B(n, m, d, allHaveSameDegree)
            else:
                raise Exception("Invalid input for B")
        else:
            self.B = B
        self.of = self.compute_obs_frequency(10000)
        
    def obs_sequence(self, steps, states = None):
        if states is None:
            states = self.state_sequence(steps)
        O = np.zeros(steps, dtype=np.uint8)
        for t in range(steps):
            O[t] = choice(self.B[states[t]])
        return O
        
    def save_image(self, img_path="imgs\\hmm.png", label_edges=False, cur_node=None, cur_obs=None):
        custom_colors = (
            np.array(
                [
                    [85, 35, 157],
                    [253, 252, 144],
                    [114, 245, 144],
                    [151, 38, 20],
                    [239, 142, 192],
                    [214, 134, 48],
                    [140, 194, 250],
                    [72, 160, 162],
                ]
            )
            / 256
        )

        if self.m > len(custom_colors):
            custom_colors = np.vstack((custom_colors, np.random.rand(self.m - len(custom_colors), 3)))        
        
        cmap = colors.ListedColormap(custom_colors[:self.m])

        g = igraph.Graph.Adjacency((self.A > 0).tolist())

        if label_edges:
            g.es['label'] = ["" if e.source == e.target else str(round(self.A[e.source, e.target], 2))[1:] for e in g.es]

        g.es['width'] = [4 * self.A[e.source, e.target] for e in g.es]

        node_colors = ['gray' for _ in range(self.n)]

        if cur_node is not None and cur_obs is not None:
            node_colors[cur_node] = cmap(cur_obs)[:3]

        g.vs['color'] = node_colors

        igraph.plot(
            g,
            target=img_path,
            layout=g.layout("kamada_kawai"),
            vertex_label=np.arange(self.n),
            vertex_size=20,

            margin = 50
        )

    def plot_path(self, n_steps=None):
        batch_size = 10
        img_path = "imgs\\mm.png"
        path = self.state_sequence(batch_size)
        O = self.obs_sequence(batch_size, path)
        self.save_image(img_path, True, path[0], O[0])
        image = mpimg.imread(img_path)
        fig, ax = plt.subplots()
        ax.axis('off')
        img_display = ax.imshow(image)
        print(path[0])
        i = 1
        def update_image(event):
            """Updates the plot with the next image when the specified key is pressed."""
            nonlocal path, O, i
            if event.key == 'n':  # 'n' key for next image
                x = path[i % batch_size]
                o = O[i % batch_size]
                self.save_image(img_path, True, x, o)
                new_image = mpimg.imread(img_path)
                img_display.set_data(new_image)
                fig.canvas.draw()
                print(o)
                i += 1
                if i % batch_size == 0:
                    path = self.state_sequence(batch_size, start=x, include_start=False)
                    O = self.obs_sequence(batch_size, path)
                if n_steps is not None and i > n_steps:
                    plt.close(fig)     
        fig.canvas.mpl_connect('key_press_event', update_image)
        plt.show()
    
    @staticmethod
    def rand_B(n, m, degree=None, allHaveSameDegree=True):
        B = np.zeros((n, m))
        if degree is None:
            for i in range(n):
                B[i] = rand_dist(m)
        else:
            degree = np.clip(degree, 1, m)
            d = degree
            for i in range(n):
                if not allHaveSameDegree:
                    d = np.clip(np.random.randint(2 * degree), 1, m)
                p = rand_dist(d)
                map = np.random.choice(m, d, replace=False)
                for j in range(d):
                    B[i][map[j]] = p[j]
        return B

    @staticmethod
    def uniform_B(n, m, degree=None, allHaveSameDegree=True):
        if degree == None:
            return np.ones((n,m)) * (1/m)
        else:
            B = np.zeros((n, m))
            degree = np.clip(degree, 1, m)
            d = degree
            for i in range(n):
                if not allHaveSameDegree:
                    d = np.clip(np.random.randint(2 * degree), 1, m)
                map = np.random.choice(m, d, replace=False)
                for j in range(d):
                    B[i][map[j]] = 1/d
            return B
    
        
    def compute_obs_frequency(self, n_iters):
        of = np.zeros(self.m)
        for i in self.obs_sequence(n_iters):
            of[i] += 1
        self.of = of / n_iters
        return self.of
    
    def compute_alpha(self, O):
        T = len(O)
        alpha = np.zeros((self.n, T))
        for i in range(self.n):
            alpha[i][0] = self.B[i][O[0]] * self.pi[i]
        for t in range(1, T):
            for i in range(self.n):
                s = 0
                for j in range(self.n):
                    s += alpha[j][t-1] * self.A[j][i]
                result = self.B[i][O[t]] * s
                alpha[i][t] = result
        return alpha
    
    def compute_beta(self, O):
        T = len(O)
        beta = np.ones((self.n, T))
        for t in range(T-2, -1, -1):
            for i in range(self.n):
                s = 0
                for j in range(self.n):
                    s += self.A[i][j] * self.B[j][O[t+1]] * beta[j][t+1]
                beta[i][t] = s
        return beta
    
    def compute_xi(self, O, alpha = None, beta = None):
        T = len(O)
        if alpha is None:
            alpha = self.compute_alpha(O)
        if beta is None:
            beta = self.compute_beta(O)
        xi = np.zeros((self.n, self.n, T))
        for i in range(self.n):
            for j in range(self.n):
                for t in range(T):
                    xi[i][j][t] = alpha[i, t] * self.A[i, j] * self.B[j, t+1] * beta[j, t+1]
        return xi / self.compute_obs_sequence_probability(O)
    
    def compute_gamma(self, O, xi = None):
        T = len(O)
        if xi is None:
            xi = self.compute_xi(O)
        gamma = np.zeros((self.n, T))
        for t in range(T):
            for i in range(self.n):
                s = 0
                for j in range(self.n):
                    s += xi[i, j, t]
                gamma[i][t] = s
        return gamma

    def compute_obs_sequence_probability(self, O, using='alpha'):
        if using == 'alpha':
            alpha = self.compute_alpha(O)
            return sum(alpha[:, len(O)-1])
        elif using == 'beta':
            beta = self.compute_beta(O)
            return sum(self.pi * self.B[:, O[0]] * beta[:, 0])
        raise Exception('must use alpha or beta')
    