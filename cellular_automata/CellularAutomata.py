import numpy as np
import matplotlib.pyplot as plt

__author__ = "nikos.daniilidis"
# Inspired by Green Tea Press: thinkcomplexity


class BaseCA(object):
    def __init__(self, rule, time_range=100, width=201):
        """
        Base Cellular Automaton class. The object stores the state of the automaton for
        a time up to time_range and predetermined width as a numpy array.
        No implementations for start_single and start_random
        :param rule: The rule of the automaton
        :param time_range: Maximum time range to use while evolving the automaton (in pxels)
        :param width: Width of the automaton (in pixels)
        :return: Nothing
        """
        self.rule = rule
        self.table = self.__make_table()
        self.table_dict = {ix: self.table[ix] for ix in range(len(self.table))}
        self.time_range = time_range
        self.width = width
        self.array = np.zeros((time_range, width), dtype=np.int8)
        self.next = 0

    def get_time_range(self):
        return self.time_range

    def __make_table(self):
        """
        Create a list of length 8 describing the rule of the automaton.
        :return: list of int. Element 0 is the evolution of 000,
                            element 7 is the evolution of 111.
        """
        s = bin(self.rule).split('b')[1]
        s = '0' * (8 - len(s)) + s
        return [int(e) for e in s][::-1]

    def slow_step(self):
        """
        Take a single step in the evolution of the automaton. Slow implementation.
        :return: Nothing
        """
        i = self.next
        self.next = (self.next + 1) % self.time_range

        a = self.array
        t = self.table
        for j in range(1, self.width - 1):
            a[i, j] = t[a[i-1, j-1] + 2*a[i-1, j] + 4*a[i-1, j+1]]

        self.array = a

    def step(self):
        """
        Take a single step in the evolution of the automaton. Fast implementation.
        :return: Nothing
        """
        i = self.next
        self.next = (self.next + 1) % self.time_range

        r = self.array[i-1, ].flatten()
        rl = np.roll(r, -1)
        rl[-1] = 0
        rr = np.roll(r, 1)
        rr[0] = 0
        ix = rr + 2 * r + 4 * rl
        new = np.copy(ix)
        for j, v in self.table_dict.iteritems():
            new[ix == j] = v
        self.array[i, ] = new

    def cycle(self):
        if self.next != 1:
            raise ValueError("cycle() method called on automaton with more/less than one line.")
        else:
            for stp in range(self.time_range-1):
                self.step()

    def show(self):
        """
        Plot the state of the automaton.
        :return: Nothing
        """
        plt.imshow(self.array, cmap='Greys', interpolation='none')
        plt.title("Rule %d" % self.rule)
        plt.axis("off")
        plt.show()


class SingleCA(BaseCA):
    def __init__(self, rule, time_range, width, seed=42):
        """
        Create a basic Cellular Automaton. The object stores the state of the automaton for
        a time up to time_range and predetermined width as a numpy array.
        :param time_range: Maximum time range to use while evolving the automaton (in pxels)
        :param width: Width of the automaton (in pixels)
        :return: Nothing
        """
        super(SingleCA, self).__init__(rule, time_range, width)
        self.seed = seed

    def start_single(self):
        """
        Initialize the CA with a single cell in the middle at time 0.
        :return: Nothing
        """
        self.array[0, self.width / 2] = 1
        self.next = (self.next + 1) % self.time_range

    def start_random(self):
        """
        Initialize the CA with a random set of cells at time 0.
        :return: Nothing
        """
        np.random.seed(self.seed)
        self.array[0, ] = np.random.random_integers(0, 1, self.width)
        self.next = (self.next + 1) % self.time_range


class EnsembleCA(BaseCA):
    def __init__(self, rule, time_range, num_blocks=2, bounded=False, seed=42):
        """
        Create an ensemble of cellular automata. The ensemble consists of multiple blocks, each block
        containing on automaton with the same rule. The blocks are separated by (2*t-2) which means they
        will not interact with each other as long as the time range (t) and width of each block (w)
        satisfy w >= (3*t-2)
        :param rule: rule of the cellular automaton
        :param time_range: maximum time evolution which can be stored. If you take more time steps than
                        time_range - 1, the automaton rolls up and continues from time zero
        :param num_blocks: Number of blocks in the ensemble
        :return: Nothing
        """
        self.bounded = bounded
        if bounded:
            self.extra_width = 1
        else:
            self.extra_width = time_range - 1
        self.block_width = time_range + 2 * self.extra_width
        width = self.block_width * num_blocks
        super(EnsembleCA, self).__init__(rule, time_range, width)
        self.num_blocks = num_blocks
        self.row_filter = self.__row_filter()
        self.seed = seed

    def get_block_width(self):
        return self.block_width

    def start_single(self):
        """
        Initialize each block in the ensemble CA  with a single cell in the middle at time 0.
        :return: Nothing
        """
        for b in range(1, self.num_blocks + 1):
            self.array[0, 1 + (2*b-1) * self.block_width / 2] = 1
        self.next = (self.next + 1) % self.time_range

    def __row_filter(self):
        filt = np.zeros(self.width)
        t = self.time_range
        for b in range(self.num_blocks):
            filt[self.extra_width + b * self.block_width:
                 self.extra_width + b * self.block_width + t, ] = np.ones(t)
        return filt

    def start_random(self):
        """
        Initialize each block in the ensemble CA with a random set of cells at time 0.
        :return: Nothing
        """
        filt = self.__row_filter()
        np.random.seed(self.seed)
        self.array[0, ] = np.random.random_integers(0, 1, self.width) * filt
        self.next = (self.next + 1) % self.time_range

    def step(self):
        super(EnsembleCA, self).step()
        if self.bounded:
            self.array[self.next-1, :] *= self.row_filter

    def show_overlayed(self):
        """
        Plot an overlay of the automata from all blocks, collapsed into one block.
        :return: Nothig
        """
        eyes = np.tile(np.eye(self.block_width), (self.num_blocks, 1))
        stacked = np.dot(self.array, eyes)
        fig = plt.imshow(stacked, cmap='Blues', interpolation='none')
        plt.title("Rule %d" % self.rule)
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.show()


if __name__ == "__main__":
    ca = EnsembleCA(rule=110, time_range=30, num_blocks=3, bounded=False, seed=444)
    ca.start_single()
    for k in range(29):
        ca.step()
    ca.show()
    #ca.show_overlayed()
    if False:
        for rul in range(256):
            ca = SingleCA(rule=rul, time_range=1000, width=2001)
            ca.start_random()
            for k in range(999):
                ca.step()
            ca.show()
