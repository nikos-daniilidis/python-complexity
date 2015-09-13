__author__ = 'nikos.daniilidis'

from Graph import Graph, Edge, Vertex
from random import sample

class RandomGraph(Graph):
    def add_random_edges(self, p):
        """Start with an edgeless graph and add edges at random
        so that there is probability p that there is an edge
        between any two nodes"""
        vs = self.vertices()
        n = len(vs)
        combinations = n * (n-1) / 2
        num_edges = int(round(p * combinations))
        rand_edges = sample(range(0, combinations), num_edges)
        cnt = 0
        for ii, v in enumerate(vs):
            for jj, w in enumerate(vs):
                if ii < jj:
                    if cnt in rand_edges:
                        self.add_edge(Edge(v, w))
                    cnt += 1