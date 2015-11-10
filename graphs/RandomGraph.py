__author__ = "nikos.daniilidis"

""" Based on Code example from Complexity and Computation, a book about
exploring complexity science with Python.  Available free from

http://greenteapress.com/complexity

Copyright 2011 Allen B. Downey.
Distributed under the GNU General Public License at gnu.org/licenses/gpl.html.
"""

from Graph import Graph, Edge
from random import sample

class RandomGraph(Graph):
    def add_random_edges(self, p):
        """Start with an edgeless graph and add edges at random
        so that there is probability p that there is an edge
        between any two nodes. The algorithm selects a fraction 
        p of the set of enumerated possible edges of the graph.
        It then runs two for loops which travel the graph vertices 
        and add an edge if it is in the random sample."""
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