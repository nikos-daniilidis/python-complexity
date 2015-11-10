__author__ = "nikos.daniilidis"

""" Based on Code example from Complexity and Computation, a book about
exploring complexity science with Python.  Available free from

http://greenteapress.com/complexity

Copyright 2011 Allen B. Downey.
Distributed under the GNU General Public License at gnu.org/licenses/gpl.html.
"""
from RandomGraph import RandomGraph, Edge
from Graph import Vertex
from random import sample
from GraphWorld import CircleLayout, GraphWorld
from pprint import pprint
import matplotlib.pyplot as plt


class SmallWorldGraph(RandomGraph):
    """Start with  regular graph and create a small world graph using the method
    described in Watts and Strogatz."""
    def rewire(self,p):
        """Select a fraction p of the edges of the graph and rewire them at random."""
        es = self.edges()
        vs = self.vertices()
        num_edges = int(round(p * len(es)))
        rand_edges = sample(range(0, len(es)), num_edges)
        cnt = 0
        for ii, v in enumerate(vs):
            for jj, w in enumerate(vs):
                if ii < jj:
                    if cnt in rand_edges:
                        self.remove_edge(Edge(v, w))
                    cnt += 1
        for ii in range(0, len(rand_edges)):
            edge_not_added = True
            while edge_not_added:
                es = self.edges()
                rand_pair = sample(vs, 2)
                e = Edge(rand_pair[0], rand_pair[1]) 
                if e in es:
                    pass
                else:
                    self.add_edge(e)
                    edge_not_added = False
                    
    def clustering_coefficient(self):
        """Find the average local clustering coefficitent of the small world graph."""
        es = self.edges()
        local_coeffs = []
        for u in self.vertices():
            N = self.out_vertices(u)
            k = len(self.out_edges(u))
            C = 0.
            for ii, v in enumerate(N):
                for jj, w in enumerate(N):
                    if ii < jj and Edge(v, w) in es:
                        C += 2./ (k * (k-1))
            local_coeffs.append(C)
        return sum(local_coeffs) / len(local_coeffs)



if __name__ == "__main__":
    # test SmallWorldGraph
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    vs = []
    for v in range(24):
        vs.append(Vertex(alphabet[v]))   
    g = SmallWorldGraph(vs)
    g.add_regular_edges(5)
    
    # draw the graph
    gw = GraphWorld()
    layout = CircleLayout(g)
    gw.show_graph(g, layout)
    gw.mainloop()
    
    # radnomize
    g.rewire(0.9)
    
    # draw the graph
    gw = GraphWorld()
    gw.show_graph(g, layout)
    gw.mainloop()
            
    clust_coeff_list = []
    ps = [x/25. for x in range(0, 25)]
    for p in ps:
        vs = []
        for v in range(100):
            vs.append(Vertex(str(v)))   
        g = SmallWorldGraph(vs)
        g.add_regular_edges(10)
        g.rewire(p)
        clust_coeff_list.append(g.clustering_coefficient())
        
    plt.plot(ps, clust_coeff_list)
    plt.show()