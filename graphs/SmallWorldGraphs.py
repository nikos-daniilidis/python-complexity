__author__ = "nikos.daniilidis"
""" The idea to implement the Graph class as a dictionary, and the Edge class as a tuple
is based on: http://greenteapress.com/complexity
The book thinkcomplexity was used as a guide in developing most of the remaining methods.
"""
from RandomGraph import RandomGraph, Edge
from Graph import Vertex
from random import sample
from GraphWorld import CircleLayout, GraphWorld
import matplotlib.pyplot as plt
from time import sleep



class SmallWorldGraph(RandomGraph):
    """A RandomGraph class with a method to create a small world graph using the procedure
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
    
    def path_lengths(self, v):
        """Find the minimum path length between node v and any other node in a graph.
        Uses the Dijkstra shortest path algorithm.
        : param v: Vertex. The vertex from which distances are computed.
        : returns distances: d, dict of {Vertex: int} pairs. 
                            The minimum path lengths from vetices to v in the graph.
        """
        distances = {u: float("inf") for u in self.vertices()}
        distances[v] = 0
        visited = []
        to_visit = [v]
        while len(to_visit) > 0:
            curr = to_visit.pop(0)
            visited.append(curr)
            d = distances[curr]
            nbr = self.out_vertices(curr)
            for u in nbr:
                if u not in visited:
                    distances[u] = min(d+1, distances[u])
                    to_visit.append(u)
        distances.pop(v, None)
        return distances
    
    def mean_path_length(self):
        """Find the average path length between vertices in the graph.
        : return d: float, the average path length"""
        d = 0.
        for v in self.vertices():
            ds = self.path_lengths(v)
            d += sum(ds.values()) / (len(ds.values()) - 1.)
        return d / len(self.vertices())
    
    def all_path_lengths(self):
        """Return a dict of dict with all the path lengths between vertices in the graph. 
        d[u][v] is the minimum path length between the vertices u and v. The graph is undirected, 
        so d[u][v] = d[v][u] (but both are in the dict). 
        : return distances: dict of dict {u: {v: float}}"""
        distances = {}
        for u in self.vertices():
            distances[u] = self.path_lengths(u)
        return distances


if __name__ == "__main__":
    gw = GraphWorld()
    if 0:
        # test SmallWorldGraph
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        vs = []
        for v in range(24):
            vs.append(Vertex(alphabet[v]))   
        g = SmallWorldGraph(vs)
        g.add_regular_edges(5)
        
        # draw the graph
        layout = CircleLayout(g)
        gw.show_graph(g, layout)
        gw.mainloop()
        
        # radnomize
        g.rewire(0.9)
        
        # draw the graph
        gw = GraphWorld()
        gw.show_graph(g, layout)
        gw.mainloop()
    
    if 0:
        # test Cludterring coefficitent 
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
        
    if 0: 
        # test shortest path
        vs = []
        for v in range(50):
            vs.append(Vertex(str(v)))
        g = SmallWorldGraph(vs)
        g.add_regular_edges(10)
        g.rewire(0.3)
        d_avg = g.mean_path_length()
        print "Mean path length: %4.3f" % d_avg
        layout = CircleLayout(g)
        gw.show_graph(g, layout)
        gw.mainloop()
        d = g.path_lengths(vs[0])
        print d
        
    if 0:
        # mean path length vs p 
        mean_path_list = []
        ps = [x/10. for x in range(0, 10)]
        for p in ps:
            vs = []
            for v in range(1000):
                vs.append(Vertex(str(v)))   
            g = SmallWorldGraph(vs)
            g.add_regular_edges(10)
            g.rewire(p)
            mean_path_list.append(g.mean_path_length())
            
        plt.plot(ps, mean_path_list)
        plt.show()
        
    if 0:
        # histogram of path lengths
        p = 0.3
        vs = []
        for v in range(500):
            vs.append(Vertex(str(v)))
        g = SmallWorldGraph(vs)
        g.add_regular_edges(10)
        g.rewire(p)
        all_distance_d = g.all_path_lengths()
        ds = []
        for u, d in all_distance_d.iteritems():
            ds.extend(d.values())
        plt.hist(ds, bins=10)
        plt.show()
        d_avg = g.mean_path_length()
        print "Mean path length: %4.3f" % d_avg
        layout = CircleLayout(g)
        gw.show_graph(g, layout)
        gw.mainloop()
        d = g.path_lengths(vs[0])        
