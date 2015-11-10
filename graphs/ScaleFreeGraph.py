__author__ = "nikos.daniilidis"
""" The idea to implement the Graph class as a dictionary, and the Edge class as a tuple
is based on: http://greenteapress.com/complexity
The book thinkcomplexity was used as a guide in developing most of the remaining methods.
"""
from RandomGraph import RandomGraph, Edge
from Graph import Vertex
from random import randint, choice
from GraphWorld import CircleLayout, GraphWorld
import matplotlib.pyplot as plt
from numpy import histogram
from time import sleep

class ScaleFreeGraph(RandomGraph):
    """A RandomGraph class with a method to create a scale free graph following the procedure 
    of Barabasi and Albert."""
    def grow_by_edges(self, m, t):
        """Start with a graph and add n = m*t edges. At each time step add one vertex, connected by m
        edges to existing vertices. The probability of connecting a new vertex to an existing one is 
        proportional to the degree of the existing vertex. The new vertices are numbered A0, ... At-1."""
        for ii in range(t):
            degrees = [(v, len(self.out_edges(v))) for v in self.vertices()]
            sum_deg = max(1, sum([tt[1] for tt in degrees])) #2 * len(self.edges()) if undirected graph             
            w = Vertex("A"+str(ii))
            n_list = [choice(range(sum_deg)) for _ in range(m)] # m random numbers with replacement
            for n in n_list:
                kk, acc = 0, 1
                while kk < len(degrees)-1:
                    if n >= acc:
                        kk += 1
                        acc += degrees[kk][1]
                    else:
                        break  
                v = degrees[kk][0]
                self.add_vertex(w)
                self.add_edge(Edge(v, w))
                #print self.edges()           
            
    def grow_by_vertices(self, m, t):
        """Start with a graph and add n = m*t edges. The vertices are added by groups of m in t steps.
        At each step, the probability of connecting a new vertex to an existing one is proportional
        to the degree of the existing vertex. The new vertices are numbered A0, ... An-1."""
        cnt = 0
        for ii in range(t):
            degrees = [(v, len(self.out_edges(v))) for v in self.vertices()]
            sum_deg = max(1, sum([tt[1] for tt in degrees])) #2 * len(self.edges()) if undirected graph 
            for jj in range(m):
                w = Vertex("A"+str(cnt))
                n = randint(1,sum_deg)
                kk, acc = 0, 1
                while kk < len(degrees)-1:
                    if n >= acc:
                        kk += 1
                        acc += degrees[kk][1]
                    else:
                        break  
                v = degrees[kk][0]
                self.add_vertex(w)
                self.add_edge(Edge(v, w))
                #print self.edges()
                cnt += 1
            
    def connectivity(self):
        """Return a list of tuples containing the degrees k of vetices in the graph and 
        the probabilies of a node with degree k."""
        degrees = [len(self.out_edges(v)) for v in self.vertices()]
        bins = [d+0.5 for d in range(max(degrees))]
        bins.append(max(degrees) + 1.5)
        hist, edges = histogram(degrees, bins)
        return zip(bins, list(hist))
            
if __name__=="__main__":
    if 0: 
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        vs = []
        for v in range(10):
            vs.append(Vertex(alphabet[v]))   
        g = ScaleFreeGraph(vs)
        g.add_regular_edges(2)
        
        # draw the graph
        gw = GraphWorld()
        layout = CircleLayout(g)
        gw.show_graph(g, layout)
        gw.mainloop()
        
        # grow one round
        g.grow_by_edges(3,500)
        
        # draw the graph
        gw = GraphWorld()
        layout = CircleLayout(g)
        gw.show_graph(g, layout)
        gw.mainloop()
        
        
    if 1: 
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        vs = []
        for v in range(10):
            vs.append(Vertex(alphabet[v]))   
        g = ScaleFreeGraph(vs)
        g.add_regular_edges(2)
        
        # grow 
        g.grow_by_vertices(3,20000)

        # log-log plot of connectivity vs degree
        tp = g.connectivity()
        degs, counts = zip(*tp)
        plt.loglog(degs, counts)
        plt.show()
        
        
    