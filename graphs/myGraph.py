__author__ = "nikos.daniilidis"
"""
Old code. See Graph.py for newest version.
"""


class Graph(dict):
    """Implements a Graph as a dictionary of dictionaries. 
    The outer dictionary maps from a vertex to an inner 
    dictionary. The inner dictionary maps from other vertices 
    to edges.
    
    For vertices a and b, graph[a][b] maps
    to the edge that connects a->b, if it exists."""
    def __init__(self, vs=[], es=[]):
        """create a new graph. 
        vs: a list of vertices;
        es: a list of edges."""
        for v in vs:
            self.add_vertex(v)
            
        for e in es:
            self.add_edge(e)
            
    def add_vertex(self, v):
        """add v to the graph"""
        self[v] = {}
        
    def add_edge(self, e):
        """add e to the graph by adding it in both directions.
        If there is already an edge between these Vertices, the new
        edge replaces it"""
        v, w = e
        self[v][w] = e
        self[w][v] = e
        
        
class Vertex(object):
    def __init__(self, label=''):
        self.label = label
        
    def __repr__(self):
        return 'Vertex(%s)' % repr(self.label)
    
    __str__= __repr__
    
    
class Edge(tuple):
    def __new__(cls, e1, e2):
        return tuple.__new__(cls, (e1, e2))
    
    def __repr__(self):
        return 'Edge(%s, %s)' % (repr(self[0]), repr(self[1]))
    
    __str__ = __repr__