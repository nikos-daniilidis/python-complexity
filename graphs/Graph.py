__author__ = "nikos.daniilidis"
""" The idea to implement the Graph class as a dictionary, and the Edge class as a tuple
is based on: http://greenteapress.com/complexity
The book thinkcomplexity was used as a guide in developing most of the remaining methods.
"""

class Vertex(object):
    """A Vertex is a node in a graph."""

    def __init__(self, label=''):
        self.label = label

    def __repr__(self):
        """Returns a string representation of this object that can
        be evaluated as a Python expression."""
        return 'Vertex(%s)' % repr(self.label)

    __str__ = __repr__
    """The str and repr forms of this object are the same."""


class Edge(tuple):
    """An Edge is a list of two vertices."""

    def __new__(cls, *vs):
        """The Edge constructor takes two vertices."""
        if len(vs) != 2:
            raise ValueError, 'Edges must connect exactly two vertices.'
        return tuple.__new__(cls, vs)

    def __repr__(self):
        """Return a string representation of this object that can
        be evaluated as a Python expression."""
        return 'Edge(%s, %s)' % (repr(self[0]), repr(self[1]))

    __str__ = __repr__
    """The str and repr forms of this object are the same."""


class Graph(dict):
    """A Graph is a dictionary of dictionaries.  The outer
    dictionary maps from a vertex to an inner dictionary.
    The inner dictionary maps from other vertices to edges.
    
    For vertices a and b, graph[a][b] maps
    to the edge that connects a->b, if it exists."""

    def __init__(self, vs=[], es=[]):
        """Creates a new graph.  
        vs: list of vertices;
        es: list of edges.
        """
        for v in vs:
            self.add_vertex(v)
            
        for e in es:
            self.add_edge(e)

    def add_vertex(self, v):
        """Add a vertex to the graph."""
        self[v] = {}

    def add_edge(self, e):
        """Adds and edge to the graph by adding an entry in both directions.

        If there is already an edge connecting these Vertices, the
        new edge replaces it.
        """
        v, w = e
        self[v][w] = e
        self[w][v] = e
        
    def get_edge(self, v, w):
        """Get the edge between v, w, if it exists, None otherwise"""
        try:
            return self[v][w]
        except KeyError:
            return 
        
    def remove_edge(self, e):
        """Remove all references to Edge e from the Graph"""
        try:
            v, w = e
        except (ValueError, TypeError):
            return  # e was not a properly formatted edge
        
        try:
            del self[v][w]
            del self[w][v]
        except KeyError:
            return  # e was not an edge in the graph
    
    def vertices(self):
        """Return the list of vertices in the Graph"""
        return self.keys()
    
    def edges(self):
        """Return the list of edges in the Graph"""
        es = []
        for v in self.keys():
            for w in self[v].keys():
                if self[v][w] not in es:
                    es.append(self[v][w])
        return es
    
    def out_vertices(self, v):
        """Get the list of vertices adjacent to Vertex v"""
        return self[v].keys()

    
    def out_edges(self, v):
        """Get the list of edges connected to Vertex v"""
        return  self[v].values()
    
    def add_all_edges(self):
        """Start with an edgeless Graph and make a complete 
        graph by adding edges between all pairs of vertices"""
        for v in self.vertices():
            for w in self.vertices():
                if v != w:
                    e = Edge(v, w)
                    self.add_edge(e)
    
    def is_edge(self, e):
        """Return true if e is an edge of the graph"""
        return e in self.edges()
                 
    def add_regular_edges(self, d, max_steps=1000):
        """Start with an edgeless Graph and add edges to make it 
        into a Graph of degree d. ix s just a safety precaution to 
        avoid infinite looping"""
        vs = self.vertices()
        n = len(vs)
        if (n * d % 2 == 1) or (d > n-1):
            print 'The requested graph does not exist.'
            return   # if n * d is not even, I cannot construct degree d
        
        ix, ix0, ix1 = 0, 0, 1
        origin  = vs[0]
        while ix0 < n and ix < max_steps:
            running = vs[ix1]
            #print len(self.out_edges(running))
            if len(self.out_edges(origin)) == d:
                ix0 += 1
                origin = vs[ix0]
            if len(self.out_edges(running)) < d and ix0 != ix1:
                #print 'add edge'
                self.add_edge(Edge(origin, running))
            if len(self.out_edges(vs[-1])) == d:
                return
            ix += 1
            ix1 = (ix1 + 1) % n
            #print ix0, ix1
        #print ix
        
    def is_connected(self, safecount=100):
        """Determine if the graph is connected"""
        visited = set([])
        worklist = []
        worklist.append(self.vertices()[0])
        cnt = 0
        while len(worklist) > 0 and cnt < safecount:
            v = worklist.pop()
            #print "adding", v
            visited = visited.union(set([v]))
            worklist = list(set(worklist).
                            union(set(self.out_vertices(v))).
                            difference(set(visited)))
            cnt += 1
        #print len(visited) 
        #print len(self.vertices())
        return len(visited) == len(self.vertices())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    