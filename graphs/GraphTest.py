__author__ = "nikos.daniilidis"

from Graph import Graph, Vertex, Edge
from GraphWorld import CircleLayout, GraphWorld
from pprint import pprint

if __name__ == "__main__":
    v = Vertex('v')
    w = Vertex('w')
    e = Edge(v,w)
    print e
    
    g = Graph([v, w], [e])
    pprint(g, width=1)
    
    e1 = g.get_edge(v, v)
    e2 = g.get_edge(w, v)
    print e1, e2
    
    e1 = Edge(v, v)
    e2 = Edge(w, w)
    g = Graph([v, w], [e, e1, e2])
    g.remove_edge(Edge(v, v))
    pprint(g, width=1)
    
    print g.vertices()
    
    print g.edges()
    
    u = Vertex('u')
    g = Graph([u, v, w])
    pprint(g, width=1)
    g.add_all_edges()
    pprint(g, width=1)
    
    print g.out_edges(v)
    print g.out_vertices(v)
    
    # test creation of Graph of degree d
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    vs = []
    for v in range(10):
        vs.append(Vertex(alphabet[v]))   
    g = Graph(vs)
    layout = CircleLayout(g)
    g.add_regular_edges(4, 100)
    
    # draw the graph
    gw = GraphWorld()
    gw.show_graph(g, layout)
    gw.mainloop()

    