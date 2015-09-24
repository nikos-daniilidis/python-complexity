from Graph import Graph, Vertex
from RandomGraph import RandomGraph
from matplotlib import pyplot as plt
import numpy as np


def estimate_probability(n, p, reps=100):
    """Estimate the probability that a random graph with edge probaility
    is connected. The probability is estimated after reps repetirions"""
    n_connected  = 0
    vs = []
    for v in range(n):
        vs.append(Vertex(str(v)))   
    
    for ix in range(reps):
        g = RandomGraph(vs)
        g.add_random_edges(p)
        if g.is_connected():
            n_connected += 1
    
    return 1. * n_connected / reps


def probability_series(n, p_list, reps=100):
    """Estimate the probability that a random graph with edge probaility
    is connected for probability values in the list p_list"""
    p_conn = []
    for p in p_list:
        p_conn.append(estimate_probability(n, p, reps))
    
    return np.array(p_conn)
        

def main():
    n = 100
    p_list = np.arange(0., 1., 0.05)
    p_conn = probability_series(n, p_list)
    
    print p_list
    print p_conn
    plt.figure(figsize=(12, 14))    
  
    # Remove the plot frame lines. They are unnecessary chartjunk.    
    ax = plt.subplot(111)    
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)    
  
    # Ensure that the axis ticks only show up on the bottom and left of the plot.    
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.    
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()    
  
    # Limit the range of the plot to only where the data is.    
    # Avoid unnecessary whitespace.    
    plt.ylim(0., 1.)    
    plt.xlim(0., 1.)    
  
    # Make sure your axis ticks are large enough to be easily read.    
    # You don't want your viewers squinting to read your plot.    
    plt.yticks(np.arange(0, 1, 0.1), [str(x) + "%" for x in np.arange(0, 1, 0.1)], fontsize=14)    
    plt.xticks(fontsize=14)    
  
    # Provide tick lines across the plot to help your viewers trace along    
    # the axis ticks. Make sure that the lines are light and small so they    
    # don't obscure the primary data lines.    
    for y in np.arange(0., 1., 0.1):    
        plt.plot(range(0, 2), [y] * len(range(0, 2)), "--", lw=0.5, color="black", alpha=0.3)
        plt.plot([y] * len(range(0, 2)), range(0, 2), "--", lw=0.5, color="black", alpha=0.3)    
  
        # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
        plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                        labelbottom="on", left="off", right="off", labelleft="on")    
  
    plt.plot(p_list, p_conn,
             lw=2.5, color='magenta') 
    plt.show()
       
    
if __name__=='__main__':
    main()