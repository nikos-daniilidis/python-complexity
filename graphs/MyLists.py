__author__ = "nikos.daniilidis"

""" Based on Code example from Complexity and Computation, a book about
exploring complexity science with Python.  Available free from

http://greenteapress.com/complexity

Copyright 2011 Allen B. Downey.
Distributed under the GNU General Public License at gnu.org/licenses/gpl.html.
"""


class MyNode(object):
    def __init__(self, value=None, nxt=None, prev=None):
        self.value = value
        self.nxt = nxt
        self.prev = prev
        
    def set_next(self, node):
        self.nxt = node
        
    def set_previous(self, node):
        self.prev = node
        
    def __repr__(self):
        return  "=[" + str(self.value) + "]=" 

class MyDoubleList(object):
    def __init__(self):
        self.first = MyNode(value="top")
        self.last = MyNode(value="bottom")
        self.first.set_next(self.last)
        self.last.set_previous(self.first)
        
    def append(self, value):
        n = MyNode(value)
        self.first.nxt.set_previous(n)
        n.set_next(self.first.nxt)
        self.first.set_next(n)
        
    def pop(self):
        if self.first.nxt == self.last:
            return None
        else:
            n = self.last.prev
            n.prev.set_next(self.last)
            self.last.set_previous(self.last.prev)
            return n.value
        
    def __repr__(self):
        n = self.first
        r = ""
        while n is not None:
            r += str(n)
            n = n.nxt
        return r
        
if __name__=="__main__":
    N = MyNode()
    L = MyDoubleList()
    L.append("some value")
    L.append("another one")
    print L
    oth = L.pop()
    print oth 
    print L  