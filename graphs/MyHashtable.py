class LinearMap(object):
    """A simple implementation of a hash table as a list of tuples"""
    def __init__(self):
        self.items = []
        
    def add(self, k, v):
        self.items.append()
        
    def get(self, k):
        for ky, val in self.items:
            if ky == k:
                return val
        raise KeyError
    
    
class BetterMap(object):
    """An implementation of hash table as 100 LinearMaps"""
    def __init__(self, n=100):
        self.maps = []
        for ii in range(n):
            self.maps.append(LinearMap())
            
    def find_map(self, k):
        index = hash(k) % len(self.maps)
        return self.maps[index]
    
    def add(self, k, v):
        m = self.find_map(k)
        m.add(k, v)
        
    def get(self, k):
        m =self.find_map(k)
        return m.get(k)
    
    
class HashMap(object):
    """An implementation of a hash table as a variable mumber of LinearMaps"""
    def __init__(self, n=2):
        self.maps = BetterMap(2)
        self.number = 0
                
    def add(self, k, v):
        m = self.find_map(k)
        lf len(m)