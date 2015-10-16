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
    """An implementation of hash table as a list of 100 LinearMaps"""
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
    
    def __len__(self):
        return self.n
    
class HashMap(object):
    """An implementation of a hash table as a variable length list of LinearMaps"""
    def __init__(self, n=2):
        self.maps = BetterMap(2)
        self.number = 0
        
    def __len__(self):
        return self.number
                
    def get(self, k):
        return self.maps.get(k)

    def add(self, k, v):
        if self.number == len(self.maps):
            self.resize()
            
        self.maps.add(k, v)
        self.number += 1
        
    def resize(self):
        new_maps = BetterMap(2*self.number)
        
        for m in self.maps.maps:
            for k, v in m.items:
                new_maps.add(k, v)
            
        self.maps = new_maps