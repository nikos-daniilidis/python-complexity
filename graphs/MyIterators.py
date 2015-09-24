import  string

class AllTrue(object):
    """A miniml iterator class that always returns True 
    as next. Example usage: zip('abc', AllTrue())"""
    def next(self):
        return True
    
    def __iter__(self):
        return self
    
    
def letters_forever():
    """Yield the letters of the alphabet, starting at a after every z"""
    while True:
        for l in string.lowercase():
            yield l
            
            
def alphanumeric_forever():
    """Yield a1, ..., z1, a2, ..., z2, ..."""
    cnt = 0
    while True:
        cnt += 1
        for l in string.lowercase():
            yield l + str(cnt)