class Base(object):
    def __init__(self, a):
        self.a = a

class First(Base):
    def __init__(self, a, b):
        super(First, self).__init__(a)
        self.b = b

    def souma(self):
        return self.a + self.b

if __name__ == "__main__":
    f = First(1, -2)
    print f.souma()
