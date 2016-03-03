import numpy as np
from CellularAutomata import EnsembleCA

__author__ = "nikos.daniilidis"


class EnigmaticAgent(EnsembleCA):
    def row_bits_slow(self, rows=[0]):
        t = self.time_range
        result = np.zeros((self.num_blocks, len(rows) * (3*t-2)))
        for b in range(self.num_blocks):
            x = np.zeros((1, len(rows) * (3*t-2)))
            for row in rows:
                x[0, row*(3*t-2):(row+1) * (3*t-2)] = \
                    self.array[row, b*(3*t-2):(b+1)*(3*t-2)]
            result[b, ] = x
        return result

    def row_bits(self, rows=[0]):
        b = self.num_blocks
        t = self.time_range
        l = np.zeros((b, len(rows)*(3*t-2)))
        for ix, row in enumerate(rows):
            l[:, ix*(3*t-2):(ix+1)*(3*t-2)] = self.array[row, :].reshape(b, 3*t-2)
        return l

    def row_sums(self, rows=[0]):
        r = self.row_bits(rows)
        return np.sum(r, axis=1)

    def column_bits(self, columns=[0], exclude_last=True):
        offset = int(exclude_last)
        b = self.num_blocks
        t = self.time_range
        a = np.vstack(np.split(self.array, b, axis=1))
        l = np.zeros((b, len(columns)*t))
        for ix, col in enumerate(columns):
            l[:, ix*t:(ix+1)*t] = np.transpose(a[:, col]).reshape(b, t)
        return l[:, :len(columns)*t-offset]

    def column_sums(self, columns=[0], exclude_last=True):
        c = self.column_bits(columns=columns, exclude_last=exclude_last)
        return np.sum(c, axis=1)

    def output_sum(self):
        t = self.time_range
        return self.row_bits(rows=[t-1]).sum(axis=1)

    def output_balance(self):
        t = self.time_range
        result = (np.greater(self.output_sum(), (3*t-2)/2.))
        return result

    def output_parity(self):
        return self.output_sum() % 2




if __name__=="__main__":
    ea = EnigmaticAgent(rule=30, time_range=6, num_blocks=10)
    ea.start_random()
    for k in range(5):
        ea.step()
    rb = ea.row_bits(rows=[0])
    print "row bits\n", rb
    rs = ea.row_sums(rows=[0])
    print "row sums", rs
    cb = ea.column_bits(columns=[0, 1, 2, 3, 4, 5, 6], exclude_last=False)
    print "column_bits", cb
    osm = ea.output_sum()
    print "output sum", osm
    ob = ea.output_balance()
    print "output balance", ob
    ea.show()
    print "output parity", ea.output_parity()
