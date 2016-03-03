import numpy as np
from CellularAutomata import EnsembleCA

__author__ = "nikos.daniilidis"


class ComplexAgent(EnsembleCA):
    """
    Complex agent class. A complex agent is a cellular automaton. The complex agent implements some
    non-trivial rule for converting between inputs and outputs. Both inputs and outputs are numbers
    derived from the same cellular automaton, e.g. the sums of elements of specific rows and columns.
    This class implements an ensemble of complex agents, and provides methods to access inputs and
    outputs of the complex agent. Typically, the outputs will be derived from the last row of the
    automaton.
    """
    def row_bits_slow(self, rows=[0]):
        """
        Return the bits for all elements in rows of the agent (slow implementation for large b)
        :param rows: list of int. The rows of which to return the bits
        :return: numpy array with num_blocks rows and len(rows)*width columns
        """
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
        """
        Return the bits for all elements in rows of the agent
        :param rows: list of int. The rows of which to return the bits
        :return: numpy array with num_blocks rows and len(rows)*width columns
        """
        b = self.num_blocks
        t = self.time_range
        l = np.zeros((b, len(rows)*(3*t-2)))
        for ix, row in enumerate(rows):
            l[:, ix*(3*t-2):(ix+1)*(3*t-2)] = self.array[row, :].reshape(b, 3*t-2)
        return l

    def row_sums(self, rows=[0]):
        """
        Return the sums of bits for rows of the agent
        :param rows: list of int. The rows of which to return the sums
        :return: numpy array with num_blocks rows and len(rows) columns
        """
        b = self.num_blocks
        t = self.time_range
        rb = self.row_bits(rows)
        rs = np.zeros((b, len(rows)))
        for ix, row in enumerate(rows):
            rs[:, ix] = np.sum(rb[:, ix*(3*t-2):(ix+1)*(3*t-2)], axis=1)
        return rs

    def column_bits(self, columns=[0], exclude_last=True):
        """
        Return the bits for all elements in columns of the agent
        :param columns: list of int. The rows of which to return the bits
        :param exclude_last: boolean. If True, do not return the sum for the last column
        :return: numpy array with num_blocks rows and len(columns)*time (or time-1) columns
        """
        offset = int(exclude_last)
        b = self.num_blocks
        t = self.time_range
        a = np.vstack(np.split(self.array, b, axis=1))
        l = np.zeros((b, len(columns)*t))
        for ix, col in enumerate(columns):
            l[:, ix*t:(ix+1)*t] = np.transpose(a[:, col]).reshape(b, t)
        return l[:, :len(columns)*t-offset]

    def column_sums(self, columns=[0], exclude_last=True):
        offset = int(exclude_last)
        b = self.num_blocks
        t = self.time_range
        cb = self.column_bits(columns=columns, exclude_last=exclude_last)
        cs = np.zeros((b, len(columns)))
        for ix, col in enumerate(columns):
            cs[:, ix] = np.sum(cb[:, ix*t:(ix+1)*t-offset], axis=1)
        return cs  # np.sum(c, axis=1)

    def output_sum(self):
        t = self.time_range
        return self.row_bits(rows=[t-1]).sum(axis=1)

    def output_balance(self):
        t = self.time_range
        result = (np.greater(self.output_sum(), (3*t-2)/2.)).astype(float)
        return result

    def output_parity(self):
        return self.output_sum() % 2




if __name__=="__main__":
    ca = ComplexAgent(rule=30, time_range=6, num_blocks=10)
    ca.start_random()
    for k in range(5):
        ca.step()
    #rb =ca.row_bits(rows=[0])
    #print "row bits\n", rb
    print "row sums", ca.row_sums(rows=[0, 1, 2, 3])
    #cb = ca.column_bits(columns=[0, 1, 2, 3, 4, 5, 6], exclude_last=False)
    #print "column_bits", cb
    print "column sums", ca.column_sums(columns=[0, 1, 2, 3, 4])
    print "output sum", ca.output_sum()
    print "output balance", ca.output_balance()
    print "output parity", ca.output_parity()
    ca.show()
