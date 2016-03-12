import numpy as np
from CellularAutomata import EnsembleCA

__author__ = "nikos.daniilidis"


class ComplexAgent(EnsembleCA):
    """
    Complex agent class. A complex agent is a cellular automaton which only exposes a number of `inputs`
    and `outputs` to the outside world (e.g. sums of bits on each row). The complex agent implements a
    non-trivial rule for converting between inputs and outputs. Both inputs and outputs are numbers
    derived from the same cellular automaton, e.g. the sums of elements of specific rows and columns.
    This class implements an ensemble of complex agents, and provides methods to access inputs and
    outputs of the complex agent. Typically, the outputs will be derived from the last row of the
    automaton.
    """
    def __init__(self, rule, time_range, num_blocks=2, bounded=False, seed=42):
        super(ComplexAgent, self).__init__(rule, time_range, num_blocks, bounded, seed)
        if self.bounded:
            self.active_width = self.time_range
        else:
            self.active_width = self.block_width

    def row_bits_slow(self, rows=(0, 1)):
        """
        Return the bits for all elements in rows of the agent (slow implementation for large b)
        :param rows: list of int. The rows of which to return the bits
        :return: numpy array with num_blocks rows and len(rows)*width columns
        """
        result = np.zeros((self.num_blocks, len(rows) * self.block_width))
        for b in range(self.num_blocks):
            x = np.zeros((1, len(rows) * self.block_width))
            for row in rows:
                x[0, row*self.block_width:(row+1) * self.block_width] = \
                    self.array[row, b*self.block_width:(b+1)*self.block_width]
            result[b, ] = x
        return result

    def row_bits(self, rows=(0, 1)):
        """
        Return the bits for all elements in rows of the agent
        :param rows: list of int. The rows of which to return the bits
        :return: numpy array with num_blocks rows and len(rows)*width columns
        """
        b = self.num_blocks
        l = np.zeros((b, len(rows)*self.block_width))
        for ix, row in enumerate(rows):
            l[:, ix*self.block_width:(ix+1)*self.block_width] = self.array[row, :].reshape(b, self.block_width)
        return l

    def row_sums(self, rows=(0, 1)):
        """
        Return the sums of bits for rows of the agent
        :param rows: list of int. The rows of which to return the sums
        :return: numpy array with num_blocks rows and len(rows) columns
        """
        b = self.num_blocks
        rb = self.row_bits(rows)
        rs = np.zeros((b, len(rows)))
        for ix, row in enumerate(rows):
            rs[:, ix] = np.sum(rb[:, ix*self.block_width:(ix+1)*self.block_width], axis=1)
        return rs

    def column_bits(self, columns=(0, 1), exclude_last=True):
        """
        Return the bits for all elements in columns of the agent
        :param columns: list of int. The rows of which to return the bits
        :param exclude_last: boolean. If True, do not return the sum for the last column
        :return: numpy array with num_blocks rows and len(columns)*time (or time-1) columns
        """
        start_offset = int(self.bounded)
        end_offset = int(exclude_last)
        b = self.num_blocks
        t = self.time_range
        a = np.vstack(np.split(self.array, b, axis=1))
        l = np.zeros((b, len(columns)*t))
        for ix, col in enumerate(columns):
            l[:, ix*t:(ix+1)*t] = np.transpose(a[:, col + start_offset]).reshape(b, t)
        return l[:, :len(columns)*t-end_offset]

    def column_sums(self, columns=(0, 1), exclude_last=True):
        """
        Return the sums of bits for columns of the agent
        :param columns: list of int. The indexes of columns to inspect
        :param exclude_last: boolean. If True, exclude te last column
        :return: numpy array of size b*len(columns). The sums for each column in columns
        """
        offset = int(exclude_last)
        b = self.num_blocks
        t = self.time_range
        cb = self.column_bits(columns=columns, exclude_last=exclude_last)
        cs = np.zeros((b, len(columns)))
        for ix, col in enumerate(columns):
            cs[:, ix] = np.sum(cb[:, ix*t:(ix+1)*t-offset], axis=1)
        return cs

    def output_sum(self):
        """
        Return the sum of the bits in the last row of the agent
        :return: numpy array of length b. The sums
        """
        t = self.time_range
        return self.row_bits(rows=[t-1]).sum(axis=1)

    def output_balance(self):
        """
        Return the balance of the bits in the last row of the agent (more than half full or not)
        :return: numpy array of length b. The balances.
        """
        return (np.greater(self.output_sum(), self.active_width/2.)).astype(float)

    def output_parity(self):
        """
        Return the parity of the bits in the last row of the agent (odd or even)
        :return: numpy array of length b. The parities.
        """
        return self.output_sum() % 2


if __name__ == "__main__":
    ca = ComplexAgent(rule=30, time_range=6, num_blocks=10, bounded=True)
    ca.start_random()
    ca.cycle()
    print "test", int(0.3*ca.get_block_width())
    # rb =ca.row_bits(rows=(0, 1))
    # print "row bits\n", rb
    # print "row sums", ca.row_sums(rows=[0, 1, 2, 3])
    # cb = ca.column_bits(columns=[0, 1, 2, 3, 4, 5, 6], exclude_last=False)
    # print "column_bits", cb
    print "column sums", ca.column_sums(columns=[0, 1, 2, 3, 4], exclude_last=False)
    print "output sum", ca.output_sum()
    print "output balance", ca.output_balance()
    print "output parity", ca.output_parity()
    # ca.show()
