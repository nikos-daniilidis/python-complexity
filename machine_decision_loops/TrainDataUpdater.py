import numpy as np

__author__ = "nikos.daniilidis"


class TrainDataUpdater:
    """
    Update old data with new incoming data and return a fixed number of rows
    """
    def __init__(self, num_events=100):
        """
        :param num_events: Int. The number of rows returned after an update
        :return: Noting
        """
        assert num_events > 0
        assert isinstance(num_events, int)
        self.num_events = num_events

    def update(self, xold, yold, xnew, ynew):
        """
        Take in the old data and update them with new incoming data
        :param xold: numpy array. Old events.
        :param yold: numpy array. Old labels.
        :param xnew: numpy array. New events.
        :param ynew: numpy array. New labels
        :return:
        """
        if xold is None or xnew is None:
            return xnew, ynew
        else:
            assert xold.shape[1] == xnew.shape[1]
            assert xold.shape[0] == yold.shape[0]
            if xnew.shape[0] == 0:
                return xold, yold
            else:
                return np.vstack((xold, xnew))[-self.num_events::], np.hstack((yold, ynew))[-self.num_events::]
