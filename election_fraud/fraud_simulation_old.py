import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__author__ = "nikos.daniilidis"


class ElectionSimulator:
    def __init__(self, num_candidates=4, names=("A", "B", "C", "D")):
        """
        Set the number of candidates, candidate names, and the prior scales (theta)
        and shapes (k) for the Gamma distribution of each candidate. The prior is
        unbiased (one vote per candidate)
        :param num_candidates: integer
        :param names: list of string
        :return: Nothing
        """
        if len(names) != num_candidates:
            print "You specified only %d candidate names. Changing num_candidates to %d." % (len(names), len(names))
            num_candidates = len(names)
        self.num_candidates = num_candidates
        self.names = names
        self.scale_dict = {name: 1./num_candidates for name in names}
        self.shape_dict = {name: 1. for name in names}

    def update_distribution(self, votes_dict):
        """
        Update the gamma distribution parameters for each candidate using the number of votes
        each one received.
        :param votes_dict: dict. Keys are candidate names, values are the votes received by each
        :return: Nothing
        """
        num_votes = sum([val for val in votes_dict.values()])
        for name in self.names:
            self.scale_dict[name] = 1./(self.num_candidates + num_votes)
            self.shape_dict[name] = 1. + votes_dict[name]

    def election_run(self, num_iterations):
        """
        Run num_iterations election simulations. At each iteration, each candidate receives
        a number of votes drawn from a Gamma distribution with scale (theta) and shape (k)
        determined in the class definition.
         :param num_iterations: integer
        :return: a Pandas Dataframe with num_candidates columns and num_iterations rows
        """
        results = np.zeros((self.num_candidates, num_iterations))
        for i, name in enumerate(self.names):
            results[i] = np.random.gamma(shape=self.shape_dict[name],
                                         scale=self.scale_dict[name],
                                         size=num_iterations)
            print results[i]
        return pd.DataFrame(data=np.around(results.T), index=None, columns=self.names)


if __name__ == "__main__":
    nea_dimokratia_election = ElectionSimulator(num_candidates=4, names=["Meimarakis",
                                                                         "Mitsotakis",
                                                                         "Tzitzikostas",
                                                                         "Georgiadis"])
    nea_dimokratia_election.update_distribution({"Meimarakis": 160823,
                                                 "Mitsotakis": 115162,
                                                 "Tzitzikostas": 82028,
                                                 "Georgiadis": 46065})

