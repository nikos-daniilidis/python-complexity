from decimal import Decimal
import numpy as np
import pandas as pd

__author__ = "nikos.daniilidis"


class ElectionSimulator:
    """
    Produce simulations of election results with the numbers of votes drawn from a normal distribution.
    """
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
        self.mean_dict = {name: 1./num_candidates for name in names}
        self.variance_dict = {name: 1./np.sqrt(num_candidates) for name in names}

    def update_distribution(self, polls_dict, num_votes):
        """
        Update the gamma distribution parameters for each candidate using poll percentages reported
        prior to the elction.
        :param polls_dict: dict. Keys are candidate names, values are the votes received by each candidate
        :param num_votes: int. Total number of votes in the election.
        :return: Nothing
        """
        for name in self.names:
            self.mean_dict[name] = num_votes * np.mean(polls_dict[name])
            self.variance_dict[name] = num_votes * np.std(polls_dict[name])

    def election_run(self, num_iterations):
        """
        Run num_iterations election simulations. At each iteration, each candidate receives
        a number of votes drawn from a normal distribution with mean and variance
        determined in the class definition.
        :param num_iterations: integer
        :return: a Pandas Dataframe with num_candidates columns and num_iterations rows
        """
        results = np.zeros((self.num_candidates, num_iterations))
        for i, name in enumerate(self.names):
            results[i] = np.random.normal(loc=self.mean_dict[name],
                                          scale=self.variance_dict[name],
                                          size=num_iterations)
        return pd.DataFrame(data=np.around(results.T), index=None, columns=self.names)


class FraudAnalyzer:
    """
    Use an externally supplied function to determine the probability that an election result
    is fraudulent.
    """
    def __init__(self, simulation_df, func):
        """
        Initialize the fraud election detector.
        :param simulation_df: pandas DataFrame. Contains the columns with the votes of each candidate
        :param func: function. Used to generate the closure clos.
                     clos determines if a result is "nice" (i.e. seems fraudulent) based on matching to
                     specified patterns
        :param patterns: tuple of string. Patterns which, combined with digits, determine if a result is "nice"
        :param digits: int. Number of digits by which the decimal point is offset to determine pattern matching
        :return: Nothing
        """
        self.df = simulation_df.div(simulation_df.sum(axis=1), axis=0)
        self.func = func

    def find_nice(self):
        """
        Find the nice rows in the simulation dataframe
        :return: Pandas series with values 1. or 0.
        """
        s = pd.Series(np.ones(self.df.shape[0]))
        for col in self.df.columns:
            s = s.multiply(pd.Series(self.df[col].
                                     values.astype(np.str)).
                           apply(lambda r: float(is_nice(r))))
        self.df['is_nice'] = s
        return None

    def fraction_nice(self):
        """
        Determine the fraction of "nice" results in the simulation DataFrame
        :return: float. The fraction of "nice results"
        """
        return self.df['is_nice'].mean()


def is_nice(x):
    """
    Function to determine if a number is 'nice'. Has hard coded parameters for now.
    :param x: float. A number
    :return: boolean. True if x has the sought out patterns.
    """
    sub = x.split(".")[1][3:5]
    return (sub == "") or (sub == "00") or (sub == "25")


def bayes_inversion(p_res_fair, p_fair, p_res_fraud):
    """
    Determine p(fair|results) given p(results|fair), p(results|fraud), p(fair), and p(fraud)
    :param p_r_fair: float. p(results|fair election)
    :param p_fair: float. p(fair election)
    :param p_res_fraud: float. p(results|fraudulent election)
    :return: p(fair election|results)
    """
    p_fraud = 1 - p_fair
    p_results = p_res_fair * p_fair + p_res_fraud * p_fraud
    return p_res_fair * p_fair / p_results


if __name__ == "__main__":
    nea_dimokratia_election = ElectionSimulator(num_candidates=4,
                                                names=["Meimarakis",
                                                       "Mitsotakis",
                                                       "Tzitzikostas",
                                                       "Georgiadis"])

    nea_dimokratia_election.update_distribution({"Meimarakis": [0.335, 0.41, 0.46],
                                                 "Mitsotakis": [0.313, 0.2, 0.251],
                                                 "Tzitzikostas": [0.227, 0.18, 0.154],
                                                 "Georgiadis": [0.111, 0.17, 0.101]}, 4000) # 404078
    simulation = nea_dimokratia_election.election_run(num_iterations=3000000)
    #print simulation
    #print simulation.apply(lambda row: row/sum(row), axis=1)
    sim_fraud = FraudAnalyzer(simulation, is_nice)
    sim_fraud.find_nice()
    print sim_fraud.fraction_nice()

    if 0:
        test_fraud = FraudAnalyzer(pd.DataFrame({"A": [0.39725, 112, 0.398],
                                                 "B": [0.28525, 353, 0.285],
                                                 "C": [0.20325, 89, 0.203],
                                                 "D": [0.11425, 76, 0.114]},),
                                   is_nice)
        test_fraud.find_nice()
        print test_fraud.df
        print test_fraud.fraction_nice()
        #print __is_nice(np.array([0.39725, 0.28525, 0.20325, 0.11425]))
        #print Decimal(np.mean(sim_fraud.df['is_nice']))
        #print(has_form(0.39800, 00))

