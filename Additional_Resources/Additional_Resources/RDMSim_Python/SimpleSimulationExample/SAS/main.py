import itertools
import functools
import random
import math
import numpy
from enum import Enum
from RDMSimExemplar.Additional_Resources.Additional_Resources.RDMSim_Python.SimpleSimulationExample.rdm_sim import (
    Topology,
)
from RDMSimExemplar.Additional_Resources.Additional_Resources.RDMSim_Python.SimpleSimulationExample.rdm_sim import (
    NetworkProperties as np,
)
from RDMSimExemplar.Additional_Resources.Additional_Resources.RDMSim_Python.SimpleSimulationExample.rdm_sim import (
    Config as config,
)
from typing import *
from scipy.special import kl_div


nfrs = ["MR", "MC", "MP"]


class PomdpType(Enum):
    EXPERT_DEFINED = "EXPERT_DEFINED",
    UNSUPERVISED = "UNSUPERVISED"
    RANDOM_UNIFORM = "RANDOM_UNIFORM"

class POMDP:
    def __init__(self, thresholds, pomdp_type: PomdpType):
        self.pomdp_type = pomdp_type
        self.policies = (
            {}
        )  # {s: (action_fn, action_label)} maps a state (S1, S2, S3) to an action ("MST" or "RT")

        # metrics
        self.actions_chosen = {"RT": 0, "MST": 0}
        self.decision_count = 0  # number of decisions made
        self.perseus_per_decisions = 10  # number of Perseus iterations per decision
        self.states_surprise_log = []  # state transition surprise
        self.novelty_log = []  # novelty by state
        self.time_steps: int = 0

        self.S = {}  # {"state_name": nfr_satisfiability}
        self.A = {}  # {"action_name": action_func, ...}
        self.T = {}  # ((s', s, action), p): transition_num  P(s' | s, a) = T(s', s, a)
        self.O = {}  # (s', action, observation): p
        self.R = {}  # (s: List[bool], action: str): r

        self.prev_state_belief: Optional[str] = None  # previous belief state
        self.prev_action_label = None  # previous action taken
        self.prev_observation_label = None  # previous observation label
        self.current_state_belief: Optional[str] = None  # current belief state

        # transition counts {(s', s, a): int} counts the number of times a transition has occurred
        self.transition_counts = {}
        # observation counts {(s', a, o): int} counts the number of times an observation has occurred
        self.observation_counts = {}

        # belief states
        self.belief_states: Dict[str, float] = {}  # {"S1": p, ...}
        # belief over transition probabilities
        self.belief_transitions: Dict[str, float] = {}  # TODO: populate
        # belief over observations probabilities
        self.belief_observations: Dict[str, float] = {}  # TODO: populate

        self.thresholds = thresholds  # must be in order: active_links, bandwidth_consumption, write_time

        self.total_reward = 0  # total reward gained
        self.state_to_label_mapping = None
        self.V: Dict[str, Optional[None, Callable]] = None
        self.state_counts: Dict[str, int] = None
        self._setup(pomdp_type=pomdp_type)

    def perseus(self, observation_label: Optional[str] = None):
        unimproved_belief_states: List[str] = list(self.S.keys())

        def helper():
            # randomly selected belief state (e.g. "S1")
            rand_belief_identifier: str = random.choice(unimproved_belief_states)
            belief_p = self.belief_states[rand_belief_identifier]

            # remove from unimproved belief states
            del unimproved_belief_states[
                unimproved_belief_states.index(rand_belief_identifier)
            ]

            # find optimal action from bellman backup
            optimal_reward, optimal_action_name = self.bellman_backup(
                state=rand_belief_identifier,
                observation_label=observation_label
            )
            weighted_optimal_reward = belief_p * optimal_reward

            print(f"[Policy] {rand_belief_identifier} -> {optimal_action_name} (weighted reward: {weighted_optimal_reward})")
            if self.V[rand_belief_identifier] is None:
                # initialize V(s) = (opt_reward, opt_act)
                self.V[rand_belief_identifier] = (
                    weighted_optimal_reward,
                    optimal_action_name,
                )
                print(
                    f"[Policy] {rand_belief_identifier} -> {optimal_action_name} (weighted reward: {weighted_optimal_reward})")
            elif weighted_optimal_reward > self.V[rand_belief_identifier][0]:
                # update V(s) = (opt_reward, opt_act)
                self.V[rand_belief_identifier] = (
                    weighted_optimal_reward,
                    optimal_action_name,
                )
                print(
                    f"[Policy] {rand_belief_identifier} -> {optimal_action_name} (weighted reward: {weighted_optimal_reward})")

            # attempt to improve other belief points
            for unimproved_state in unimproved_belief_states.copy():
                # transition probability to transition from improved state to unimproved state
                transition_p = self.T[
                    (unimproved_state, rand_belief_identifier, optimal_action_name)
                ]
                unimproved_state_belief_p = self.belief_states[unimproved_state]
                if transition_p * optimal_reward >= self.V[unimproved_state][0]:
                    # improve other belief
                    self.V[unimproved_state] = (
                        unimproved_state_belief_p * optimal_reward,
                        optimal_action_name,
                    )
                    print(
                        f"[Policy] {unimproved_state} -> {optimal_action_name} (weighted reward: {unimproved_state_belief_p * optimal_reward})")
                    # remove from B_prime
                    del unimproved_belief_states[
                        unimproved_belief_states.index(unimproved_state)
                    ]

            if not unimproved_belief_states:  # all belief states improved
                return
            # continue to improve unimproved belief states
            helper()

        helper()

    def bellman_backup(self, state: str, depth=3, discount_factor=1, observation_label: Optional[str] = None):
        if depth == 0:  # base case: depth reached
            return 1, None

        else:  # recursive case: calculate expected reward
            max_reward, optimal_action = -float("inf"), None
            for a in self.A.keys():
                expected_reward = self.R[(state, a)]

                for s_prime in self._get_connecting_state_labels(state):
                    transition_p = self.T[(s_prime, state, a)]
                    expected_reward += (
                        discount_factor
                        * transition_p
                        * self.bellman_backup(
                        state=s_prime,
                        depth=depth - 1,
                        observation_label=observation_label)[0]
                    )

                    """ Surprise reward
                                        if observation_label:
                        surprise_strength = 10
                        prior = self.belief_states[s_prime]
                        posterior = self._estimate_observation_p(s_label=s_prime, a_label=a, o_label=observation_label)
                        surprise = self._get_bayesian_surprise(prior=prior, posterior=posterior)
                        surprise_reward = discount_factor * transition_p * (surprise_strength * surprise)
                        print(f"{state} -> {s_prime} | {a} - Expected reward: {expected_reward} + surprise {surprise_reward}")
                        expected_reward += surprise_reward
                    """




                # extract optimal action
                if expected_reward > max_reward:
                    max_reward, optimal_action = expected_reward, a

            return max_reward, optimal_action

    def make_decision(self, observations: List[float]):
        """
        Selects an optimal action based on partial state (observations) from the environment
        :param observations: List[float] - [active_links, bandwidth_consumption, write_time] readings
        """
        if self.prev_state_belief and self.prev_action_label and self.prev_observation_label:
            self.observation_counts[(self.prev_state_belief, self.prev_action_label, self.prev_observation_label)] += 1
            if self.pomdp_type == PomdpType.UNSUPERVISED:
                # update observation probability of previous observation
                self._observation_belief_update(
                    prev_s_label=self.prev_state_belief,
                    prev_a_label=self.prev_action_label,
                    prev_o_label=self.prev_observation_label,
                )

        observation_label: str = self._map_observation_to_label(observations)
        # take observations and update current belief state
        self.set_current_belief_state(observations)

        # log novelty of reaching belief state
        self.novelty_log.append(self.get_state_novelty(self.current_state_belief))

        # perform action based on current belief state
        action, action_label = self.policies[self.current_state_belief]

        # calculate reward of taking action in current state R(s, a)
        self.total_reward += self.R[(self.current_state_belief, action_label)]

        action()  # perform action

        # count transition
        if self.prev_state_belief:
            self.transition_counts[
                (self.current_state_belief, self.prev_state_belief, action_label)
            ] += 1

        if self.pomdp_type == PomdpType.UNSUPERVISED:
            # TODO: move to the top
            # update belief states
            self._belief_update(
                s_prime_label=self.current_state_belief,
                observation_label=observation_label,
                action_label=action_label,
            )

            # update belief over transition probabilities
            self._transition_belief_update(
                s_label=self.prev_state_belief,
                s_prime_label=self.current_state_belief,
                action_label=action_label
            )
        self.time_steps += 1  # increment time step

        self.decision_count += 1
        self._perseus_generate_policies(observation_label=observation_label)
        self.prev_action_label = action_label
        self.prev_observation_label = observation_label
        print(self.policies)

    def set_current_belief_state(self, observations: List[float]):
        """
        :param observations: *args - must be in the order: active_links, bandwidth_consumption, write_time
        """
        current_belief_state = tuple(
            self._is_nfr_satisfied(nfrs[i], observations[i], self.thresholds[i])
            for i in range(len(observations))
        )
        print(f"State: {current_belief_state}")

        # map state tuple to label (e.g. "S1")
        current_belief_state_label: str = self.state_to_label_mapping[
            current_belief_state
        ]
        self.state_counts[current_belief_state_label] += 1

        self.prev_state_belief = self.current_state_belief
        self.current_state_belief = current_belief_state_label

        if not self.prev_state_belief:
            self.prev_state_belief = current_belief_state_label

    def get_state_novelty(self, state_label: str):
        empirical_observation_freq = (self.state_counts[state_label] + 1) / (
            self.time_steps + len(self.S.keys())
        )
        return -numpy.log2(empirical_observation_freq)

    def _setup(self, pomdp_type: PomdpType):
        self._generate_states()
        self._generate_state_counts()
        self._generate_state_to_label_mapping()
        self._set_initial_belief_states()
        self._generate_actions()
        self._generate_rewards()
        self._generate_transition_probabilities(pomdp_type)
        self._generate_belief_transitions()
        self._generate_observation_probabilities(pomdp_type)
        self._generate_transition_counts()
        self._generate_observation_counts()

        self.V = {
            s: (0, None) for s in list(self.S.keys())
        }  # {"S1": (opt_reward, opt_act), ...}

        self._perseus_generate_policies()
        assert self.state_to_label_mapping is not None
        assert self.policies is not None

    def _generate_states(self):
        self.S = {
            "S1": (True, True, True),
            "S2": (True, True, False),
            "S3": (True, False, True),
            "S4": (True, False, False),
            "S5": (False, True, True),
            "S6": (False, True, False),
            "S7": (False, False, True),
            "S8": (False, False, False),
        }

    def _generate_actions(self):
        self.A = {"MST": self._set_topology_mst, "RT": self._set_topology_rt}

    def _generate_transition_probabilities(self, pomdp_type: PomdpType):
        """
        T(s', s, a) = P(s' | s;, a)
        """
        T = {
            ("S1", "S1", "MST"): 0.5265,
            ("S2", "S1", "MST"): 0.0585,
            ("S3", "S1", "MST"): 0.2835,
            ("S4", "S1", "MST"): 0.0315,
            ("S5", "S1", "MST"): 0.0585,
            ("S6", "S1", "MST"): 0.0065,
            ("S7", "S1", "MST"): 0.0315,
            ("S8", "S1", "MST"): 0.0035,
            ("S1", "S2", "MST"): 0.4114,
            ("S2", "S2", "MST"): 0.0726,
            ("S3", "S2", "MST"): 0.3366,
            ("S4", "S2", "MST"): 0.0594,
            ("S5", "S2", "MST"): 0.0561,
            ("S6", "S2", "MST"): 0.0099,
            ("S7", "S2", "MST"): 0.0459,
            ("S8", "S2", "MST"): 0.0081,
            ("S1", "S3", "MST"): 0.50784,
            ("S2", "S3", "MST"): 0.04416,
            ("S3", "S3", "MST"): 0.33856,
            ("S4", "S3", "MST"): 0.02944,
            ("S5", "S3", "MST"): 0.04416,
            ("S6", "S3", "MST"): 0.00384,
            ("S7", "S3", "MST"): 0.02944,
            ("S8", "S3", "MST"): 0.00256,
            ("S1", "S4", "MST"): 0.43065,
            ("S2", "S4", "MST"): 0.06435,
            ("S3", "S4", "MST"): 0.35235,
            ("S4", "S4", "MST"): 0.05265,
            ("S5", "S4", "MST"): 0.04785,
            ("S6", "S4", "MST"): 0.00715,
            ("S7", "S4", "MST"): 0.03915,
            ("S8", "S4", "MST"): 0.00585,
            ("S1", "S5", "MST"): 0.4488,
            ("S2", "S5", "MST"): 0.0612,
            ("S3", "S5", "MST"): 0.2992,
            ("S4", "S5", "MST"): 0.0408,
            ("S5", "S5", "MST"): 0.0792,
            ("S6", "S5", "MST"): 0.0108,
            ("S7", "S5", "MST"): 0.0528,
            ("S8", "S5", "MST"): 0.0072,
            ("S1", "S6", "MST"): 0.378895,
            ("S2", "S6", "MST"): 0.077605,
            ("S3", "S6", "MST"): 0.310005,
            ("S4", "S6", "MST"): 0.063495,
            ("S5", "S6", "MST"): 0.077605,
            ("S6", "S6", "MST"): 0.015895,
            ("S7", "S6", "MST"): 0.063495,
            ("S8", "S6", "MST"): 0.013005,
            ("S1", "S7", "MST"): 0.27405,
            ("S2", "S7", "MST"): 0.03045,
            ("S3", "S7", "MST"): 0.50895,
            ("S4", "S7", "MST"): 0.05655,
            ("S5", "S7", "MST"): 0.04095,
            ("S6", "S7", "MST"): 0.00455,
            ("S7", "S7", "MST"): 0.07605,
            ("S8", "S7", "MST"): 0.00845,
            ("S1", "S8", "MST"): 0.325125,
            ("S2", "S8", "MST"): 0.057375,
            ("S3", "S8", "MST"): 0.397375,
            ("S4", "S8", "MST"): 0.070125,
            ("S5", "S8", "MST"): 0.057375,
            ("S6", "S8", "MST"): 0.010125,
            ("S7", "S8", "MST"): 0.070125,
            ("S8", "S8", "MST"): 0.012375,
            ("S1", "S1", "RT"): 0.66994,
            ("S2", "S1", "RT"): 0.14706,
            ("S3", "S1", "RT"): 0.03526,
            ("S4", "S1", "RT"): 0.00774,
            ("S5", "S1", "RT"): 0.10906,
            ("S6", "S1", "RT"): 0.02394,
            ("S7", "S1", "RT"): 0.00574,
            ("S8", "S1", "RT"): 0.00126,
            ("S1", "S2", "RT"): 0.6111,
            ("S2", "S2", "RT"): 0.2037,
            ("S3", "S2", "RT"): 0.0189,
            ("S4", "S2", "RT"): 0.0063,
            ("S5", "S2", "RT"): 0.1164,
            ("S6", "S2", "RT"): 0.0388,
            ("S7", "S2", "RT"): 0.0036,
            ("S8", "S2", "RT"): 0.0012,
            ("S1", "S3", "RT"): 0.687456,
            ("S2", "S3", "RT"): 0.130944,
            ("S3", "S3", "RT"): 0.051744,
            ("S4", "S3", "RT"): 0.009856,
            ("S5", "S3", "RT"): 0.093744,
            ("S6", "S3", "RT"): 0.017856,
            ("S7", "S3", "RT"): 0.007056,
            ("S8", "S3", "RT"): 0.001344,
            ("S1", "S4", "RT"): 0.62909,
            ("S2", "S4", "RT"): 0.18791,
            ("S3", "S4", "RT"): 0.03311,
            ("S4", "S4", "RT"): 0.00989,
            ("S5", "S4", "RT"): 0.10241,
            ("S6", "S4", "RT"): 0.03059,
            ("S7", "S4", "RT"): 0.00539,
            ("S8", "S4", "RT"): 0.00161,
            ("S1", "S5", "RT"): 0.56648,
            ("S2", "S5", "RT"): 0.14162,
            ("S3", "S5", "RT"): 0.01752,
            ("S4", "S5", "RT"): 0.00438,
            ("S5", "S5", "RT"): 0.20952,
            ("S6", "S5", "RT"): 0.05238,
            ("S7", "S5", "RT"): 0.00648,
            ("S8", "S5", "RT"): 0.00162,
            ("S1", "S6", "RT"): 0.513117,
            ("S2", "S6", "RT"): 0.189783,
            ("S3", "S6", "RT"): 0.005183,
            ("S4", "S6", "RT"): 0.001917,
            ("S5", "S6", "RT"): 0.209583,
            ("S6", "S6", "RT"): 0.077517,
            ("S7", "S6", "RT"): 0.002117,
            ("S8", "S6", "RT"): 0.000783,
            ("S1", "S7", "RT"): 0.235125,
            ("S2", "S7", "RT"): 0.192375,
            ("S3", "S7", "RT"): 0.012375,
            ("S4", "S7", "RT"): 0.010125,
            ("S5", "S7", "RT"): 0.287375,
            ("S6", "S7", "RT"): 0.235125,
            ("S7", "S7", "RT"): 0.015125,
            ("S8", "S7", "RT"): 0.012375,
            ("S1", "S8", "RT"): 0.152775,
            ("S2", "S8", "RT"): 0.186725,
            ("S3", "S8", "RT"): 0.004725,
            ("S4", "S8", "RT"): 0.005775,
            ("S5", "S8", "RT"): 0.283725,
            ("S6", "S8", "RT"): 0.346775,
            ("S7", "S8", "RT"): 0.008775,
            ("S8", "S8", "RT"): 0.010725,
        }

        # override expert-provided transition probabilities
        if pomdp_type == pomdp_type.UNSUPERVISED or pomdp_type == pomdp_type.RANDOM_UNIFORM:
            p = 1 / len(self.S.keys())
            T = {transition: p for transition in T.keys()}

        # assert transition probabilities sum to 1
        for s in self.S.keys():
            s_sum_mst = 0
            s_sum_rt = 0

            for s_prime in self.S.keys():
                s_sum_mst += T[(s_prime, s, "MST")]
                s_sum_rt += T[(s_prime, s, "RT")]

            assert math.isclose(s_sum_mst, 1)
            assert math.isclose(s_sum_rt, 1)
        self.T = T

    def _generate_rewards(self):
        """
        Rewards are awarded by taking an action a at a state s
        R(s, a) = r
        """
        self.R = {
            ("S1", "MST"): 47.0,
            ("S2", "MST"): 49.0,
            ("S3", "MST"): 28.0,
            ("S4", "MST"): 40.0,
            ("S5", "MST"): 15.0,
            ("S6", "MST"): 30.0,
            ("S7", "MST"): 13.0,
            ("S8", "MST"): 6.0,
            # RT rewards
            ("S1", "RT"): 47.0,
            ("S2", "RT"): 42.0,
            ("S3", "RT"): 26.0,
            ("S4", "RT"): 23.0,
            ("S5", "RT"): 45.0,
            ("S6", "RT"): 29.0,
            ("S7", "RT"): 14.0,
            ("S8", "RT"): 7.0,
        }

    def _transition_belief_update(
        self, s_label, s_prime_label, action_label
    ):
        prior_p = self.T[(s_prime_label, s_label, action_label)]
        posterior_p = self._estimate_transition_p(
            s_prime_label=s_prime_label, s_label=s_label, action_label=action_label
        )

        # calculate bayesian surprise (between prior / bayes posterior)
        bayesian_surprise = self._get_bayesian_surprise(
            prior=prior_p, posterior=posterior_p
        )

        # use bayesian surprise to calculate adaptation rate
        adaptation_rate = self._get_adaptation_rate(bayesian_surprise)
        posterior_p = (adaptation_rate * posterior_p) + (
            (1 - adaptation_rate) * prior_p
        )
        print(
            f"Transition update:\nT({s_label} -> {s_prime_label} | {action_label}) | prior: {prior_p} | bayes posterior: {posterior_p} | posterior: {posterior_p}"
        )
        self.T[(s_prime_label, s_label, action_label)] = posterior_p

    def _observation_belief_update(self, prev_s_label, prev_a_label, prev_o_label):
        prior_p = self.O[(prev_s_label, prev_a_label, prev_o_label)]
        posterior_p = self._estimate_observation_p(
            s_label=prev_s_label, a_label=prev_a_label, o_label=prev_o_label
        )

        # calculate bayesian surprise (between prior / bayes posterior)
        bayesian_surprise = self._get_bayesian_surprise(
            prior=prior_p, posterior=posterior_p
        )

        # use bayesian surprise to calculate adaptation rate
        adaptation_rate = self._get_adaptation_rate(bayesian_surprise)
        posterior_p = (adaptation_rate * posterior_p) + (
            (1 - adaptation_rate) * prior_p
        )
        print(
            f"Observation update:\nO({prev_s_label} | {prev_a_label}, {prev_o_label}) | prior: {prior_p}  | posterior: {posterior_p}"
        )
        self.O[(prev_s_label, prev_a_label, prev_o_label)] = posterior_p

    def _estimate_transition_p(self, s_prime_label: str, s_label, action_label):
        """
        Estimate the transition probability T(s_prime_label | s_label, action_label) = T(s' | s, a)
        """
        transition_count = self.transition_counts[
            (s_prime_label, s_label, action_label)
        ]
        denominator = 0

        for _s_label in self.S.keys():
            _transition_count = self.transition_counts[
                (_s_label, s_label, action_label)
            ]
            denominator += _transition_count
        posterior = transition_count / (denominator + 1)
        return posterior

    def _estimate_observation_p(self, s_label, a_label, o_label):
        """
        Estimate the observation probability O(s_label | a_label, o_label) = O(s | a, o)
        using Eq 10 (pcbi.1009070.s001-2.pdf)
        """
        observation_count = self.observation_counts[(s_label, a_label, o_label)]
        denominator = 0

        for _s_label in self.S.keys():
            _observation_count = self.observation_counts[(_s_label, a_label, o_label)]
            denominator += _observation_count
        posterior = observation_count / (denominator + 1)
        return posterior


    def _belief_update(self, s_prime_label, observation_label, action_label):
        """
        Update belief states given an observation `observation_label`

        Note: using the Variational Surprise Minimizing Learning (VarSMiLe) algorithm
        TODO: cite
        """
        prior_belief = self.belief_states[s_prime_label]

        # calculate posterior beliefs using Bayes theorem
        posterior_belief = self.get_posterior_belief(
            s_prime_label=s_prime_label,
            action_label=action_label,
            observation_label=observation_label,
        )

        # calculate bayesian surprise between prior/posterior belief
        bayesian_surprise = self._get_bayesian_surprise(
            prior=prior_belief,
            posterior=posterior_belief,
        )

        # log surprise
        print(
            f"prior: {prior_belief} | posterior {posterior_belief} | state surprise: {bayesian_surprise}"
        )
        self.states_surprise_log.append(bayesian_surprise)

        # TODO: adaptation rate is just for updating transition probabilities?
        """
        adaptation_rate = self._get_adaptation_rate(bayesian_surprise)
        posterior_p = (adaptation_rate * posterior_belief) + ((1 - adaptation_rate) * prior_belief)
        """
        self.belief_states[s_prime_label] = posterior_belief
        print(self.belief_states.items())

    def _get_bayesian_surprise(self, prior, posterior):
        """
        Bayesian surprise is defined as the Kullback-Leibler Divergence between the prior and posterior distributions
        :param prior: List[float] - prior beliefs over state
        :param posterior: List[float] - posterior beliefs over state
        :return: float - bayesian surprise (KL divergence between prior/posterior)
        """
        return kl_div(prior, posterior)

    def _get_adaptation_rate(self, bayesian_surprise: float) -> float:
        """
        Surprise-modulated adaptation rate
        pcbi.1009070.s001.pdf
        :param bayesian_surprise: float - surprise between prior and posterior beliefs (given by KL-divergence)
        :return: float - adaptation rate (between 0 and 1)
        """
        m = self._get_volatility()
        adaptation_rate = (m * bayesian_surprise) / (1 + (m * bayesian_surprise))
        print(f"m {m} surprise {bayesian_surprise}")
        print(f"Adaptation rate: {adaptation_rate}")
        assert 0 <= adaptation_rate <= 1
        return adaptation_rate

    def get_posterior_belief(
        self, s_prime_label, action_label, observation_label
    ) -> float:
        """
        Update belief states using the belief update rule
        https://link.springer.com/content/pdf/10.1007/s10458-012-9200-2.pdf
        """
        # observation probability of observing o, given transition s -> s' via action a
        p_observation = self.O[(s_prime_label, action_label, observation_label)]

        # transition probability: p of reaching s' given action a
        p_transition = 0
        for _s_label in self.S.keys():
            p_belief_s = self.belief_states[_s_label]
            p_transition += (
                self.T[(s_prime_label, _s_label, action_label)]
            ) * p_belief_s

        # observation probability: p of observing o
        p_observation_total = 0
        for _s_label in self.S.keys():
            p_belief_s = self.belief_states[_s_label]
            p_observation_s = 0

            for _s_prime_label in self.S.keys():
                _p_transition = self.T[(_s_prime_label, _s_label, action_label)]
                _p_observation = self.O[
                    (_s_prime_label, action_label, observation_label)
                ]
                p_observation_total += _p_transition * _p_observation
            p_observation_total += p_belief_s * p_observation_s

        # posterior belief b(s')
        posterior_belief = (p_observation * p_transition) / p_observation_total
        return posterior_belief

    def _get_volatility(self):
        """
        Get volatility of environment change (pc = probability of change (expert-provided))
        """
        probability_of_change = config.probability_of_change[str(config.scenario)]
        if not probability_of_change:
            raise ValueError(
                f"Probability of change not provided for scenario {config.scenario}"
            )
        return probability_of_change / (1 - probability_of_change)

    def _generate_observation_probabilities(self, pomdp_type: PomdpType):
        """
        Probability of observing observation z given a resulting state s' from action a
        O(s', a, z) = P(z | s', a)
        """
        self.O = {
            ("S1", "MST", "O1"): 0.03984,
            ("S1", "MST", "O2"): 0.00624,
            ("S1", "MST", "O3"): 0.00192,
            ("S1", "MST", "O4"): 0.10624,
            ("S1", "MST", "O5"): 0.01664,
            ("S1", "MST", "O6"): 0.00512,
            ("S1", "MST", "O7"): 0.51792,
            ("S1", "MST", "O8"): 0.08112,
            ("S1", "MST", "O9"): 0.02496,
            ("S1", "MST", "O10"): 0.00747,
            ("S1", "MST", "O11"): 0.00117,
            ("S1", "MST", "O12"): 0.00036,
            ("S1", "MST", "O13"): 0.01992,
            ("S1", "MST", "O14"): 0.00312,
            ("S1", "MST", "O15"): 0.00096,
            ("S1", "MST", "O16"): 0.09711,
            ("S1", "MST", "O17"): 0.01521,
            ("S1", "MST", "O18"): 0.00468,
            ("S1", "MST", "O19"): 0.00249,
            ("S1", "MST", "O20"): 0.00039,
            ("S1", "MST", "O21"): 0.00012,
            ("S1", "MST", "O22"): 0.00664,
            ("S1", "MST", "O23"): 0.00104,
            ("S1", "MST", "O24"): 0.00032,
            ("S1", "MST", "O25"): 0.03237,
            ("S1", "MST", "O26"): 0.00507,
            ("S1", "MST", "O27"): 0.00156,
            ("S2", "MST", "O1"): 0.03216,
            ("S2", "MST", "O2"): 0.01104,
            ("S2", "MST", "O3"): 0.0048,
            ("S2", "MST", "O4"): 0.08576,
            ("S2", "MST", "O5"): 0.02944,
            ("S2", "MST", "O6"): 0.0128,
            ("S2", "MST", "O7"): 0.41808,
            ("S2", "MST", "O8"): 0.14352,
            ("S2", "MST", "O9"): 0.0624,
            ("S2", "MST", "O10"): 0.00603,
            ("S2", "MST", "O11"): 0.00207,
            ("S2", "MST", "O12"): 0.0009,
            ("S2", "MST", "O13"): 0.01608,
            ("S2", "MST", "O14"): 0.00552,
            ("S2", "MST", "O15"): 0.0024,
            ("S2", "MST", "O16"): 0.07839,
            ("S2", "MST", "O17"): 0.02691,
            ("S2", "MST", "O18"): 0.0117,
            ("S2", "MST", "O19"): 0.00201,
            ("S2", "MST", "O20"): 0.00069,
            ("S2", "MST", "O21"): 0.0003,
            ("S2", "MST", "O22"): 0.00536,
            ("S2", "MST", "O23"): 0.00184,
            ("S2", "MST", "O24"): 0.0008,
            ("S2", "MST", "O25"): 0.02613,
            ("S2", "MST", "O26"): 0.00897,
            ("S2", "MST", "O27"): 0.0039,
            ("S3", "MST", "O1"): 0.07968,
            ("S3", "MST", "O2"): 0.01248,
            ("S3", "MST", "O3"): 0.00384,
            ("S3", "MST", "O4"): 0.1328,
            ("S3", "MST", "O5"): 0.0208,
            ("S3", "MST", "O6"): 0.0064,
            ("S3", "MST", "O7"): 0.45152,
            ("S3", "MST", "O8"): 0.07072,
            ("S3", "MST", "O9"): 0.02176,
            ("S3", "MST", "O10"): 0.01494,
            ("S3", "MST", "O11"): 0.00234,
            ("S3", "MST", "O12"): 0.00072,
            ("S3", "MST", "O13"): 0.0249,
            ("S3", "MST", "O14"): 0.0039,
            ("S3", "MST", "O15"): 0.0012,
            ("S3", "MST", "O16"): 0.08466,
            ("S3", "MST", "O17"): 0.01326,
            ("S3", "MST", "O18"): 0.00408,
            ("S3", "MST", "O19"): 0.00498,
            ("S3", "MST", "O20"): 0.00078,
            ("S3", "MST", "O21"): 0.00024,
            ("S3", "MST", "O22"): 0.0083,
            ("S3", "MST", "O23"): 0.0013,
            ("S3", "MST", "O24"): 0.0004,
            ("S3", "MST", "O25"): 0.02822,
            ("S3", "MST", "O26"): 0.00442,
            ("S3", "MST", "O27"): 0.00136,
            ("S4", "MST", "O1"): 0.06432,
            ("S4", "MST", "O2"): 0.02208,
            ("S4", "MST", "O3"): 0.0096,
            ("S4", "MST", "O4"): 0.1072,
            ("S4", "MST", "O5"): 0.0368,
            ("S4", "MST", "O6"): 0.016,
            ("S4", "MST", "O7"): 0.36448,
            ("S4", "MST", "O8"): 0.12512,
            ("S4", "MST", "O9"): 0.0544,
            ("S4", "MST", "O10"): 0.01206,
            ("S4", "MST", "O11"): 0.00414,
            ("S4", "MST", "O12"): 0.0018,
            ("S4", "MST", "O13"): 0.0201,
            ("S4", "MST", "O14"): 0.0069,
            ("S4", "MST", "O15"): 0.003,
            ("S4", "MST", "O16"): 0.06834,
            ("S4", "MST", "O17"): 0.02346,
            ("S4", "MST", "O18"): 0.0102,
            ("S4", "MST", "O19"): 0.00402,
            ("S4", "MST", "O20"): 0.00138,
            ("S4", "MST", "O21"): 0.0006,
            ("S4", "MST", "O22"): 0.0067,
            ("S4", "MST", "O23"): 0.0023,
            ("S4", "MST", "O24"): 0.001,
            ("S4", "MST", "O25"): 0.02278,
            ("S4", "MST", "O26"): 0.00782,
            ("S4", "MST", "O27"): 0.0034,
            ("S5", "MST", "O1"): 0.035856,
            ("S5", "MST", "O2"): 0.005616,
            ("S5", "MST", "O3"): 0.001728,
            ("S5", "MST", "O4"): 0.095616,
            ("S5", "MST", "O5"): 0.014976,
            ("S5", "MST", "O6"): 0.004608,
            ("S5", "MST", "O7"): 0.466128,
            ("S5", "MST", "O8"): 0.073008,
            ("S5", "MST", "O9"): 0.022464,
            ("S5", "MST", "O10"): 0.008964,
            ("S5", "MST", "O11"): 0.001404,
            ("S5", "MST", "O12"): 0.000432,
            ("S5", "MST", "O13"): 0.023904,
            ("S5", "MST", "O14"): 0.003744,
            ("S5", "MST", "O15"): 0.001152,
            ("S5", "MST", "O16"): 0.116532,
            ("S5", "MST", "O17"): 0.018252,
            ("S5", "MST", "O18"): 0.005616,
            ("S5", "MST", "O19"): 0.00498,
            ("S5", "MST", "O20"): 0.00078,
            ("S5", "MST", "O21"): 0.00024,
            ("S5", "MST", "O22"): 0.01328,
            ("S5", "MST", "O23"): 0.00208,
            ("S5", "MST", "O24"): 0.00064,
            ("S5", "MST", "O25"): 0.06474,
            ("S5", "MST", "O26"): 0.01014,
            ("S5", "MST", "O27"): 0.00312,
            ("S6", "MST", "O1"): 0.028944,
            ("S6", "MST", "O2"): 0.009936,
            ("S6", "MST", "O3"): 0.00432,
            ("S6", "MST", "O4"): 0.077184,
            ("S6", "MST", "O5"): 0.026496,
            ("S6", "MST", "O6"): 0.01152,
            ("S6", "MST", "O7"): 0.376272,
            ("S6", "MST", "O8"): 0.129168,
            ("S6", "MST", "O9"): 0.05616,
            ("S6", "MST", "O10"): 0.007236,
            ("S6", "MST", "O11"): 0.002484,
            ("S6", "MST", "O12"): 0.00108,
            ("S6", "MST", "O13"): 0.019296,
            ("S6", "MST", "O14"): 0.006624,
            ("S6", "MST", "O15"): 0.00288,
            ("S6", "MST", "O16"): 0.094068,
            ("S6", "MST", "O17"): 0.032292,
            ("S6", "MST", "O18"): 0.01404,
            ("S6", "MST", "O19"): 0.00402,
            ("S6", "MST", "O20"): 0.00138,
            ("S6", "MST", "O21"): 0.0006,
            ("S6", "MST", "O22"): 0.01072,
            ("S6", "MST", "O23"): 0.00368,
            ("S6", "MST", "O24"): 0.0016,
            ("S6", "MST", "O25"): 0.05226,
            ("S6", "MST", "O26"): 0.01794,
            ("S6", "MST", "O27"): 0.0078,
            ("S7", "MST", "O1"): 0.071712,
            ("S7", "MST", "O2"): 0.011232,
            ("S7", "MST", "O3"): 0.003456,
            ("S7", "MST", "O4"): 0.11952,
            ("S7", "MST", "O5"): 0.01872,
            ("S7", "MST", "O6"): 0.00576,
            ("S7", "MST", "O7"): 0.406368,
            ("S7", "MST", "O8"): 0.063648,
            ("S7", "MST", "O9"): 0.019584,
            ("S7", "MST", "O10"): 0.017928,
            ("S7", "MST", "O11"): 0.002808,
            ("S7", "MST", "O12"): 0.000864,
            ("S7", "MST", "O13"): 0.02988,
            ("S7", "MST", "O14"): 0.00468,
            ("S7", "MST", "O15"): 0.00144,
            ("S7", "MST", "O16"): 0.101592,
            ("S7", "MST", "O17"): 0.015912,
            ("S7", "MST", "O18"): 0.004896,
            ("S7", "MST", "O19"): 0.00996,
            ("S7", "MST", "O20"): 0.00156,
            ("S7", "MST", "O21"): 0.00048,
            ("S7", "MST", "O22"): 0.0166,
            ("S7", "MST", "O23"): 0.0026,
            ("S7", "MST", "O24"): 0.0008,
            ("S7", "MST", "O25"): 0.05644,
            ("S7", "MST", "O26"): 0.00884,
            ("S7", "MST", "O27"): 0.00272,
            ("S8", "MST", "O1"): 0.057888,
            ("S8", "MST", "O2"): 0.019872,
            ("S8", "MST", "O3"): 0.00864,
            ("S8", "MST", "O4"): 0.09648,
            ("S8", "MST", "O5"): 0.03312,
            ("S8", "MST", "O6"): 0.0144,
            ("S8", "MST", "O7"): 0.328032,
            ("S8", "MST", "O8"): 0.112608,
            ("S8", "MST", "O9"): 0.04896,
            ("S8", "MST", "O10"): 0.014472,
            ("S8", "MST", "O11"): 0.004968,
            ("S8", "MST", "O12"): 0.00216,
            ("S8", "MST", "O13"): 0.02412,
            ("S8", "MST", "O14"): 0.00828,
            ("S8", "MST", "O15"): 0.0036,
            ("S8", "MST", "O16"): 0.082008,
            ("S8", "MST", "O17"): 0.028152,
            ("S8", "MST", "O18"): 0.01224,
            ("S8", "MST", "O19"): 0.00804,
            ("S8", "MST", "O20"): 0.00276,
            ("S8", "MST", "O21"): 0.0012,
            ("S8", "MST", "O22"): 0.0134,
            ("S8", "MST", "O23"): 0.0046,
            ("S8", "MST", "O24"): 0.002,
            ("S8", "MST", "O25"): 0.04556,
            ("S8", "MST", "O26"): 0.01564,
            ("S8", "MST", "O27"): 0.0068,
            ("S1", "RT", "O1"): 0.0312,
            ("S1", "RT", "O2"): 0.00585,
            ("S1", "RT", "O3"): 0.00195,
            ("S1", "RT", "O4"): 0.0936,
            ("S1", "RT", "O5"): 0.01755,
            ("S1", "RT", "O6"): 0.00585,
            ("S1", "RT", "O7"): 0.4992,
            ("S1", "RT", "O8"): 0.0936,
            ("S1", "RT", "O9"): 0.0312,
            ("S1", "RT", "O10"): 0.0064,
            ("S1", "RT", "O11"): 0.0012,
            ("S1", "RT", "O12"): 0.0004,
            ("S1", "RT", "O13"): 0.0192,
            ("S1", "RT", "O14"): 0.0036,
            ("S1", "RT", "O15"): 0.0012,
            ("S1", "RT", "O16"): 0.1024,
            ("S1", "RT", "O17"): 0.0192,
            ("S1", "RT", "O18"): 0.0064,
            ("S1", "RT", "O19"): 0.0024,
            ("S1", "RT", "O20"): 0.00045,
            ("S1", "RT", "O21"): 0.00015,
            ("S1", "RT", "O22"): 0.0072,
            ("S1", "RT", "O23"): 0.00135,
            ("S1", "RT", "O24"): 0.00045,
            ("S1", "RT", "O25"): 0.0384,
            ("S1", "RT", "O26"): 0.0072,
            ("S1", "RT", "O27"): 0.0024,
            ("S2", "RT", "O1"): 0.02457,
            ("S2", "RT", "O2"): 0.00975,
            ("S2", "RT", "O3"): 0.00468,
            ("S2", "RT", "O4"): 0.07371,
            ("S2", "RT", "O5"): 0.02925,
            ("S2", "RT", "O6"): 0.01404,
            ("S2", "RT", "O7"): 0.39312,
            ("S2", "RT", "O8"): 0.156,
            ("S2", "RT", "O9"): 0.07488,
            ("S2", "RT", "O10"): 0.00504,
            ("S2", "RT", "O11"): 0.002,
            ("S2", "RT", "O12"): 0.00096,
            ("S2", "RT", "O13"): 0.01512,
            ("S2", "RT", "O14"): 0.006,
            ("S2", "RT", "O15"): 0.00288,
            ("S2", "RT", "O16"): 0.08064,
            ("S2", "RT", "O17"): 0.032,
            ("S2", "RT", "O18"): 0.01536,
            ("S2", "RT", "O19"): 0.00189,
            ("S2", "RT", "O20"): 0.00075,
            ("S2", "RT", "O21"): 0.00036,
            ("S2", "RT", "O22"): 0.00567,
            ("S2", "RT", "O23"): 0.00225,
            ("S2", "RT", "O24"): 0.00108,
            ("S2", "RT", "O25"): 0.03024,
            ("S2", "RT", "O26"): 0.012,
            ("S2", "RT", "O27"): 0.00576,
            ("S3", "RT", "O1"): 0.0624,
            ("S3", "RT", "O2"): 0.0117,
            ("S3", "RT", "O3"): 0.0039,
            ("S3", "RT", "O4"): 0.11232,
            ("S3", "RT", "O5"): 0.02106,
            ("S3", "RT", "O6"): 0.00702,
            ("S3", "RT", "O7"): 0.44928,
            ("S3", "RT", "O8"): 0.08424,
            ("S3", "RT", "O9"): 0.02808,
            ("S3", "RT", "O10"): 0.0128,
            ("S3", "RT", "O11"): 0.0024,
            ("S3", "RT", "O12"): 0.0008,
            ("S3", "RT", "O13"): 0.02304,
            ("S3", "RT", "O14"): 0.00432,
            ("S3", "RT", "O15"): 0.00144,
            ("S3", "RT", "O16"): 0.09216,
            ("S3", "RT", "O17"): 0.01728,
            ("S3", "RT", "O18"): 0.00576,
            ("S3", "RT", "O19"): 0.0048,
            ("S3", "RT", "O20"): 0.0009,
            ("S3", "RT", "O21"): 0.0003,
            ("S3", "RT", "O22"): 0.00864,
            ("S3", "RT", "O23"): 0.00162,
            ("S3", "RT", "O24"): 0.00054,
            ("S3", "RT", "O25"): 0.03456,
            ("S3", "RT", "O26"): 0.00648,
            ("S3", "RT", "O27"): 0.00216,
            ("S4", "RT", "O1"): 0.04914,
            ("S4", "RT", "O2"): 0.0195,
            ("S4", "RT", "O3"): 0.00936,
            ("S4", "RT", "O4"): 0.088452,
            ("S4", "RT", "O5"): 0.0351,
            ("S4", "RT", "O6"): 0.016848,
            ("S4", "RT", "O7"): 0.353808,
            ("S4", "RT", "O8"): 0.1404,
            ("S4", "RT", "O9"): 0.067392,
            ("S4", "RT", "O10"): 0.01008,
            ("S4", "RT", "O11"): 0.004,
            ("S4", "RT", "O12"): 0.00192,
            ("S4", "RT", "O13"): 0.018144,
            ("S4", "RT", "O14"): 0.0072,
            ("S4", "RT", "O15"): 0.003456,
            ("S4", "RT", "O16"): 0.072576,
            ("S4", "RT", "O17"): 0.0288,
            ("S4", "RT", "O18"): 0.013824,
            ("S4", "RT", "O19"): 0.00378,
            ("S4", "RT", "O20"): 0.0015,
            ("S4", "RT", "O21"): 0.00072,
            ("S4", "RT", "O22"): 0.006804,
            ("S4", "RT", "O23"): 0.0027,
            ("S4", "RT", "O24"): 0.001296,
            ("S4", "RT", "O25"): 0.027216,
            ("S4", "RT", "O26"): 0.0108,
            ("S4", "RT", "O27"): 0.005184,
            ("S5", "RT", "O1"): 0.0272,
            ("S5", "RT", "O2"): 0.0051,
            ("S5", "RT", "O3"): 0.0017,
            ("S5", "RT", "O4"): 0.0816,
            ("S5", "RT", "O5"): 0.0153,
            ("S5", "RT", "O6"): 0.0051,
            ("S5", "RT", "O7"): 0.4352,
            ("S5", "RT", "O8"): 0.0816,
            ("S5", "RT", "O9"): 0.0272,
            ("S5", "RT", "O10"): 0.008,
            ("S5", "RT", "O11"): 0.0015,
            ("S5", "RT", "O12"): 0.0005,
            ("S5", "RT", "O13"): 0.024,
            ("S5", "RT", "O14"): 0.0045,
            ("S5", "RT", "O15"): 0.0015,
            ("S5", "RT", "O16"): 0.128,
            ("S5", "RT", "O17"): 0.024,
            ("S5", "RT", "O18"): 0.008,
            ("S5", "RT", "O19"): 0.0048,
            ("S5", "RT", "O20"): 0.0009,
            ("S5", "RT", "O21"): 0.0003,
            ("S5", "RT", "O22"): 0.0144,
            ("S5", "RT", "O23"): 0.0027,
            ("S5", "RT", "O24"): 0.0009,
            ("S5", "RT", "O25"): 0.0768,
            ("S5", "RT", "O26"): 0.0144,
            ("S5", "RT", "O27"): 0.0048,
            ("S6", "RT", "O1"): 0.02142,
            ("S6", "RT", "O2"): 0.0085,
            ("S6", "RT", "O3"): 0.00408,
            ("S6", "RT", "O4"): 0.06426,
            ("S6", "RT", "O5"): 0.0255,
            ("S6", "RT", "O6"): 0.01224,
            ("S6", "RT", "O7"): 0.34272,
            ("S6", "RT", "O8"): 0.136,
            ("S6", "RT", "O9"): 0.06528,
            ("S6", "RT", "O10"): 0.0063,
            ("S6", "RT", "O11"): 0.0025,
            ("S6", "RT", "O12"): 0.0012,
            ("S6", "RT", "O13"): 0.0189,
            ("S6", "RT", "O14"): 0.0075,
            ("S6", "RT", "O15"): 0.0036,
            ("S6", "RT", "O16"): 0.1008,
            ("S6", "RT", "O17"): 0.04,
            ("S6", "RT", "O18"): 0.0192,
            ("S6", "RT", "O19"): 0.00378,
            ("S6", "RT", "O20"): 0.0015,
            ("S6", "RT", "O21"): 0.00072,
            ("S6", "RT", "O22"): 0.01134,
            ("S6", "RT", "O23"): 0.0045,
            ("S6", "RT", "O24"): 0.00216,
            ("S6", "RT", "O25"): 0.06048,
            ("S6", "RT", "O26"): 0.024,
            ("S6", "RT", "O27"): 0.01152,
            ("S7", "RT", "O1"): 0.0544,
            ("S7", "RT", "O2"): 0.0102,
            ("S7", "RT", "O3"): 0.0034,
            ("S7", "RT", "O4"): 0.09792,
            ("S7", "RT", "O5"): 0.01836,
            ("S7", "RT", "O6"): 0.00612,
            ("S7", "RT", "O7"): 0.39168,
            ("S7", "RT", "O8"): 0.07344,
            ("S7", "RT", "O9"): 0.02448,
            ("S7", "RT", "O10"): 0.016,
            ("S7", "RT", "O11"): 0.003,
            ("S7", "RT", "O12"): 0.001,
            ("S7", "RT", "O13"): 0.0288,
            ("S7", "RT", "O14"): 0.0054,
            ("S7", "RT", "O15"): 0.0018,
            ("S7", "RT", "O16"): 0.1152,
            ("S7", "RT", "O17"): 0.0216,
            ("S7", "RT", "O18"): 0.0072,
            ("S7", "RT", "O19"): 0.0096,
            ("S7", "RT", "O20"): 0.0018,
            ("S7", "RT", "O21"): 0.0006,
            ("S7", "RT", "O22"): 0.01728,
            ("S7", "RT", "O23"): 0.00324,
            ("S7", "RT", "O24"): 0.00108,
            ("S7", "RT", "O25"): 0.06912,
            ("S7", "RT", "O26"): 0.01296,
            ("S7", "RT", "O27"): 0.00432,
            ("S8", "RT", "O1"): 0.04284,
            ("S8", "RT", "O2"): 0.017,
            ("S8", "RT", "O3"): 0.00816,
            ("S8", "RT", "O4"): 0.077112,
            ("S8", "RT", "O5"): 0.0306,
            ("S8", "RT", "O6"): 0.014688,
            ("S8", "RT", "O7"): 0.308448,
            ("S8", "RT", "O8"): 0.1224,
            ("S8", "RT", "O9"): 0.058752,
            ("S8", "RT", "O10"): 0.0126,
            ("S8", "RT", "O11"): 0.005,
            ("S8", "RT", "O12"): 0.0024,
            ("S8", "RT", "O13"): 0.02268,
            ("S8", "RT", "O14"): 0.009,
            ("S8", "RT", "O15"): 0.00432,
            ("S8", "RT", "O16"): 0.09072,
            ("S8", "RT", "O17"): 0.036,
            ("S8", "RT", "O18"): 0.01728,
            ("S8", "RT", "O19"): 0.00756,
            ("S8", "RT", "O20"): 0.003,
            ("S8", "RT", "O21"): 0.00144,
            ("S8", "RT", "O22"): 0.013608,
            ("S8", "RT", "O23"): 0.0054,
            ("S8", "RT", "O24"): 0.002592,
            ("S8", "RT", "O25"): 0.054432,
            ("S8", "RT", "O26"): 0.0216,
            ("S8", "RT", "O27"): 0.010368,
        }

        if pomdp_type == PomdpType.UNSUPERVISED or pomdp_type == PomdpType.RANDOM_UNIFORM:
            N = len(self.O.keys())
            self.O = {o: 1 / 27 for o in self.O.keys()}

        # ensure observation probabilities sum to 1
        for state in self.S.keys():
            rt_sum, mst_sum = 0, 0
            for observation in self.O.keys():
                if observation[0] == state:
                    if observation[1] == "RT":
                        rt_sum += self.O[observation]
                    else:
                        mst_sum += self.O[observation]
            assert math.isclose(mst_sum, 1, abs_tol=1e-9)
            assert math.isclose(rt_sum, 1, abs_tol=1e-9)

    def _generate_state_to_label_mapping(self):
        self.state_to_label_mapping = {state: label for label, state in self.S.items()}
        print(f"State label mapping: {self.state_to_label_mapping}")

    def _generate_state_counts(self):
        self.state_counts = {state_label: 0 for state_label in self.S.keys()}

    def _perseus_generate_policies(self, observation_label: Optional[str] = None):
        """
        Populates approximate policies (S -> A) using Perseus output
        :param `iterations`: int - number of iterations to run Perseus
        """
        self.perseus(observation_label=observation_label)

        self.policies = {
            state_identifier: (
                self.A[opt_reward_action_tuple[1]],  # action function
                opt_reward_action_tuple[1],  # action label
            )
            for state_identifier, opt_reward_action_tuple in self.V.items()
        }

    def _set_initial_belief_states(self):
        """
        The probability of starting at an initial state (s1, s2, ..., s8) is equal
        """
        N = len(self.S.keys())
        self.belief_states = {s: 1 / N for s in self.S.keys()}

    def _get_connecting_state_labels(self, state_label) -> List[str]:
        """
        Assumes that each state s can reach every other state s'
        :return: List[List[bool]] - A list of connecting states
        """
        return list(self.S.keys())

    def _generate_belief_transitions(self):
        """
        Initially the belief of transition probabilities is uniform

        Note: requires transition probabilities (self.T) to be populated
        """
        assert len(self.T) > 0

        # number of transition probabilities
        N = len(self.T)
        p = 1 / N
        self.belief_transitions = {t: p for t in self.T}

    def _set_topology_mst(self):
        self.actions_chosen["MST"] += 1
        print("MST chosen")
        Topology.setTopologyName(1)

    def _set_topology_rt(self):
        self.actions_chosen["RT"] += 1
        print("RT chosen")
        Topology.setTopologyName(0)

    def _is_nfr_satisfied(self, nfr, nfr_reading, nfr_threshold) -> bool:
        # TODO: thresholds are NOT thresholds in the common sense, they are percentages
        # TODO: there is a formula for MC, we have it wrong
        if nfr == "MC":
            # MC satisfiability (BW <= BW_th)
            # bandwith consumption
            return nfr_reading <= nfr_threshold

        elif nfr == "MP":
            # MP satisfiability (TTW <= TTW threshold)
            return nfr_reading <= nfr_threshold
        else:
            assert nfr == "MR"
            # MR satisfiability
            return nfr_reading >= nfr_threshold

    def _map_observation_to_label(self, observation: List[float]) -> str:
        """
        Map an observation to a label.

        :param observation: List[float] - Observation to map
        :return: str - Label of observation
        """
        ac_reading, bw_reading, ttw_reading = observation
        observation = (
            self._get_mr_satisfiability(ac_reading),
            self._get_mc_satisfiability(bw_reading),
            self._get_mp_satisfiability(ttw_reading),
        )
        """
        `label_map` maps an observation `o` to a label, given a state label `s`
        -1 = `o` <= lower_bound
        0 = lower_bound < `o` < upper_bound
        1 = `o` >= upper_bound
        """
        observation_to_observation_label_map = {
            (-1, -1, -1): "O1",
            (-1, -1, 0): "O2",
            (-1, -1, 1): "O3",
            (-1, 0, -1): "O4",
            (-1, 0, 0): "O5",
            (-1, 0, 1): "O6",
            (-1, 1, -1): "O7",
            (-1, 1, 0): "O8",
            (-1, 1, 1): "O9",
            (0, -1, -1): "O10",
            (0, -1, 0): "O11",
            (0, -1, 1): "O12",
            (0, 0, -1): "O13",
            (0, 0, 0): "O14",
            (0, 0, 1): "O15",
            (0, 1, -1): "O16",
            (0, 1, 0): "O17",
            (0, 1, 1): "O18",
            (1, -1, -1): "O19",
            (1, -1, 0): "O20",
            (1, -1, 1): "O21",
            (1, 0, -1): "O22",
            (1, 0, 0): "O23",
            (1, 0, 1): "O24",
            (1, 1, -1): "O25",
            (1, 1, 0): "O26",
            (1, 1, 1): "O27",
        }
        return observation_to_observation_label_map[observation]

    def _generate_transition_counts(self):
        # self.transition_counts[i][j] = number of transitions from state i to state j
        self.transition_counts = {key: 0 for key in self.T.keys()}

    def _generate_observation_counts(self):
        self.observation_counts = {key: 0 for key in self.O.keys()}

    def _get_mr_satisfiability(self, ac_reading: int, alpha: float = 1.0):
        """
        Calculate the MR satisfiability of the current topology.

        RT:
        r: lower bound of active links (r = total_active_links * (config.rt_active_links[0] / 100))
        s: upper bound of active links (s = total_active_links * (config.rt_active_links[1] / 100))

        MST:
        r: lower bound of active links (r = total_active_links * (config.mst_active_links[0] / 100))
        s: upper bound of active links (s = total_active_links * (config.mst_active_links[1] / 100))

        Returns:
        -1 if `reading` <= `r`
        0 if r < `reading` < `s`
        1 if `reading` >= `s`

        :return: int - mr satisfiability
        """
        if Topology.getTopologyName() == "MST":
            r = np.number_of_links * (config.mst_active_links[0] / 100) * alpha
            s = np.number_of_links * (config.mst_active_links[1] / 100) * alpha
        elif Topology.getTopologyName() == "RT":
            r = np.number_of_links * (config.rt_active_links[0] / 100) * alpha
            s = np.number_of_links * (config.rt_active_links[1] / 100) * alpha
        else:
            raise ValueError(f"Invalid topology: {Topology.getTopologyName()}")
        if ac_reading <= r:
            return -1
        elif r < ac_reading < s:
            return 0
        elif ac_reading >= s:
            return 1

    def _get_mp_satisfiability(self, ttw_reading: float, alpha: float = 1.0):
        max_time_to_write = np.number_of_links * config.mst_writing_time[1] * alpha
        # f = lower bound time to write
        f = (config.mst_writing_time[0] / 100) * max_time_to_write
        # g = upper bound time to write
        g = (config.mst_writing_time[1] / 100) * max_time_to_write

        if ttw_reading <= f:
            return -1
        elif f < ttw_reading < g:
            return 0
        elif ttw_reading >= g:
            return 1

    def _get_mc_satisfiability(self, bw_reading: float, alpha: float = 1.0):
        max_bw_consumption = (
            np.number_of_links * config.mst_bandwidth_consumption[1] * alpha
        )
        # x = lower bound bandwidth consumption
        x = (config.mst_bandwidth_consumption[0] / 100) * max_bw_consumption
        # y = upper bound bandwidth consumption
        y = (config.mst_bandwidth_consumption[1] / 100) * max_bw_consumption

        if bw_reading <= x:
            return -1
        elif x < bw_reading < y:
            return 0
        elif bw_reading >= y:
            return 1

    def _get_state_number(self, state_label: str) -> int:
        """
        Get the state number of a state label
        :param state_label: str - State label
        :return: int - State number
        """
        return list(self.S.keys()).index(state_label)
