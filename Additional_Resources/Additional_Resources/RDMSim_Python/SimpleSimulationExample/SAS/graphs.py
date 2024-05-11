import matplotlib.pyplot as plt
import seaborn as sns
import RDMSimExemplar.Additional_Resources.Additional_Resources.RDMSim_Python.SimpleSimulationExample.rdm_sim.Config as config
import pandas as pd
import numpy as np
from RDMSimExemplar.Additional_Resources.Additional_Resources.RDMSim_Python.SimpleSimulationExample.SAS.main import PomdpType

class POMDPGraphs:
    def __init__(self, pomdp):
        self.pomdp = pomdp

    def action_count_bar_chart(self):
        bar_positions = [1, 2]
        plt.bar(
            bar_positions,
            [self.pomdp.actions_chosen["MST"], self.pomdp.actions_chosen["RT"]],
        )
        plt.xlabel("Action Chosen")
        plt.ylabel("Frequency")
        plt.title(
            f"{self.get_scenario_label()} {self.get_pomdp_label()} Actions Chosen"
        )
        plt.xticks(bar_positions, ("MST", "RT"))
        plt.show()

    def state_counts_bar_chart(self):
        state_labels = list(self.pomdp.S.keys())
        bar_positions = [i for i in range(1, len(state_labels) + 1)]
        bar_heights = [self.pomdp.state_counts[s] for s in state_labels]
        plt.bar(bar_positions, bar_heights)
        plt.xlabel("State Visited")
        plt.ylabel("Frequency")
        plt.title(
            f"{self.get_scenario_label()} {self.get_pomdp_label()} States Visited"
        )
        plt.xticks(bar_positions, labels=state_labels)
        plt.show()

    def plot_surprise(self):
        plt.plot(self.pomdp.states_surprise_log)
        plt.xlabel("Step")
        plt.ylabel("Surprise")
        plt.title(
            f"{self.get_scenario_label()} {self.get_pomdp_label()} Surprise over time"
        )
        plt.show()

    def plot_novelty(self):
        plt.plot(self.pomdp.novelty_log)
        plt.xlabel("Step")
        plt.ylabel("Novelty")
        plt.title(
            f"{self.get_scenario_label()} {self.get_pomdp_label()} Novelty over time"
        )
        plt.show()

    def plot_transition_heatmap(self):
        # row = prev state
        # col = transitioned state
        fig, ax = plt.subplots()
        total = 0
        data = [[0] * len(self.pomdp.S) for _ in range(len(self.pomdp.S))]
        for (s_prime, s, action), count in self.pomdp.transition_counts.items():
            s_prime_idx = list(self.pomdp.S.keys()).index(s_prime)
            s_idx = list(self.pomdp.S.keys()).index(s)
            data[s_idx][s_prime_idx] += count
            total += count

        print(f"TOTAL TRANSITIONS: {total}")

        sns.heatmap(data, annot=True, cmap="YlGnBu", fmt="d")
        ax.set_xticklabels(["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"])
        ax.set_yticklabels(["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"])
        ax.set_ylabel("From State")
        ax.set_xlabel("To State")
        plt.title(
            f"{self.get_scenario_label()} {self.get_pomdp_label()} Transition Counts"
        )
        plt.show()

    def get_scenario_label(self) -> str:
        return f"[Scenario {config.scenario}]"

    def get_pomdp_label(self) -> str:
        if self.pomdp.pomdp_type == PomdpType.UNSUPERVISED:
            return "[Unsupervised]"
        elif self.pomdp.pomdp_type == PomdpType.RANDOM_UNIFORM:
            return "[Random Uniform]"
        return "[Expert Informed]"

    def plot_box_plots_bw(self, data):
        threshold = [3600, 3600]
        self.plot_box_plot(data=data, threshold=threshold, ylabel="Bandwidth consumption")


    def plot_box_plots_ac(self, data):
        threshold = [105.0, 105.0]
        self.plot_box_plot(data=data, threshold=threshold, ylabel="Active links")

    def plot_box_plots_ttw(self, data):
        threshold = [2700.0, 2700.0]
        self.plot_box_plot(data=data, threshold=threshold, ylabel="Time to write")

    def plot_box_plot(self, data, threshold, ylabel):
        df = pd.DataFrame(data)
        plt.plot([0, config.time_steps - 1], threshold, color="red", label="Satisfaction threshold")
        ax = df.plot(kind='box', patch_artist=True)
        plt.setp(ax.lines, color='black')
        plt.axhline(y=threshold[0], color='r', linestyle='-', label="Satisfaction threshold", linewidth=3)

        plt.xlabel("Scenario")
        plt.ylabel(ylabel)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                  ncol=3, fancybox=True, shadow=True)
        plt.show()

    def plot_novelty_surprise_correction(self):
        plt.plot(self.pomdp.states_surprise_log, label="Surprise")
        plt.plot(self.pomdp.novelty_log, label="Novelty")
        plt.xlabel("Time")
        plt.title(f"{self.get_scenario_label()} {self.get_pomdp_label()} Novelty vs Surprise")
        plt.legend(loc="upper left")
        plt.show()

    def plot_scenario_actions_chosen_bar_charts(self, data):
        fig, ax = plt.subplots()
        mst_chosen = [x[0] for x in data.values()]
        rt_chosen = [x[1] for x in data.values()]

        index = np.arange(len(data))
        bar_width = 0.35

        bar1 = ax.bar(index - bar_width/2, mst_chosen, bar_width, label="MST chosen")
        bar2 = ax.bar(index + bar_width/2, rt_chosen, bar_width, label="RT chosen")
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Frequency")
        ax.set_xticks(np.arange(0, 7))
        ax.set_xticklabels(data.keys())
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                  ncol=3, fancybox=True, shadow=True)
        plt.tight_layout()
        plt.show()

