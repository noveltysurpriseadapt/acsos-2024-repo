from matplotlib import pyplot as plt
import simpy as sim
import json
import random
import RDMSimExemplar.Additional_Resources.Additional_Resources.RDMSim_Python.SimpleSimulationExample.rdm_sim.NetworkManagement as nm
import RDMSimExemplar.Additional_Resources.Additional_Resources.RDMSim_Python.SimpleSimulationExample.rdm_sim.NetworkProperties as np
import RDMSimExemplar.Additional_Resources.Additional_Resources.RDMSim_Python.SimpleSimulationExample.rdm_sim.Config as config
import RDMSimExemplar.Additional_Resources.Additional_Resources.RDMSim_Python.SimpleSimulationExample.rdm_sim.Logger as log
import RDMSimExemplar.Additional_Resources.Additional_Resources.RDMSim_Python.SimpleSimulationExample.rdm_sim.Topology as topology
from RDMSimExemplar.Additional_Resources.Additional_Resources.RDMSim_Python.SimpleSimulationExample.MAPEKLoop import (
    Run,
)
from RDMSimExemplar.Additional_Resources.Additional_Resources.RDMSim_Python.SimpleSimulationExample.SAS.main import (
    POMDP,
    PomdpType
)
from RDMSimExemplar.Additional_Resources.Additional_Resources.RDMSim_Python.SimpleSimulationExample.SAS.graphs import (
    POMDPGraphs,
)

plt.rcParams.update({"font.size": 11})


def Simulator(
    env,
    deviation_list,
    topology_list,
    bandwidths_list,
    links_list,
    times_list,
    step,
    pomdp: POMDP,
):
    while True:
        # Runs the MAPEK loop
        Run(pomdp)
        active_links, bandwidth_consumption, write_time = nm.probe()
        print(
            f"=== Probe ===\nActive Links: {active_links}\nBandwidth consumption: {bandwidth_consumption}\nWrite time: {write_time}"
        )
        # This records the data to be displayed on the plots
        step.append(env.now)
        bandwidths_list.append(bandwidth_consumption)
        deviation_list.append(nm.deviation)
        topology_list.append(topology.getTopologyName())
        links_list.append(active_links)
        times_list.append(write_time)

        # Print the data on the console
        print("timestep:", env.now)
        # print("Selected topology:", topology.getTopologyName())

        yield env.timeout(1)


def RunRDMSim():
    # Load the configuration files and checks for errors
    if True:
        config.load_no_GUI()
        config.load_config()

    error, source = config.check_values()
    if error != 0:
        if error == 1:
            print("Configuration error. Deviation values must be between 0 and 100")
            print("Source of error:", source)
            return

        if error == 2:
            print(
                "Configuration error. Deviation values format [lower limit, upper limit]"
            )
            print("Source of error:", source)
            return

    # Close any previously open plot, useful when displaying the GUI
    plt.close()

    # Sets the random seed, if any
    random.seed(config.random_seed)

    # Sets the number of mirror and total links
    np.setNumberOfMirrors(config.mirror_number)
    np.setNumberOfLinks()

    # Sets the threshold satisfaction values
    link_threshold = np.number_of_links * 0.35
    print(f"Total links: {link_threshold}")

    #
    # bandwidth_threshold = np.number_of_links * 30 * 0.31
    bandwidth_threshold = 3600
    # time_threshold = np.number_of_links * 20 * 0.26
    time_threshold = np.number_of_links * 20 * 0.45

    env = sim.Environment()
    bandwidths_list = []
    deviation_list = []
    topology_list = []
    links_list = []
    times_list = []
    step = []

    # Runs the simulation
    thresholds = [link_threshold, bandwidth_threshold, time_threshold]

    # initialise POMDP
    pomdp = POMDP(thresholds=thresholds, pomdp_type=PomdpType.UNSUPERVISED)

    env.process(
        Simulator(
            env,
            deviation_list,
            topology_list,
            bandwidths_list,
            links_list,
            times_list,
            step,
            pomdp,
        )
    )
    env.run(until=config.time_steps)
    print(f"Reward: [{pomdp.total_reward}]")  # total reward gained by pomdp

    # generate graphs
    """
    graphs = POMDPGraphs(pomdp=pomdp)
    graphs.action_count_bar_chart()
    graphs.state_counts_bar_chart()
    graphs.plot_surprise()
    graphs.plot_novelty()
    graphs.plot_novelty_surprise_correction()
    graphs.plot_transition_heatmap()
    graphs.plot_box_plots_bw()
    graphs.plot_box_plots_ac()
    graphs.plot_box_plots_ttw()
    graphs.plot_scenario_actions_chosen_bar_charts()
    """

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 4), dpi=80)
    fig.tight_layout(pad=3.0)

    th_step = [0, config.time_steps - 1]
    bw_threshold = [bandwidth_threshold, bandwidth_threshold]
    print(f"Bandwidth threshold: {bandwidth_threshold}")
    l_threshold = [link_threshold, link_threshold]
    print(f"Active links threshold: {l_threshold}")
    t_threshold = [time_threshold, time_threshold]
    print(f"Time to write threshold: {t_threshold}")

    # Bandwiths' plot
    print(f"Bandwidth list:\n{bandwidths_list}\n threshold: {bandwidth_threshold}")
    ax1.plot(step, bandwidths_list, color="orange")
    ax1.plot(th_step, bw_threshold, color="red", label="Satisfaction threshold")
    ax1.set_title("Bandwidth consumption over time")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Bandwidth Consumption (GB)")
    ax1.legend(loc="lower right")

    # Active links' plot
    print(f"AC list:\n{links_list}\nthreshold: {link_threshold}")
    ax2.plot(step, links_list, color="green")
    ax2.plot(th_step, l_threshold, color="red", label="Satisfaction threshold")
    ax2.set_title("Active network links over time")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Active network links")
    ax2.legend(loc="lower right")

    # Writing times' plot
    print(f"TTW list:\n{times_list}\nthreshold: {time_threshold}")
    ax3.plot(step, times_list)
    ax3.plot(th_step, t_threshold, color="red", label="Satisfaction threshold")
    ax3.set_title("Time to write data over time")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Time to write data (ms)")
    ax3.legend(loc="lower right")

    # Actions chosen log
    print(f"Actions chosen:\nMST: {pomdp.actions_chosen['MST']}]\nRT: {pomdp.actions_chosen['RT']}")

    # Log surprise
    print(f"Surprise:\n{pomdp.states_surprise_log}")
    # Makes the log entry in the log file
    log.make_log(
        config.scenario,
        config.random_seed,
        time_threshold,
        link_threshold,
        bandwidth_threshold,
        deviation_list,
        topology_list,
        step,
        times_list,
        links_list,
        bandwidths_list,
    )

    plt.show()


if __name__ == "__main__":
    fig = RunRDMSim()
