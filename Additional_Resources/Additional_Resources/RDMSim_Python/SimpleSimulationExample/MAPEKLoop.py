from rdm_sim import NetworkManagement as nm
from rdm_sim import NetworkProperties as np
from rdm_sim import Topology


# Calls the probe function to get the values for the monitorable metrics
def Monitor():
    active_links, bandwidth_consumption, write_time = nm.probe()
    return active_links, bandwidth_consumption, write_time


# Decides a Topology based on the monitorable metrics' values
# (naive decision making)
def AnalyzeandPlan(active_links, bandwidth_consumption, write_time):
    link_threshold = np.number_of_links * 0.55
    bandwidth_threshold = (np.number_of_links * 30) * 0.31
    time_threshold = (np.number_of_links * 20) * 0.26

    if active_links < link_threshold:
        Topology.setTopologyName(0)

    else:
        if bandwidth_consumption > bandwidth_threshold or write_time > time_threshold:
            Topology.setTopologyName(1)

        else:
            Topology.setTopologyName(0)


# Calls the effector function to set the topology for the network
def Execute():
    nm.effector()


# Calls all the previous functions
def Run(pomdp):
    observations = Monitor()  # active_links, bandwidth_consumption, write_time
    # AnalyzeandPlan(*observations) (naive decision making)
    pomdp.make_decision(
        observations
    )  # make decision based off approximate (optimal) policies
    Execute()
