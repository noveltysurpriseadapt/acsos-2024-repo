import random
import RDMSimExemplar.Additional_Resources.Additional_Resources.RDMSim_Python.SimpleSimulationExample.rdm_sim.Config as config


# This will calculate a deviation value for the number of links based on the topology and selected scenario
def link_deviation(topology_name):
    if topology_name == "MST":
        if config.scenario == 1:
            return random.randint(
                config.deviation_scenario_1[0], config.deviation_scenario_1[1]
            )

        elif config.scenario == 3:
            return random.randint(
                config.deviation_scenario_3[0], config.deviation_scenario_3[1]
            )

        elif config.scenario == 4:
            return random.randint(
                config.deviation_scenario_4[0], config.deviation_scenario_4[1]
            )

        elif config.scenario == 6:
            return random.randint(
                config.deviation_scenario_6[0], config.deviation_scenario_6[1]
            )

    elif topology_name == "RT":
        if config.scenario == 5:
            return random.randint(
                config.deviation_scenario_5[0], config.deviation_scenario_5[1]
            )

        elif config.scenario == 6:
            return random.randint(
                config.deviation_scenario_6[0], config.deviation_scenario_6[1]
            )

    return 0


# This will calculate a deviation value for the bandwidth consumption and writing times based on the topology and selected scenario
def BC_WT_deviation(topology_name):
    if topology_name == "RT":
        if config.scenario == 2:
            return random.randint(
                config.deviation_scenario_2[0], config.deviation_scenario_2[1]
            )

        elif config.scenario == 3:
            return random.randint(
                config.deviation_scenario_3[0], config.deviation_scenario_3[1]
            )

        elif config.scenario == 5:
            return random.randint(
                config.deviation_scenario_5[0], config.deviation_scenario_5[1]
            )

        elif config.scenario == 6:
            return random.randint(
                config.deviation_scenario_6[0], config.deviation_scenario_6[1]
            )

    elif topology_name == "MST":
        if config.scenario == 4:
            return random.randint(
                config.deviation_scenario_4[0], config.deviation_scenario_4[1]
            )

        elif config.scenario == 6:
            return random.randint(
                config.deviation_scenario_6[0], config.deviation_scenario_6[1]
            )

    return 0
