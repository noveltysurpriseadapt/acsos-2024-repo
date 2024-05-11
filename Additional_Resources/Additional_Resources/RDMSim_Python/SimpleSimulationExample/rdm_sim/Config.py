import json

random_seed = None
scenario = None
time_steps = 0
mirror_number = 0
mst_active_links = []
mst_bandwidth_consumption = []
mst_writing_time = []
rt_active_links = []
deviation_scenario_1 = []
deviation_scenario_2 = []
deviation_scenario_3 = []
deviation_scenario_4 = []
deviation_scenario_5 = []
deviation_scenario_6 = []
probability_of_change = {}
volatility = 0


# This function loads the values for seed and scenario, from the configuration files, this will be used if run without GUI
def load_no_GUI():
    global scenario, random_seed, probability_of_change

    # Load the configuration.json file's values
    with open("rdm_sim/configuration.json", "r") as config:
        configuration = json.load(config)

        random_seed = configuration["random_seed"]
        scenario = configuration["scenario"]
        probability_of_change = configuration["probability_of_change"]

    config.close()


# This function loads the values from the configuration files
def load_config():
    global time_steps, mirror_number, mst_active_links, mst_bandwidth_consumption, mst_writing_time, rt_active_links, probability_of_change
    global deviation_scenario_1, deviation_scenario_2, deviation_scenario_3, deviation_scenario_4, deviation_scenario_5, deviation_scenario_6

    # Load the configuration.json file's values
    with open("rdm_sim/configuration.json", "r") as config:
        configuration = json.load(config)

        time_steps = configuration["time_steps"]
        mirror_number = configuration["mirror_number"]
        probability_of_change = configuration["probability_of_change"]

        for i in range(2):
            mst_active_links.append(configuration["mst_active_links"][i])
            mst_bandwidth_consumption.append(
                configuration["mst_bandwidth_consumption"][i]
            )
            mst_writing_time.append(configuration["mst_writing_time"][i])
            rt_active_links.append(configuration["rt_active_links"][i])
    config.close()

    # Load the configuration_detrimental.json file's values
    with open("rdm_sim/configuration_detrimental.json", "r") as config:
        configuration = json.load(config)

        for i in range(2):
            deviation_scenario_1.append(configuration["deviation_scenario_1"][i])
            deviation_scenario_2.append(configuration["deviation_scenario_2"][i])
            deviation_scenario_3.append(configuration["deviation_scenario_3"][i])
            deviation_scenario_4.append(configuration["deviation_scenario_4"][i])
            deviation_scenario_5.append(configuration["deviation_scenario_5"][i])
            deviation_scenario_6.append(configuration["deviation_scenario_6"][i])
    config.close()


# This function check if there is a mistake with the configuration values
def check_values():

    # Check if the deviation is outside the allowed range of 0~100
    for i in range(len(mst_active_links)):
        if mst_active_links[i] < 0 or mst_active_links[i] > 100:
            return 1, "mst_links"

    for i in range(len(mst_bandwidth_consumption)):
        if mst_bandwidth_consumption[i] < 0 or mst_bandwidth_consumption[i] > 100:
            return 1, "mst_bandwidth_consumption"

    for i in range(len(mst_writing_time)):
        if mst_writing_time[i] < 0 or mst_writing_time[i] > 100:
            return 1, "mst_writing_time"

    for i in range(len(rt_active_links)):
        if rt_active_links[i] < 0 or rt_active_links[i] > 100:
            return 1, "rt_active_links"

    for i in range(len(deviation_scenario_1)):
        if deviation_scenario_1[i] < 0 or deviation_scenario_1[i] > 100:
            return 1, "deviation_scenario_1"

    for i in range(len(deviation_scenario_2)):
        if deviation_scenario_2[i] < 0 or deviation_scenario_2[i] > 100:
            return 1, "deviation_scenario_2"

    for i in range(len(deviation_scenario_3)):
        if deviation_scenario_3[i] < 0 or deviation_scenario_3[i] > 100:
            return 1, "deviation_scenario_3"

    for i in range(len(deviation_scenario_4)):
        if deviation_scenario_4[i] < 0 or deviation_scenario_4[i] > 100:
            return 1, "deviation_scenario_4"

    for i in range(len(deviation_scenario_5)):
        if deviation_scenario_5[i] < 0 or deviation_scenario_5[i] > 100:
            return 1, "deviation_scenario_5"

    for i in range(len(deviation_scenario_6)):
        if deviation_scenario_6[i] < 0 or deviation_scenario_6[i] > 100:
            return 1, "deviation_scenario_6"

    # Check the order of the deviation values
    if mst_active_links[0] > mst_active_links[1]:
        return 2, "mst_links"

    if mst_bandwidth_consumption[0] > mst_bandwidth_consumption[1]:
        return 2, "mst_bandwidth_consumption"

    if mst_writing_time[0] > mst_writing_time[1]:
        return 2, "mst_writing_time"

    if rt_active_links[0] > rt_active_links[1]:
        return 2, "rt_active_links"

    if deviation_scenario_1[0] > deviation_scenario_1[1]:
        return 2, "deviation_scenario_1"

    if deviation_scenario_3[0] > deviation_scenario_3[1]:
        return 2, "deviation_scenario_3"

    if deviation_scenario_4[0] > deviation_scenario_4[1]:
        return 2, "deviation_scenario_4"

    if deviation_scenario_5[0] > deviation_scenario_5[1]:
        return 2, "deviation_scenario_5"

    if deviation_scenario_6[0] > deviation_scenario_6[1]:
        return 2, "deviation_scenario_6"

    return 0, None
