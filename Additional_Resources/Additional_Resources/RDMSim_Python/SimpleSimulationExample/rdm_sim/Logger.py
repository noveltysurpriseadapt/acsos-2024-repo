import simpy as sim
import json


# Make a json file to log the results for each simulation
def make_log(
    scenario,
    seed,
    time_t,
    link_t,
    bandwidth_t,
    deviation,
    topologies,
    steps,
    times,
    links,
    bandwidths,
):
    try:
        jsonfile = open("log.json", "r")
        jsonfile.close()
    except:
        jsonfile = open("log.json", "w")
        jsonfile.write("{\n\n}")
        jsonfile.close()

    with open("log.json") as jsonfile:
        log = json.load(jsonfile)
        entries = len(log)
        new_entry = {}

        for i in range(len(steps)):
            new_entry.update(
                {
                    steps[i]: {
                        "Selected topology": topologies[i],
                        "Time to write data": times[i],
                        "Active links": links[i],
                        "Bandwidths consumption": bandwidths[i],
                        "Deviation": deviation[i],
                    }
                }
            )

        log.update(
            {
                entries: {
                    "Scenario": scenario,
                    "Seed": seed,
                    "Writing times bandwidths": time_t,
                    "Active link threshold": link_t,
                    "Bandwidth consumption threshold": bandwidth_t,
                    "Values per timestep": new_entry,
                }
            }
        )

    jsonfile.close()

    with open("log.json", "w") as jsonfile:
        json.dump(log, jsonfile, indent=1)
