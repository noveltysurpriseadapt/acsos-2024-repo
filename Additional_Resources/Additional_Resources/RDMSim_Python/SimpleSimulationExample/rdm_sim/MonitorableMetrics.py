# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 20:24:06 2020

@author: 160010321
"""
import random
import RDMSimExemplar.Additional_Resources.Additional_Resources.RDMSim_Python.SimpleSimulationExample.rdm_sim.NetworkProperties as np
import RDMSimExemplar.Additional_Resources.Additional_Resources.RDMSim_Python.SimpleSimulationExample.rdm_sim.DetrimentalScenarios as ss

active_network_links = 0
bandwidth_consumption = 0
time_to_write_data = 0
deviation = 0


# Determine a number of active links based on the total of links, a percentage of active links impact of the topology, and the deviation function
def setActiveLinks(percentage, topology_name):
    global active_network_links
    global deviation
    deviation = ss.link_deviation(topology_name)
    active_network_links = int(percentage * np.number_of_links / 100)
    active_network_links = int(active_network_links * (1 - deviation / 100))


# Return the number of active links
def getActiveLinks():
    return active_network_links


# Determine bandwidth consumption
def setBandwidthConsumption(singleconsumption, topology_name):
    global bandwidth_consumption
    global deviation
    deviation = ss.BC_WT_deviation(topology_name)
    bandwidth_consumption = active_network_links * singleconsumption
    bandwidth_consumption = float(bandwidth_consumption * (1 + deviation / 100))


# Return bandwidth consumption
def getBandwidthConsumption():
    return bandwidth_consumption


# Determine the time to write
def setTimetoWrite(singlewrite):
    global time_to_write_data
    time_to_write_data = active_network_links * singlewrite
    time_to_write_data = float(time_to_write_data * (1 + deviation / 100))


# Return time to write
def getTimetoWrite():
    return time_to_write_data


# Return current deviation value
def getDeviation():
    return deviation
