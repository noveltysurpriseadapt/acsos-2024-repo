# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 20:23:08 2020

@author: 160010321
"""
import random
import RDMSimExemplar.Additional_Resources.Additional_Resources.RDMSim_Python.SimpleSimulationExample.rdm_sim.Config as config
import RDMSimExemplar.Additional_Resources.Additional_Resources.RDMSim_Python.SimpleSimulationExample.rdm_sim.MonitorableMetrics as mm

topology_name = "RT"
topologies = ["RT", "MST"]


# Sets the topology


# Returns the current topology
def getTopologyName():
    global topology_name
    return topology_name


def setTopologyName(topology):
    global topology_name
    topology_name = topologies[topology]
    print(f"new topology: {getTopologyName()}")


# Sets values for the monitorable metrics, MST topology
def MSTImpact():
    mm.setActiveLinks(
        random.randint(config.mst_active_links[0], config.mst_active_links[1]),
        topology_name,
    )
    mm.setBandwidthConsumption(
        random.randint(
            config.mst_bandwidth_consumption[0], config.mst_bandwidth_consumption[1]
        ),
        topology_name,
    )
    mm.setTimetoWrite(
        random.randint(config.mst_writing_time[0], config.mst_writing_time[1])
    )


# Sets values for the monitorable metrics, RT topology
def RTImpact():
    mm.setActiveLinks(
        random.randint(config.rt_active_links[0], config.rt_active_links[1]),
        topology_name,
    )
    mm.setBandwidthConsumption(random.randint(20, 30), topology_name)
    mm.setTimetoWrite(random.randint(10, 20))
