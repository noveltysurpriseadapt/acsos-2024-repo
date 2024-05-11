# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 20:24:42 2020

@author: 160010321
"""
import RDMSimExemplar.Additional_Resources.Additional_Resources.RDMSim_Python.SimpleSimulationExample.rdm_sim.Topology as Topology
import RDMSimExemplar.Additional_Resources.Additional_Resources.RDMSim_Python.SimpleSimulationExample.rdm_sim.MonitorableMetrics as mm

deviation = mm.getDeviation()


# Gets the values for the monitorable metrics and the current topology
def probe():
    return mm.getActiveLinks(), mm.getBandwidthConsumption(), mm.getTimetoWrite()


# Sets the network topology, by calling the impact functions
def effector():
    current_topology = Topology.getTopologyName()

    if current_topology == "RT":
        print(f"Current topology: {current_topology} -> RT Impact")
        Topology.RTImpact()

    elif current_topology == "MST":
        print(f"Current topology: {current_topology} -> MST Impact")
        Topology.MSTImpact()

    else:
        print("An unexpected error occurred. Unknown topology")
        return
