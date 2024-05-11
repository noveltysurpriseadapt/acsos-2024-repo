# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 20:11:19 2020

@author: Huma Samin
"""
number_of_mirrors = 0
number_of_links = 0


# Sets the number of mirrors
def setNumberOfMirrors(mirrorsnumber):
    global number_of_mirrors
    number_of_mirrors = mirrorsnumber


# Returns the number of mirrors
def getNumberofMirrors():
    return number_of_mirrors


# Sets the total number of links
def setNumberOfLinks():
    global number_of_links
    number_of_links = (number_of_mirrors * (number_of_mirrors - 1)) / 2


# Returns the total number of links
def getNumberofLinks():
    return number_of_links
