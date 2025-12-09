from numba import njit
import numba as nb 
from collections import defaultdict
import numpy as np
import os
import json
import itertools
import networkx as nx
import pandas as pd
import random
from time import time
from tqdm import tqdm
import pickle as pk
import math

class SimuNonlinearContagion_SIR():
    def __init__(self, HGraph):
        self.HG = HGraph  # List of hyperedges, each as a list of nodes
        temp = []
        for i in self.HG:
            temp.extend(i)
        nodeSet = set(temp)
        self.Hnodes = list(nodeSet)
        self.Hnodes.sort()  # List of node IDs (int) in the hypergraph, sorted in ascending order
        self.HedgeDic = dict()  # Key: hyperedge ID (int), value: hyperedge (list of nodes)
        self.HedgeSizeDic = dict()  # Key: hyperedge ID (int), value: hyperedge size (int)
        self.maxHegdeSize = 0
        self.HneiEdgeDic = defaultdict(list)  # Key: node ID (int), value: list of incident hyperedge IDs
        self.N = len(self.Hnodes)
        i = 1  # i denotes the hyperedge ID
        for hyperedge in self.HG:
            self.HedgeDic[i] = hyperedge
            self.HedgeSizeDic[i] = len(hyperedge)
            if len(hyperedge) > self.maxHegdeSize:
                self.maxHegdeSize = len(hyperedge)
            for node in hyperedge:
                self.HneiEdgeDic[node].append(i)                
            i += 1

        self.iAgentSet = set()
        self.sAgentSet = set()
        self.rAgentSet = set()

    def infectAgent(self, agent):
        self.iAgentSet.add(agent)
        self.sAgentSet.remove(agent)
        self.nodeStateDic[agent] = 'I'
        return 1

    def recoverAgent_R(self, agent):
        self.rAgentSet.add(agent)
        self.iAgentSet.remove(agent)
        self.nodeStateDic[agent] = 'R'
        return 0
    
    def immuneAgent(self, agent):
        self.rAgentSet_imm.add(agent)
        self.sAgentSet.remove(agent)
        self.nodeStateDic[agent] = 'R_imm'
        return 0

    def initialization(self, initial_infected, immunilized=[]):  # SIR: always specify the initial infected nodes
        self.nodeStateDic = dict()
        self.sAgentSet = set()
        self.iAgentSet = set()
        self.rAgentSet = set()
        self.rAgentSet_imm = set()
        for n in self.Hnodes:
            self.sAgentSet.add(n)
            self.nodeStateDic[n] = 'S'
            
        self.infect_this_setup = initial_infected
        for to_infect in self.infect_this_setup:  # to_infect is the node ID
            self.infectAgent(to_infect)
            self.nodeStateDic[to_infect] = 'I'

        if immunilized != []:
            self.immunilize_this_setup = immunilized
            for to_immu in self.immunilize_this_setup:
                self.immuneAgent(to_immu)
                self.nodeStateDic[to_immu] = 'R_imm'

    def initialization_random(self, randomInfect=0):
        self.nodeStateDic = dict()  # Node states: key = node, value = state
        for n in self.Hnodes:
            self.sAgentSet.add(n)
            self.nodeStateDic[n] = 'S'
        
        self.iAgentSet = set()
        self.rAgentSet = set()
            
        if randomInfect > 0:
            self.infect_this_setup = random.sample(self.Hnodes, randomInfect)
        else:
            print("error, initialization unsetted") 
        for to_infect in self.infect_this_setup:  # to_infect is the node ID
            self.infectAgent(to_infect)

    def contagionSimulation(self, l, mu, nu, tmax=float("Inf"), timeSeries=False):

        infectionProbLst = [np.exp(-l*i**nu) for i in range(0, self.maxHegdeSize)]
        I = [len(self.iAgentSet)]
        R = [len(self.rAgentSet)]
        S = [len(self.sAgentSet)]
        t = 0
        iNodesNum_inHedge = {}
        while t < tmax and I[-1] != 0:
            t += 1
            for edgeID in self.HedgeDic:
                iNodesNum_inHedge[edgeID] = 0
                for i in self.HedgeDic[edgeID]:
                    if self.nodeStateDic[i] == 'I':
                        iNodesNum_inHedge[edgeID] += 1
            
            # Infection process
            newI = []
            """
            potentialI = set()  # To speed up the simulation
            for edgeID in iNodesNum_inHedge:  
                if iNodesNum_inHedge > 0:
                    for node in self.HedgeDic[edgeID]:
                        potentialI.add(node)
            """
            for node in self.sAgentSet:
                p = 1
                for edgeID in self.HneiEdgeDic[node]:  # Incident hyperedge IDs for node
                    p = p * infectionProbLst[iNodesNum_inHedge[edgeID]]
                if np.random.random() < 1 - p:
                    newI.append(node)
            
            # Recovery process (use newR because sets cannot be modified while iterating)
            newR = set()
            for n_to_recover in self.iAgentSet:
                if (np.random.random() < mu):
                    newR.add(n_to_recover)
            for n_to_recover in newR:        
                self.recoverAgent_R(n_to_recover)
            
            # Update node states
            for n_to_infect in newI:
                self.infectAgent(n_to_infect)
            I.append(len(self.iAgentSet))
            R.append(len(self.rAgentSet))
            S.append(len(self.sAgentSet))

        if timeSeries:
            return R, t, I, S
        else:
            return R[-1], t, I[-1], S[-1]
    
    def prepare_for_numba(self):
        # 1. Convert nodeStateDic to np.array of shape [max_node_id + 1, 3]
        #    Column 0 = 1: infected, column 1 = 1: recovered, column 2 = 1: susceptible
        nodes = list(self.nodeStateDic.keys())
        nodeStates = np.zeros((np.max(nodes) + 1, 3), dtype=np.int64)
        for key, val in self.nodeStateDic.items():
            if val == 'I':
                nodeStates[key, 0] = 1
            elif val == 'R':
                nodeStates[key, 1] = 1
            elif val == 'S':
                nodeStates[key, 2] = 1
        self.nodeStates = nodeStates

        # 2. Convert iNodesNum_inHedge to np.array
        edgeIDs = list(self.HedgeDic.keys())
        iNodesNum_inHedge = np.zeros((np.max(edgeIDs) + 1), dtype=np.int64)
        self.iNodesNum_inHedge = iNodesNum_inHedge  # Not synchronized with initialization; updated in the simulation loop

        # 3. Convert self.HneiEdgeDic and self.HedgeDic to numba-compatible dictionaries
        self.nb_HneiEdgeDic = dict2numba(self.HneiEdgeDic)
        self.nb_HedgeDic = dict2numba(self.HedgeDic)

def dict2numba(dic):
    """Convert a Python dict into the format required by numba."""
    output = nb.typed.Dict.empty(
        key_type = nb.types.int64, 
        value_type = nb.types.Array(nb.types.int64, 1, 'C')
    )
    for key, value in dic.items():
        output[key] = np.array(value, dtype=np.int64)
    return output

@njit
def contagion_simulation_numba(nodeStates, iNodesNum_inHedge, HneiEdgeDic, HedgeDic, infectionProbLst, mu, tmax):
    # Initialize counts of different node states
    I = [np.sum(nodeStates[:, 0])]
    R = [np.sum(nodeStates[:, 1])]
    S = [np.sum(nodeStates[:, 2])]
    t = 0

    while t < tmax and I[-1] != 0:
        t += 1
        for edgeID in HedgeDic:
            # Use HedgeDic[edgeID] as indices to sum infected nodes in each hyperedge
            iNodesNum_inHedge[edgeID] = np.sum(nodeStates[HedgeDic[edgeID], 0])

        newI = []
        # Use nonzero on the third column to get indices of susceptible nodes
        indexes = np.nonzero(nodeStates[:, 2])[0] 
        for i in indexes:
            p = 1
            for edgeID in HneiEdgeDic[i]:
                p *= infectionProbLst[iNodesNum_inHedge[edgeID]]
            if np.random.random() < 1 - p:
                newI.append(i)
        

        newR = []
        # Use nonzero on the first column to get indices of infected nodes
        indexes = np.nonzero(nodeStates[:, 0])[0]
        for n_to_recover in indexes:
            if np.random.random() < mu:
                newR.append(n_to_recover)

        for n_to_recover in newR:
            nodeStates[n_to_recover, 0] = 0
            nodeStates[n_to_recover, 1] = 1
            nodeStates[n_to_recover, 2] = 0

        for n_to_infect in newI:
            nodeStates[n_to_infect, 0] = 1
            nodeStates[n_to_infect, 1] = 0
            nodeStates[n_to_infect, 2] = 0

        I.append(np.sum(nodeStates[:, 0]))
        R.append(np.sum(nodeStates[:, 1]))
        S.append(np.sum(nodeStates[:, 2]))

    return R[-1], t, I[-1], S[-1]


@njit
def contagion_simulation_numba_timeDetail(nodeStates, iNodesNum_inHedge, HneiEdgeDic, HedgeDic, infectionProbLst, mu, tmax):
    # Initialize counts of different node states
    I = [np.sum(nodeStates[:, 0])]
    R = [np.sum(nodeStates[:, 1])]
    S = [np.sum(nodeStates[:, 2])]
    t = 0
    Inodes = []
    Rnodes = []

    while t < tmax and I[-1] != 0:
        t += 1
        for edgeID in HedgeDic:
            # Use HedgeDic[edgeID] as indices to sum infected nodes in each hyperedge
            iNodesNum_inHedge[edgeID] = np.sum(nodeStates[HedgeDic[edgeID], 0])

        newI = []
        # Use nonzero on the third column to get indices of susceptible nodes
        indexes = np.nonzero(nodeStates[:, 2])[0] 
        for i in indexes:
            p = 1
            for edgeID in HneiEdgeDic[i]:
                p *= infectionProbLst[iNodesNum_inHedge[edgeID]]
            if np.random.random() < 1 - p:
                newI.append(i)
        

        newR = []
        # Use nonzero on the first column to get indices of infected nodes
        indexes = np.nonzero(nodeStates[:, 0])[0]
        for n_to_recover in indexes:
            if np.random.random() < mu:
                newR.append(n_to_recover)

        for n_to_recover in newR:
            nodeStates[n_to_recover, 0] = 0
            nodeStates[n_to_recover, 1] = 1
            nodeStates[n_to_recover, 2] = 0

        for n_to_infect in newI:
            nodeStates[n_to_infect, 0] = 1
            nodeStates[n_to_infect, 1] = 0
            nodeStates[n_to_infect, 2] = 0

        I.append(np.sum(nodeStates[:, 0]))
        R.append(np.sum(nodeStates[:, 1]))
        S.append(np.sum(nodeStates[:, 2]))

        Inodes.append(np.nonzero(nodeStates[:, 0])[0])
        Rnodes.append(np.nonzero(nodeStates[:, 1])[0])
    
    return Rnodes, Inodes
