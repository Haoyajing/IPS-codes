import sys
import os
# Get the current file directory and add it to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from numba import njit
import numba as nb 
from collections import defaultdict
from collections import Counter
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
from chooseTopNodes import *

class Immunization_SIS():
    def __init__(self, HGraph):
        self.HG = HGraph  # list of hyperedges, each hyperedge is a list of nodes
        temp = []
        for i in self.HG:
            temp.extend(i)
        nodeSet = set(temp)
        self.Hnodes = list(nodeSet)
        self.Hnodes.sort()  # list of hypergraph node indices (int), sorted in ascending order
        self.HedgeDic = dict()  # key: hyperedge index (int); value: hyperedge as a list of nodes
        self.HedgeSizeDic = dict()  # key: hyperedge index (int); value: hyperedge size (int)
        self.maxHegdeSize = 0
        self.HneiEdgeDic = defaultdict(list)  # key: node index (int); value: list of incident hyperedges
        self.N = len(self.Hnodes)
        i = 1  # i is the hyperedge index
        for hyperedge in self.HG:
            self.HedgeDic[i] = hyperedge
            self.HedgeSizeDic[i] = len(hyperedge)
            if len(hyperedge) > self.maxHegdeSize:
                self.maxHegdeSize = len(hyperedge)
            for node in hyperedge:
                self.HneiEdgeDic[node].append(i)
            i += 1
    
    def cal_connected_component(self):
        G = nx.Graph()  # use 2-projection to compute connected components
        for node in self.Hnodes:
            for neiedge in self.HneiEdgeDic[node]:
                for neinode in self.HedgeDic[neiedge]:
                    G.add_edge(node, neinode)
        CCs_size = [len(item) for i, item in enumerate(nx.connected_components(G))]
        return dict(Counter(CCs_size))

    def prepare_for_numba(self, IFdefinded=False):
        # 1. convert self.Hnodes to a np.array
        self.nb_Hnodes = np.array(self.Hnodes)

        # 2. generate an np.array of hyperedge IDs
        self.edgeIDs = np.array(list(self.HedgeDic.keys()))
        
        # 3. convert self.HneiEdgeDic and self.HedgeDic to numba-compatible dict formats
        self.nb_HneiEdgeDic = dict2numba(self.HneiEdgeDic)
        self.nb_HedgeDic = dict2numba(self.HedgeDic)

        # 4. convert self.HedgeSizeDic to an np.array
        self.nb_HedgeSize = np.zeros((np.max(self.edgeIDs) + 1), dtype=np.int64)
        for key, value in self.HedgeSizeDic.items():
            self.nb_HedgeSize[key] = np.int64(value)

def dict2numba(dic):
    """ Convert a Python dict into a numba-typed dict format. """
    output = nb.typed.Dict.empty(
        key_type = nb.types.int64, 
        value_type = nb.types.Array(nb.types.int64, 1, 'C')
    )
    for key, value in dic.items():
        output[key] = np.array(value, dtype=np.int64)
    return output

def cal_crt_measure_return_delnode(crthypergraph, net, mtd):
    calmeasure = CalNodesMeasures(crthypergraph, net)
    if mtd == 'random':
        delnode, timesum = calmeasure.h_random()
    if mtd == 'degree':
        delnode, timesum = calmeasure.h_degree()
    elif mtd == 'neiNodesNum':
        delnode, timesum = calmeasure.h_neiNodesNum()
    elif mtd == 'clsCloseness':
        delnode, timesum = calmeasure.h_clsCloseness()
    elif mtd == 'clsBetweenness':
        delnode, timesum = calmeasure.h_clsBetweenness()
    elif mtd == 'clsEigenvector':
        delnode, timesum = calmeasure.h_clsEigenvector()
    elif mtd == 'nodeEdgeEigenvector_linear':
        delnode, timesum = calmeasure.h_nodeEdgeEigenvector(funcGrp='linear')
    elif mtd == 'nodeEdgeEigenvector_max':
        delnode, timesum = calmeasure.h_nodeEdgeEigenvector(funcGrp='max')
    elif mtd == 'clsKcore':
        delnode, timesum = calmeasure.h_clsKcore()
    elif mtd == 'KMcore_g1':
        delnode, timesum = calmeasure.h_KMcore_g1()
    elif mtd == 'KMcore_gf':
        delnode, timesum = calmeasure.h_KMcore_gf()
    elif mtd == 'new_neiNodeSum_1':
        delnode, timesum = calmeasure.h_new_neiNodeSum_1()

    return delnode, timesum

def generate_new_net(oldnet,delnode):
    newnet = []
    for edge in oldnet:
        newnet.append([n for n in edge if n != delnode])
    return newnet

@njit
def initialization_random_numba(nodes, iniNodes):
    # nodeStates: rows are node indices; column 0 = I-state flag, column 1 = S-state flag
    nodeStates = np.zeros((np.max(nodes) + 1, 2), dtype=np.int64)
    for item in nodes:
        nodeStates[item, 0] = 0
        nodeStates[item, 1] = 1
    for seed in iniNodes:
        nodeStates[seed, 0] = 1
        nodeStates[seed, 1] = 0
    return nodeStates

@njit
def calrho_numba_mc(N, Hnodes, edgeID_all, HneiEdgeDic, HedgeDic, infectionProbLst_numba, iniConfig, mu, tWait=500, tMax=1500):
    # track the counts of nodes in different states
    nodeStates = initialization_random_numba(Hnodes, iniConfig)
    iNodesNum_inHedge = np.zeros((np.max(edgeID_all) + 1), dtype=np.int64)
    for edgeID in HedgeDic:
        # use HedgeDic[edgeID] as indices to sum over nodeStates and obtain the infected count in each hyperedge
        iNodesNum_inHedge[edgeID] = np.sum(nodeStates[HedgeDic[edgeID], 0])
    I = [np.sum(nodeStates[:, 0])]
    S = [np.sum(nodeStates[:, 1])]
    countInfectedTimes = np.zeros((np.max(Hnodes) + 1), dtype=np.int64)

    t = 0

    while t < tMax:
        t += 1
        newI = []
        indexes_S = np.nonzero(nodeStates[:, 1])[0]
        for i in indexes_S:
            p = 1
            for edgeID in HneiEdgeDic[i]:
                p *= infectionProbLst_numba[iNodesNum_inHedge[edgeID]]
            if np.random.random() < 1 - p:
                newI.append(i)
        newR = []
        # directly use nonzero to obtain indices of infected nodes
        indexes_I = np.nonzero(nodeStates[:, 0])[0]
        for n_to_recover in indexes_I:
            if np.random.random() < mu:
                newR.append(n_to_recover)
        
        for n_to_recover in newR:
            nodeStates[n_to_recover, 0] = 0
            nodeStates[n_to_recover, 1] = 1
            for edgeID in HneiEdgeDic[n_to_recover]:
                iNodesNum_inHedge[edgeID] -= 1

        for n_to_infect in newI:
            nodeStates[n_to_infect, 0] = 1
            nodeStates[n_to_infect, 1] = 0
            for edgeID in HneiEdgeDic[n_to_infect]:
                iNodesNum_inHedge[edgeID] += 1
        
        if t > tWait:  # only count infections after the system reaches steady state
            for node in indexes_I:
                countInfectedTimes[node] += 1

            I.append(np.sum(nodeStates[:, 0]))
            S.append(np.sum(nodeStates[:, 1]))
        
    rho = sum(np.array(I[-100:]) / N) / 100
    finalconfig = np.nonzero(nodeStates[:, 0])[0]

    return rho, finalconfig
