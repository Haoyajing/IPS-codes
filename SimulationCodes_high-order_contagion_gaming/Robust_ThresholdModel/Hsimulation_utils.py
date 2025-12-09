from collections import defaultdict
import numpy as np
import os
import json
import itertools
import networkx as nx
import pandas as pd
import random
import collections
from time import time
from tqdm import tqdm
import pickle as pk
import math

class SimuThresholdContagion_SIR():
    def __init__(self, HGraph):
        self.HG = HGraph  # list of hyperedges; each hyperedge is a list of nodes
        temp = []
        for i in self.HG:
            temp.extend(i)
        nodeSet = set(temp)
        self.Hnodes = list(nodeSet)
        self.Hnodes.sort()  # list of hypergraph node indices (sorted ascending)
        self.HedgeDic = dict()  # key: hyperedge index (int); value: hyperedge (list of nodes)
        self.HedgeSizeDic = dict()  # key: hyperedge index (int); value: size of the hyperedge
        self.maxHegdeSize = 0
        self.HneiEdgeDic = defaultdict(list)  # key: node index (int); value: list of incident hyperedges
        self.N = len(self.Hnodes)
        i = 1  # hyperedge index
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

    def initialization(self, initial_infected):  # SIR requires explicitly specifying initial infected nodes
        self.nodeStateDic = dict()
        self.sAgentSet = set()
        self.iAgentSet = set()
        self.rAgentSet = set()
        for n in self.Hnodes:
            self.sAgentSet.add(n)
            self.nodeStateDic[n] = 'S'
            
        self.infect_this_setup = initial_infected
        for to_infect in self.infect_this_setup:  # to_infect is node index
            self.infectAgent(to_infect)
            self.nodeStateDic[to_infect] = 'I'

    def initialization_random(self, randomInfect=0):
        self.nodeStateDic = dict()  # record node status; key: node, value: state
        self.sAgentSet = set()
        self.iAgentSet = set()
        self.rAgentSet = set()
        self.iNodesNum_inHedge = defaultdict(int)  # key: hyperedge index; value: number of infected nodes
        
        for n in self.Hnodes:
            self.sAgentSet.add(n)
            self.nodeStateDic[n] = 'S'
        
        if randomInfect > 0:
            self.infect_this_setup = random.sample(self.Hnodes, randomInfect)
        else:
            print("error, initialization unset")
        for to_infect in self.infect_this_setup:  # to_infect is node index
            self.infectAgent(to_infect)
            for neiEdge in self.HneiEdgeDic[to_infect]:
                self.iNodesNum_inHedge[neiEdge] += 1

    def contagionSimulation(self, theta, lbd, miu, tmax=float("Inf"), timeSeries=False):
        I = [len(self.iAgentSet)]
        R = [len(self.rAgentSet)]
        S = [len(self.sAgentSet)]
        t = 0
        iNodesNum_inHedge = {}

        while t < tmax and I[-1] != 0:
            t += 1

            # infection step
            newI = set()
            for edgeID in self.HedgeDic:
                iNodesNum_inHedge[edgeID] = 0
                SnodesLst = []
                for i in self.HedgeDic[edgeID]:
                    if self.nodeStateDic[i] == 'I':
                        iNodesNum_inHedge[edgeID] += 1
                    elif self.nodeStateDic[i] == 'S':
                        SnodesLst.append(i)

                # threshold-based infection rule
                if (
                    len(SnodesLst) > 0 
                    and iNodesNum_inHedge[edgeID] >= math.ceil(theta * len(self.HedgeDic[edgeID]))
                    and random.random() <= lbd
                ):
                    for node in SnodesLst:
                        newI.add(node)
            
            # recovery step; cannot modify the set while iterating, so newR is used
            newR = set()
            for n_to_recover in self.iAgentSet:
                if random.random() < miu:
                    newR.add(n_to_recover)
            for n_to_recover in newR:
                self.recoverAgent_R(n_to_recover)
            
            # update node states
            for n_to_infect in newI:
                self.infectAgent(n_to_infect)

            I.append(len(self.iAgentSet))
            R.append(len(self.rAgentSet))
            S.append(len(self.sAgentSet))

        if timeSeries:
            return R, t, I, S
        else:
            return R[-1], t, I[-1], S[-1]


def chooseTopNodes(hrt, netName, chooseMtd, chooseNum):  # choose nodes with the highest measure values
    if chooseMtd == 'random':
        with open(f'../Networks/networks/{netName}.json', 'r') as fl:
            HyperGraph = json.load(fl)
        Hnodes = []
        for edge in HyperGraph:
            Hnodes.extend(edge)
        HnodesLst = list(set(Hnodes))
        immediatedNodes = random.sample(HnodesLst, chooseNum)
        return immediatedNodes

    else: 
        immediatedNodes = []
        with open(f"../MeasureValuesRanking/NodesMeasures/{hrt}_{chooseMtd}_{netName}_rvs.json","r") as f:
            # file is a dict: key = measure value (float); value = list of nodes sharing that value
            msNodesDic = json.load(f)

            msList = [float(i) for i in msNodesDic.keys()]
            msList.sort(reverse=True)  # descending order

            aggchoosenNodes = 0
            for ms in msList:
                currN = len(msNodesDic[str(ms)])
                aggchoosenNodes += currN

                if aggchoosenNodes < chooseNum:
                    ext = msNodesDic[str(ms)][:]
                    random.shuffle(ext)
                    immediatedNodes.extend(ext)

                else:
                    immediatedNodes.extend(
                        random.sample(msNodesDic[str(ms)], chooseNum - aggchoosenNodes + currN)
                    )
                    return immediatedNodes  # list of node indices
