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

class Simu_NCsize_SIR():
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
        i=1  # i is the hyperedge index
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

    def infectAgent(self,agent):
        self.iAgentSet.add(agent)
        self.sAgentSet.remove(agent)
        self.nodeStateDic[agent]='I'
        return 1

    def recoverAgent_R(self, agent):
        self.rAgentSet.add(agent)
        self.iAgentSet.remove(agent)
        self.nodeStateDic[agent]='R'
        return 0
    
    def immuneAgent(self, agent):
        self.rAgentSet_imm.add(agent)
        self.sAgentSet.remove(agent)
        self.nodeStateDic[agent]='R_imm'
        return 0

    def initialization(self, initial_infected, immunilized=[]):  # for SIR, initial infected nodes must be specified
        self.nodeStateDic = dict()
        self.sAgentSet = set()
        self.iAgentSet = set()
        self.rAgentSet = set()
        self.rAgentSet_imm = set()
        for n in self.Hnodes:
            self.sAgentSet.add(n)
            self.nodeStateDic[n] = 'S'
            
        self.infect_this_setup = initial_infected
        for to_infect in self.infect_this_setup:  # to_infect is the node index
            self.infectAgent(to_infect)
            self.nodeStateDic[to_infect] = 'I'

        if immunilized != []:
            self.immunilize_this_setup = immunilized
            for to_immu in self.immunilize_this_setup:
                self.immuneAgent(to_immu)
                self.nodeStateDic[to_immu] = 'R_imm'

    def initialization_random(self, randomInfect=0):
        self.nodeStateDic = dict()  # record node states; key: node, value: state
        for n in self.Hnodes:
            self.sAgentSet.add(n)
            self.nodeStateDic[n] = 'S'
        
        self.iAgentSet = set()
        self.rAgentSet = set()
            
        if randomInfect > 0:
            self.infect_this_setup = random.sample(self.Hnodes, randomInfect)
        else:
            print("error, initialization unsetted") 
        for to_infect in self.infect_this_setup:  # to_infect is the node index
            self.infectAgent(to_infect)
    
    def contagionSimulation(self, l, mu, nu, tmax=float("Inf"), timeSeries=False):
        infectionProbDic = dict()
        for sizeE in range(1, self.maxHegdeSize+1):
            infectionProbDic[sizeE] = [np.exp(-(l/sizeE)*i**nu) for i in range(0, sizeE+1)]
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
                    if self.nodeStateDic[i]=='I':
                        iNodesNum_inHedge[edgeID] += 1
            
            # infection step
            newI = []
            """
            potentialI = set()  # speed-up: preselect susceptible nodes with at least one infected neighbor
            for edgeID in iNodesNum_inHedge:  
                if iNodesNum_inHedge > 0:
                    for node in self.HedgeDic[edgeID]:
                        potentialI.add(node)
            """
            for node in self.sAgentSet:
                p=1
                for edgeID in self.HneiEdgeDic[node]:  # indices of hyperedges incident to node
                    p=p*infectionProbDic[self.HedgeSizeDic[edgeID]][iNodesNum_inHedge[edgeID]]
                if random.random() < 1-p:
                    newI.append(node)
            
            # recovery step; we cannot modify a set while iterating, so use a temporary set newR
            newR = set()
            for n_to_recover in self.iAgentSet:
                if (random.random() < mu):
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


def chooseTopNodes(hrt, netName, chooseMtd, chooseNum):  # choose nodes with the largest values of a given measure
    if chooseMtd == 'random':
        with open(f'../Networks/networks/{netName}.json', 'r') as fl:
            HyperGraph = json.load(fl)
        Hnodes = []
        for edge in HyperGraph:
            Hnodes.extend(edge)
        HnodesLst = list(set(Hnodes))
        immediatedNodes = random.sample(HnodesLst,chooseNum)
        return immediatedNodes
    else: 
        immediatedNodes = []
        with open(f"../MeasureValuesRanking/NodesMeasures/{hrt}_{chooseMtd}_{netName}_rvs.json","r") as f:
            # file is a dict: key = measure value (float), value = list of nodes (int) with that value
            msNodesDic = json.load(f)

            msList = [float(i) for i in msNodesDic.keys()]
            msList.sort(reverse=True)  # sort in descending order

            aggchoosenNodes = 0
            for ms in msList:
                currN = len(msNodesDic[str(ms)])
                aggchoosenNodes += currN
                if aggchoosenNodes < chooseNum:  # if adding all nodes with the current measure value is still not enough, add them all
                    ext = msNodesDic[str(ms)][:]
                    random.shuffle(ext)
                    immediatedNodes.extend(ext)
                else:  # randomly sample the remaining number of nodes needed from the current measure group
                    immediatedNodes.extend((random.sample(msNodesDic[str(ms)], chooseNum-aggchoosenNodes+currN)))
                    return immediatedNodes  # list of node indices
