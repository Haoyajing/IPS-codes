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

class SimuThresholdContagion_SIR():
    def __init__(self, HGraph):
        self.HG = HGraph  
        temp = []
        for i in self.HG:
            temp.extend(i)
        nodeSet = set(temp)
        self.Hnodes = list(nodeSet)
        self.Hnodes.sort()  
        self.HedgeDic = dict()  
        self.HedgeSizeDic = dict()  
        self.maxHegdeSize = 0
        self.HneiEdgeDic = defaultdict(list)  
        self.N = len(self.Hnodes)
        i=1 
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

    def initialization(self, initial_infected): 
        self.nodeStateDic = dict()
        self.sAgentSet = set()
        self.iAgentSet = set()
        self.rAgentSet = set()
        for n in self.Hnodes:
            self.sAgentSet.add(n)
            self.nodeStateDic[n] = 'S'
            
        self.infect_this_setup = initial_infected
        for to_infect in self.infect_this_setup:  
            self.infectAgent(to_infect)
            self.nodeStateDic[to_infect] = 'I'

    def initialization_random(self, randomInfect=0):
        self.nodeStateDic = dict()  
        self.sAgentSet = set()
        self.iAgentSet = set()
        self.rAgentSet = set()
        for n in self.Hnodes:
            self.sAgentSet.add(n)
            self.nodeStateDic[n] = 'S'
        
        if randomInfect > 0:
            self.infect_this_setup = random.sample(self.Hnodes, randomInfect)
        else:
            print("error, initialization unsetted") 
        for to_infect in self.infect_this_setup:  
            self.infectAgent(to_infect)


    def contagionSimulation(self, theta, lbd, miu, tmax=float("Inf"), timeSeries=False):
        I = [len(self.iAgentSet)]
        R = [len(self.rAgentSet)]
        S = [len(self.sAgentSet)]
        t = 0
        iNodesNum_inHedge = {}
        while t < tmax and I[-1] != 0:
            t += 1
            # infection
            newI = set()
            for edgeID in self.HedgeDic:
                iNodesNum_inHedge[edgeID] = 0
                SnodesLst = []
                for i in self.HedgeDic[edgeID]:
                    if self.nodeStateDic[i]=='I':
                        iNodesNum_inHedge[edgeID] += 1
                    elif self.nodeStateDic[i]=='S':
                        SnodesLst.append(i)
                # np.random.seed(12)
                # rd = np.random.random()
                # print(rd)
                if len(SnodesLst)>0 and (iNodesNum_inHedge[edgeID]>=math.ceil(theta*len(self.HedgeDic[edgeID]))) and (np.random.random() <= lbd):
                    for node in SnodesLst:
                        newI.add(node)
            # print('t=', t, newI)
            
            # recover
            newR = set()
            for n_to_recover in self.iAgentSet:
                # np.random.seed(18)
                # rd = np.random.random()
                # print(rd)
                if (np.random.random() < miu):
                    newR.add(n_to_recover)
            for n_to_recover in newR:        
                self.recoverAgent_R(n_to_recover)
            # print('t=', t,newR)
            # 
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
        # 1. 
        # 
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

        # 2. 
        edgeIDs = list(self.HedgeDic.keys())
        iNodesNum_inHedge = np.zeros((np.max(edgeIDs) + 1), dtype=np.int64)
        self.iNodesNum_inHedge = iNodesNum_inHedge  

        # 3. 
        self.nb_HedgeDic = dict2numba(self.HedgeDic)

        # 4. 
        self.nb_HedgeSize = np.zeros((np.max(edgeIDs) + 1), dtype=np.int64)
        for key, value in self.HedgeSizeDic.items():
            self.nb_HedgeSize[key] = np.int64(value)

def dict2numba(dic):
    """ numba
    """
    output = nb.typed.Dict.empty(
        key_type = nb.types.int64, 
        value_type = nb.types.Array(nb.types.int64, 1, 'C')
    )
    for key, value in dic.items():
        output[key] = np.array(value, dtype=np.int64)
    return output

@njit
def contagion_simulation_numba(nodeStates, iNodesNum_inHedge, HedgeDic, HedgeSize, theta, lbd, mu, tmax=float("Inf")):
    # nodes
    I = [np.sum(nodeStates[:, 0])]
    R = [np.sum(nodeStates[:, 1])]
    S = [np.sum(nodeStates[:, 2])]
    t = 0

    while t < tmax and I[-1] != 0:
        t += 1
        newI = []
        for edgeID in HedgeDic:
            # infected number
            iNodesNum_inHedge[edgeID] = np.sum(nodeStates[HedgeDic[edgeID], 0])
            # s
            sNodes_inHedge = HedgeDic[edgeID][np.nonzero(nodeStates[HedgeDic[edgeID], 2])[0]] 
            sNodeNum = len(sNodes_inHedge)
            # np.random.seed(12)
            # rd = np.random.random()
            # print(rd)
            if sNodeNum>0 and (iNodesNum_inHedge[edgeID]>=math.ceil(theta*HedgeSize[edgeID])) and (np.random.random() <= lbd):
                for snode in sNodes_inHedge:
                    newI.append(snode)
        # print('t=', t, newI)
        newR = []
        # infected
        indexes = np.nonzero(nodeStates[:, 0])[0]
        for n_to_recover in indexes:
            # np.random.seed(18)
            # rd = np.random.random()
            # print(rd)
            if np.random.random() < mu:
                newR.append(n_to_recover)
        # print('t=', t, newR)
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

        # print(I[-1], R[-1], S[-1])
    # return R, t, I, S
    return R[-1], t, I[-1], S[-1], I[1]