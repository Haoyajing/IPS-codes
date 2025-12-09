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

class SimuNonlinearContagion_SIR():
    def __init__(self, HGraph):
        self.HG = HGraph  # list of hyperedges, each hyperedge is a list of nodes
        temp = []
        for i in self.HG:
            temp.extend(i)
        nodeSet = set(temp)
        self.Hnodes = list(nodeSet)
        self.Hnodes.sort()  # list of hypergraph nodes (int), sorted in ascending order
        self.HedgeDic = dict()  # key: hyperedge index (int); value: hyperedge as list of nodes
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

        self.iAgentSet = set()
        self.sAgentSet = set()
        self.rAgentSet = set()

    def infectAgent(self,agent):
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

    def initialization(self, initial_infected, immunilized=[]):  # SIR: initial infected nodes must be specified
        self.nodeStateDic = dict()
        self.sAgentSet = set()
        self.iAgentSet = set()
        self.rAgentSet = set()
        self.rAgentSet_imm = set()
        for n in self.Hnodes:
            self.sAgentSet.add(n)
            self.nodeStateDic[n] = 'S'
            
        self.infect_this_setup = initial_infected
        for to_infect in self.infect_this_setup:  # to_infect is node index
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
        for to_infect in self.infect_this_setup:  # to_infect is node index
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
                p = 1
                for edgeID in self.HneiEdgeDic[node]:  # indices of hyperedges incident to node
                    p = p * infectionProbLst[iNodesNum_inHedge[edgeID]]
                if random.random() < 1-p:
                    newI.append(node)
            
            # recovery step; we cannot modify a set while iterating over it, so use a temporary set newR
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

    
    def contagionSimulation_details(self, l, mu, nu, tmax=float("Inf")):
        pass

class SimuNonlinearContagion_SIS():
    def __init__(self, HGraph):
        self.HG = HGraph  # list of hyperedges, each hyperedge is a list of nodes
        temp = []
        for i in self.HG:
            temp.extend(i)
        nodeSet = set(temp)
        self.Hnodes = list(nodeSet)
        self.Hnodes.sort()  # list of hypergraph node indices (int), sorted in ascending order
        self.N = len(self.Hnodes)
        self.HedgeDic = dict()  # key: hyperedge index (int); value: hyperedge as list of nodes
        self.HedgeSizeDic = dict()  # key: hyperedge index (int); value: hyperedge size (int)
        self.maxHegdeSize = 0
        self.HneiEdgeDic = defaultdict(list)  # key: node index (int); value: list of incident hyperedges
        i = 1  # i is the hyperedge index
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
        self.iNodesNum_inHedge = {}  # key: hyperedge index; value: number of infected nodes in the hyperedge

    def infectAgent(self,agent):
        self.iAgentSet.add(agent)
        self.sAgentSet.remove(agent)
        self.nodeStateDic[agent] = 'I'
        return 1

    def recoverAgent_S(self, agent):
        self.sAgentSet.add(agent)
        self.iAgentSet.remove(agent)
        self.nodeStateDic[agent] = 'S'
        return 0

    def initialization_random(self, randomInfect=0):
        self.nodeStateDic = dict()  # record node states; key: node, value: state
        self.iNodesNum_inHedge = defaultdict(int)
        for n in self.Hnodes:
            self.sAgentSet.add(n)
            self.nodeStateDic[n] = 'S'
        
        self.iAgentSet = set()
            
        if randomInfect > 0:
            self.infect_this_setup = random.sample(self.Hnodes, randomInfect)
        else:
            print("error, initialization unsetted") 
        for to_infect in self.infect_this_setup:  # to_infect is node index
            self.infectAgent(to_infect)
            for neiE in self.HneiEdgeDic[to_infect]:
                self.iNodesNum_inHedge[neiE] += 1

    def initialization_defined(self, definiteInfect=[]):
        self.nodeStateDic = dict()  # record node states; key: node, value: state
        self.iNodesNum_inHedge = defaultdict(int)
        for n in self.Hnodes:
            self.sAgentSet.add(n)
            self.nodeStateDic[n] = 'S'
        
        self.iAgentSet = set()
            
        if definiteInfect != []:
            self.infect_this_setup = definiteInfect
        else:
            print("error, initialization unsetted") 
        for to_infect in self.infect_this_setup:  # to_infect is node index
            self.infectAgent(to_infect)
            for neiE in self.HneiEdgeDic[to_infect]:
                self.iNodesNum_inHedge[neiE] += 1

    def contagionSimulation(self, nu, l, mu, iniI=1, tMax=3*float(1e3), tWait=500, tWindow=1000, tCheck=10, errorbar=0.05):
        # iniI: number (or list) of initially infected nodes
        # tMax: maximum simulation time
        # tWait: minimum transient time before checking stationarity
        # tWindow: time window used to compute the steady-state infection level
        # tCheck: window length used to check stationarity (based on the last tCheck values)
        infectionProbLst = [np.exp(-l*i**nu) for i in range(0, self.maxHegdeSize)]
        I = [len(self.iAgentSet)]
        S = [len(self.sAgentSet)]
        countInfectedTimes = {i:0 for i in self.Hnodes}
        t = 0
        t_stable = 0  # once stationarity is reached, run for tWindow more steps and then stop
        IFstable = False

        while t <= tMax:
            t += 1
            if I[-1] == 0:  # contagion died out; reinitialize with new seeds and reset the time series
                if type(iniI) == int:
                    self.initialization_random(randomInfect=iniI)
                elif type(iniI) == list:
                    self.initialization_defined(definiteInfect=iniI)
                else:
                    return 'error'
                I = [len(self.iAgentSet)]
                S = [len(self.sAgentSet)]
                # print('contagion vanish, current I and S are:', I, S)
                countInfectedTimes = {i:0 for i in self.Hnodes}  # could also store a list of times instead of an integer
                t = 0
                t_stable = 0
                IFstable = False
                continue
            elif not IFstable:  # contagion has not reached stationarity
                if t <= tWait:  # before tWait we do not test stationarity to ensure a minimum runtime
                    newI = []
                    for node in self.sAgentSet:
                        p = 1
                        for edgeID in self.HneiEdgeDic[node]:  # indices of hyperedges incident to node
                            p = p * infectionProbLst[self.iNodesNum_inHedge[edgeID]]
                        if random.random() < 1-p:
                            newI.append(node)
                elif np.max(np.abs(I[len(I)-tCheck:len(I)]-np.mean(I[len(I)-tCheck:len(I)]))) < errorbar*self.N:  # after tWait, check stationarity
                    IFstable = True  # start recording steady-state data within the specified window
                    continue
                else:  # after tWait but stationarity has not been reached yet
                    newI = []
                    for node in self.sAgentSet:
                        p = 1
                        for edgeID in self.HneiEdgeDic[node]:  # indices of hyperedges incident to node
                            p = p * infectionProbLst[self.iNodesNum_inHedge[edgeID]]
                        if random.random() < 1-p:
                            newI.append(node)
            else:  # stationarity has been reached (I[-1] != 0 and IFstable is True)
                t_stable += 1
                newI = []
                for node in self.sAgentSet:
                    p = 1
                    for edgeID in self.HneiEdgeDic[node]:  # indices of hyperedges incident to node
                        p = p * infectionProbLst[self.iNodesNum_inHedge[edgeID]]
                    if random.random() < 1-p:
                        newI.append(node)
            
            # recovery step; we cannot modify a set while iterating, so use a temporary set newR
            newR = set()
            for n_to_recover in self.iAgentSet:
                if (random.random() < mu):
                    newR.add(n_to_recover)
            for n_to_recover in newR:        
                self.recoverAgent_S(n_to_recover)
                for neiEdge in self.HneiEdgeDic[n_to_recover]:  # update iNodesNum_inHedge; these recovered nodes reduce infected count in each incident hyperedge
                    self.iNodesNum_inHedge[neiEdge] -= 1
            
            # update node states
            for n_to_infect in newI:
                self.infectAgent(n_to_infect)
                for neiEdge in self.HneiEdgeDic[n_to_infect]:  # update iNodesNum_inHedge; newly infected nodes increase infected count in incident hyperedges
                    self.iNodesNum_inHedge[neiEdge] += 1
            I.append(len(self.iAgentSet))
            S.append(len(self.sAgentSet))
            if 1 <= t_stable and t_stable <= tWindow:  # ensure only steady-state steps are counted
                for node in self.iAgentSet:
                    countInfectedTimes[node] += 1
            # if t%100 == 0:
            #     print('currentTime', t, IFstable, 'InodesNum', I[-1])
            if t_stable == tWindow:
                break
        if t == tMax:
            return f'maxTime reached! iFstable={IFstable} t_stable={t_stable}(/{tWindow})'
            # At this point the system may have reached steady state but tMax was not sufficient to accumulate tWindow steady-state steps;
            # it is also possible that the system has not reached stationarity.
        elif t_stable == tWindow:
            pass
            # print(f"finalTime={t}, end correct~~")
        else:
            return 'something wrong...'
        # return a dict: key = node, value = number of times this node has been infected within the window; 
        # if tMax is reached before stationarity, the function returns a string above
        return countInfectedTimes

    def contagionSimuTmieSeries(self, nu, l, mu, iniI, tWait=500, tMaxInitalized=10, tMax=1500):
        infectionProbLst = [np.exp(-l*i**nu) for i in range(0, self.maxHegdeSize)]
        I = [len(self.iAgentSet)]
        S = [len(self.sAgentSet)]
        countInfectedTimes = {i:0 for i in self.Hnodes}
        t = 0
        t_initalized = 0

        while t <= tMax:
            t += 1
            if I[-1] == 0:  # contagion died out; reinitialize with new seeds and reset the time series
                if t_initalized <= tMaxInitalized:
                    if type(iniI) == int:
                        self.initialization_random(randomInfect=iniI)
                    elif type(iniI) == list:
                        self.initialization_defined(definiteInfect=iniI)
                    else:
                        # keep the number of return values consistent with other branches
                        return 'error', 'error', 'error'
                    I = [len(self.iAgentSet)]
                    S = [len(self.sAgentSet)]
                    countInfectedTimes = {i:0 for i in self.Hnodes}
                    # print('contagion vanish, current I and S are:', I, S)
                    t = 0
                    t_initalized += 1
                    # print('continue')
                    continue
                else:
                    # print('break')
                    return countInfectedTimes, I, S
            else:
                newI = []
                for node in self.sAgentSet:
                    p = 1
                    for edgeID in self.HneiEdgeDic[node]:  # indices of hyperedges incident to node
                        p = p * infectionProbLst[self.iNodesNum_inHedge[edgeID]]
                    if random.random() < 1-p:
                        newI.append(node)
            
            # recovery step; we cannot modify a set while iterating, so use a temporary set newR
            newR = set()
            for n_to_recover in self.iAgentSet:
                if (random.random() < mu):
                    newR.add(n_to_recover)
            for n_to_recover in newR:        
                self.recoverAgent_S(n_to_recover)
                for neiEdge in self.HneiEdgeDic[n_to_recover]:  # update iNodesNum_inHedge for hyperedges containing this recovered node
                    self.iNodesNum_inHedge[neiEdge] -= 1
            
            # update node states
            for n_to_infect in newI:
                self.infectAgent(n_to_infect)
                for neiEdge in self.HneiEdgeDic[n_to_infect]:  # update iNodesNum_inHedge for hyperedges containing this newly infected node
                    self.iNodesNum_inHedge[neiEdge] += 1
            I.append(len(self.iAgentSet))
            S.append(len(self.sAgentSet))
            if t >= tWait:  # only count infections after tWait (steady-state part)
                for node in self.iAgentSet:
                    countInfectedTimes[node] += 1
        # print('return')    
        return countInfectedTimes, I, S

class SimuThresholdContagion_SIR():
    def __init__(self, HGraph):
        self.HG = HGraph  # list of hyperedges, each hyperedge is a list of nodes
        temp = []
        for i in self.HG:
            temp.extend(i)
        nodeSet = set(temp)
        self.Hnodes = list(nodeSet)
        self.Hnodes.sort()  # list of hypergraph node indices (int), sorted in ascending order
        self.HedgeDic = dict()  # key: hyperedge index (int); value: hyperedge as list of nodes
        self.HedgeSizeDic = dict()  # key: hyperedge index (int); value: hyperedge size (int)
        self.maxHegdeSize = 0
        self.HneiEdgeDic = defaultdict(list)  # key: node index (int); value: list of incident hyperedges
        self.N = len(self.Hnodes)
        i = 1  # i is hyperedge index
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
        self.nodeStateDic[agent] = 'I'
        return 1

    def recoverAgent_R(self, agent):
        self.rAgentSet.add(agent)
        self.iAgentSet.remove(agent)
        self.nodeStateDic[agent] = 'R'
        return 0

    def initialization(self, initial_infected):  # SIR: initial infected nodes must be specified
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
        self.nodeStateDic = dict()  # record node states; key: node, value: state
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
            print("error, initialization unsetted") 
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
                if len(SnodesLst) > 0 and \
                   (iNodesNum_inHedge[edgeID] >= math.ceil(theta*len(self.HedgeDic[edgeID]))) and \
                   (random.random() <= lbd):
                    for node in SnodesLst:
                        newI.add(node)
            
            # recovery step; we cannot modify a set while iterating, so use a temporary set newR
            newR = set()
            for n_to_recover in self.iAgentSet:
                if (random.random() < miu):
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

class SimuThresholdContagion_SIS():
    def __init__(self, HGraph):
        self.HG = HGraph  # list of hyperedges, each hyperedge is a list of nodes
        temp = []
        for i in self.HG:
            temp.extend(i)
        nodeSet = set(temp)
        self.Hnodes = list(nodeSet)
        self.Hnodes.sort()  # list of hypergraph node indices (int), sorted in ascending order
        self.HedgeDic = dict()  # key: hyperedge index (int); value: hyperedge as list of nodes
        self.HedgeSizeDic = dict()  # key: hyperedge index (int); value: hyperedge size (int)
        self.maxHegdeSize = 0
        self.HneiEdgeDic = defaultdict(list)  # key: node index (int); value: list of incident hyperedges
        self.N = len(self.Hnodes)
        i = 1  # i is hyperedge index
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
        self.nodeStateDic[agent] = 'I'
        return 1

    def recoverAgent_S(self, agent):
        self.sAgentSet.add(agent)
        self.iAgentSet.remove(agent)
        self.nodeStateDic[agent] = 'S'
        return 0
    
    def inmmuneAgent(self, agent):
        self.rAgentSet.add(agent)
        self.sAgentSet.remove(agent)
        self.nodeStateDic[agent] = 'R'
        return 0

    def initialization_random(self, randomInfect=0):
        self.nodeStateDic = dict()  # record node states; key: node, value: state
        self.iNodesNum_inHedge = defaultdict(int)  # key: hyperedge index; value: number of infected nodes
        for n in self.Hnodes:
            self.sAgentSet.add(n)
            self.nodeStateDic[n] = 'S'
        
        self.iAgentSet = set()
            
        if randomInfect > 0:
            self.infect_this_setup = random.sample(self.Hnodes, randomInfect)
        else:
            print("error, initialization unsetted") 
        for to_infect in self.infect_this_setup:  # to_infect is node index
            self.infectAgent(to_infect)
            for neiEdge in self.HneiEdgeDic[to_infect]:
                self.iNodesNum_inHedge[neiEdge] += 1

    def initialization_defined(self, definiteInfect=[]):
        self.nodeStateDic = dict()  # record node states; key: node, value: state
        self.iNodesNum_inHedge = defaultdict(int)  # key: hyperedge index; value: number of infected nodes
        for n in self.Hnodes:
            self.sAgentSet.add(n)
            self.nodeStateDic[n] = 'S'
        
        self.iAgentSet = set()
            
        if definiteInfect != []:
            self.infect_this_setup = definiteInfect
        else:
            print("error, initialization unsetted") 
        for to_infect in self.infect_this_setup:  # to_infect is node index
            self.infectAgent(to_infect)
            for neiEdge in self.HneiEdgeDic[to_infect]:
                self.iNodesNum_inHedge[neiEdge] += 1


    def contagionSimulation(self, theta, lbd, mu, iniI=1, tMax=3*float(1e3), tWait=500, tWindow=1000, tCheck=10, errorbar=0.05):
        # iniI: number (or list) of initially infected nodes
        # tMax: maximum simulation time
        # tWait: minimum transient time before checking stationarity
        # tWindow: time window used to compute the steady-state infection level
        # tCheck: window length used to check stationarity (based on the last tCheck values)
        I = [len(self.iAgentSet)]
        S = [len(self.sAgentSet)]
        countInfectedTimes = {i:0 for i in self.Hnodes}
        t = 0
        t_stable = 0  # once stationarity is reached, run for tWindow more steps and then stop
        IFstable = False
        t_initalized = 0
        
        while t <= tMax and t_initalized <= tMax:
            t += 1
            if I[-1] == 0:  # contagion died out; reinitialize with new seeds and reset the time series
                if type(iniI) == int:
                    self.initialization_random(randomInfect=iniI)
                elif type(iniI) == list:
                    self.initialization_defined(definiteInfect=iniI)
                else:
                    return 'error'
                I = [len(self.iAgentSet)]
                S = [len(self.sAgentSet)]
                # print('contagion vanish, current I and S are:', I, S)
                countInfectedTimes = {i:0 for i in self.Hnodes}  # could store times instead of counts
                t = 0
                t_stable = 0
                IFstable = False
                t_initalized += 1
                continue
            elif not IFstable:  # contagion has not reached stationarity
                if t <= tWait:  # before tWait we do not test stationarity to ensure a minimum runtime
                    newI = set()
                    for hyperE in self.HedgeDic:
                        hElength = len(self.HedgeDic[hyperE])
                        if hElength != self.iNodesNum_inHedge[hyperE] and \
                           (self.iNodesNum_inHedge[hyperE] >= math.ceil(theta*hElength)) and \
                           (random.random() <= lbd):
                            for node in self.HedgeDic[hyperE]:
                                if self.nodeStateDic[node] == 'S':
                                    newI.add(node)
                elif np.max(np.abs(I[len(I)-tCheck:len(I)]-np.mean(I[len(I)-tCheck:len(I)]))) < errorbar*self.N:  # check stationarity
                    IFstable = True  # start recording steady-state data within the specified window
                    continue
                else:  # after tWait but stationarity has not been reached yet
                    newI = set()
                    for hyperE in self.HedgeDic:
                        hElength = len(self.HedgeDic[hyperE])
                        if hElength != self.iNodesNum_inHedge[hyperE] and \
                           (self.iNodesNum_inHedge[hyperE] >= math.ceil(theta*hElength)) and \
                           (random.random() <= lbd):
                            for node in self.HedgeDic[hyperE]:
                                if self.nodeStateDic[node] == 'S':
                                    newI.add(node)
            else:  # stationarity has been reached (I[-1] != 0 and IFstable is True)
                t_stable += 1
                newI = set()
                for hyperE in self.HedgeDic:
                    hElength = len(self.HedgeDic[hyperE])
                    if hElength != self.iNodesNum_inHedge[hyperE] and \
                       (self.iNodesNum_inHedge[hyperE] >= math.ceil(theta*hElength)) and \
                       (random.random() <= lbd):
                        for node in self.HedgeDic[hyperE]:
                            if self.nodeStateDic[node] == 'S':
                                newI.add(node)
            
            # recovery step; we cannot modify a set while iterating, so use a temporary set newR
            newR = set()
            for n_to_recover in self.iAgentSet:
                if (random.random() < mu):
                    newR.add(n_to_recover)
            for n_to_recover in newR:        
                self.recoverAgent_S(n_to_recover)
                for neiEdge in self.HneiEdgeDic[n_to_recover]:  # update iNodesNum_inHedge for hyperedges containing this recovered node
                    self.iNodesNum_inHedge[neiEdge] -= 1

            # update node states
            for n_to_infect in newI:
                self.infectAgent(n_to_infect)
                for neiEdge in self.HneiEdgeDic[n_to_infect]:
                    self.iNodesNum_inHedge[neiEdge] += 1
            I.append(len(self.iAgentSet))
            S.append(len(self.sAgentSet))
            if 1 <= t_stable and t_stable <= tWindow:  # ensure only steady-state steps are counted
                for node in self.iAgentSet:
                    countInfectedTimes[node] += 1
            # if t%100 == 0:
            #     print('currentTime', t, IFstable, 'InodesNum', I[-1])
            if t_stable == tWindow:
                break
        if t == tMax:
            # print(f'maxTime reached! iFstable={IFstable}')
            # system may have reached stationarity but tMax is not enough to get tWindow steps, or stationarity not reached at all
            return f'maxTime reached! iFstable={IFstable}'
        elif t_stable == tWindow:
            pass
            # print(f"finalTime={t}, end correct~~")
        else:
            # print('something wrong...')
            return 'something wrong...'
        # return a dict (counts), and I, S; if tMax is reached before stationarity, returns a string above
        return countInfectedTimes, I, S

    def contagionSimuTmieSeries(self, theta, lbd, mu, iniI=1, tWait=500, tMaxInitalized=10, tMax=1500):
        I = [len(self.iAgentSet)]
        S = [len(self.sAgentSet)]
        countInfectedTimes = {i:0 for i in self.Hnodes} 
        t = 0
        t_initalized = 0
        
        while t <= tMax and t_initalized <= tMax:
            t += 1
            if I[-1] == 0:  # contagion died out; reinitialize with new seeds and reset the time series
                if t_initalized <= tMaxInitalized:
                    if type(iniI) == int:
                        self.initialization_random(randomInfect=iniI)
                    elif type(iniI) == list:
                        self.initialization_defined(definiteInfect=iniI)
                    else:
                        return 'error', 'error', 'error'
                    I = [len(self.iAgentSet)]
                    S = [len(self.sAgentSet)]
                    # print('contagion vanish, current I and S are:', I, S)
                    countInfectedTimes = {i:0 for i in self.Hnodes}  # could store times instead of counts
                    t = 0
                    t_initalized += 1
                    continue
                else:
                    return countInfectedTimes, I, S
            else:
                newI = set()
                for hyperE in self.HedgeDic:
                    hElength = len(self.HedgeDic[hyperE])
                    if hElength != self.iNodesNum_inHedge[hyperE] and \
                       (self.iNodesNum_inHedge[hyperE] >= math.ceil(theta*hElength)) and \
                       (random.random() <= lbd):
                        for node in self.HedgeDic[hyperE]:
                            if self.nodeStateDic[node] == 'S':
                                newI.add(node)
            
            # recovery step; we cannot modify a set while iterating, so use a temporary set newR
            newR = set()
            for n_to_recover in self.iAgentSet:
                if (random.random() < mu):
                    newR.add(n_to_recover)
            for n_to_recover in newR:        
                self.recoverAgent_S(n_to_recover)
                for neiEdge in self.HneiEdgeDic[n_to_recover]:  # update iNodesNum_inHedge for hyperedges containing this recovered node
                    self.iNodesNum_inHedge[neiEdge] -= 1

            # update node states
            for n_to_infect in newI:
                self.infectAgent(n_to_infect)
                for neiEdge in self.HneiEdgeDic[n_to_infect]:
                    self.iNodesNum_inHedge[neiEdge] += 1
            I.append(len(self.iAgentSet))
            S.append(len(self.sAgentSet))
            if t > tWait:  # count infection frequencies from tWait+1 to tMax
                for node in self.iAgentSet:
                    countInfectedTimes[node] += 1
        
        # for the return, one can first check whether len(I) == tMax+1; otherwise, the result is invalid
        return  countInfectedTimes, I, S

class SimuDismantling():
    def __init__(self, HGraph):
        self.HG = HGraph  # list of hyperedges (list of nodes), as loaded from json
        temp = []
        for i in self.HG:
            temp.extend(i)
        nodeSet = set(temp)
        self.Hnodes = list(nodeSet)
        self.Hnodes.sort()  # list of hypergraph node indices (int), sorted in ascending order
        self.HedgeDic = dict()  # key: hyperedge index (int); value: hyperedge as list of nodes
        self.HedgeSizeDic = dict()  # key: hyperedge index (int); value: hyperedge size (int)
        self.maxHegdeSize = 0  # maximum hyperedge size
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
    
    def dismantling(self, removedNodes):
        for node in removedNodes:
            self.Hnodes.remove(node)
            for Hedge in self.HneiEdgeDic[node]:
                # keep hyperedges but remove the dismantled node; empty edges are not explicitly deleted
                self.HedgeDic[Hedge].remove(node)
        
        for Hedge in self.HedgeDic:
            self.HedgeSizeDic[Hedge] = len(self.HedgeDic[Hedge])
        
        self.HneiEdgeDic = defaultdict(list)
        for Hedge in self.HedgeDic:
            for node in self.HedgeDic[Hedge]:
                self.HneiEdgeDic[node].append(Hedge)
          
        self.maxHegdeSize = 0
        for es in self.HedgeSizeDic:
            if self.HedgeSizeDic[es] > self.maxHegdeSize:
                self.maxHegdeSize = self.HedgeSizeDic[es]

        
    def calLCCsize_CCfre(self):
        # compute the largest connected component via 2-projection
        G = nx.Graph()
        for node in self.Hnodes:
            for neiedge in self.HneiEdgeDic[node]:
                for neinode in self.HedgeDic[neiedge]:
                    G.add_edge(node, neinode)
        G.remove_edges_from(nx.selfloop_edges(G))
        CCs = list(nx.connected_components(G))
        maxCC = max(CCs, key=len, default=set())
        CCs_size = [len(i) for i in CCs]
        CCs_fre = collections.Counter(CCs_size)

        return len(maxCC), CCs_fre

    def calLSCC_size(self, sLst=[]):
        # compute s-line graph, find all connected components, then map back to the original hypergraph and measure component sizes
        # return a dict: key = s, value = size of the largest connected component in the original hypergraph
        smaxCC = dict()
        if sLst == []:
            sLst = [i+2 for i in range(self.maxHegdeSize-1)]
        for s in sLst:  # s is hyperedge size (order = s-1); minimum size is 2, hence +2 above
            # for each s: build the s-line graph, find connected components; map back to the original hypergraph to compute component sizes
            print(s)
            LG = nx.Graph()
            for hedge in self.HedgeDic:
                if len(self.HedgeDic[hedge]) >= s:
                    neiEdges = set()
                    for node in self.HedgeDic[hedge]:  # find all incident hyperedges
                        neiEdges.update(self.HneiEdgeDic[node])
                else:
                    continue
                for neihdg in neiEdges:
                    if len(set(self.HedgeDic[hedge]) & set(self.HedgeDic[neihdg])) >= s:
                        LG.add_edge(hedge, neihdg)
                LG.add_node(hedge)  # ensure isolated hyperedges are also added
            LSCCs_LineG = list(nx.connected_components(LG))  # list of connected components (each is a set of hyperedge indices)
            szLst = []  # for each component, number of nodes in the corresponding component in the original hypergraph
            for lc in LSCCs_LineG:
                if len(lc) > 1:
                    print(lc)
                lcsz = set()
                for hedge in lc:
                    lcsz.update(set(self.HedgeDic[hedge]))
                szLst.append(len(lcsz))
            smaxCC[s] = max(szLst)
        return smaxCC, szLst

    def calEdgeSizeFrequency_total(self):
        # compute hyperedge size frequency in the full hypergraph (zeros are allowed)
        sizeFrq_tt = defaultdict(int)
        for heg in self.HedgeSizeDic:
            sizeFrq_tt[self.HedgeSizeDic[heg]] += 1
        # ineractions_tt = len(self.HedgeSizeDic)  # total number of hyperedges
        # print(ineractions_tt)
        # for f in sizeFrq_tt:
        #     sizeFrq_tt[f] = sizeFrq_tt[f]/ineractions_tt
        
        return sizeFrq_tt

class SimuFullEDismantling():
    def __init__(self, HGraph):
        self.HG = HGraph  # list of hyperedges (list of nodes), as loaded from json
        temp = []
        for i in self.HG:
            temp.extend(i)
        nodeSet = set(temp)
        self.Hnodes = list(nodeSet)
        self.Hnodes.sort()  # list of hypergraph node indices (int), sorted in ascending order
        self.HedgeDic = dict()  # key: hyperedge index (int); value: hyperedge as list of nodes
        self.HedgeSizeDic = dict()  # key: hyperedge index (int); value: hyperedge size (int)
        self.maxHegdeSize = 0  # maximum hyperedge size
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
    
    def fullEdismantling(self, removedNodes):
        for node in removedNodes:
            for Hedge in self.HneiEdgeDic[node]:
                # remove all hyperedges involving this node by clearing them
                self.HedgeDic[Hedge] = []

        self.node = []
        for Hedge in self.HedgeDic:
            self.HedgeSizeDic[Hedge] = len(self.HedgeDic[Hedge])
            self.node.extend(self.HedgeDic[Hedge])
        temp_nodeSet = set(self.node)
        self.Hnodes = list(temp_nodeSet)
        self.Hnodes.sort()
        
        self.HneiEdgeDic = defaultdict(list)
        # self.Hnodes and self.HneiEdgeDic contain only the remaining nodes in the current (residual) network;
        # isolated nodes created solely by edge removal are not kept, which is reasonable when focusing on the largest connected component
        for Hedge in self.HedgeDic:
            for node in self.HedgeDic[Hedge]:
                self.HneiEdgeDic[node].append(Hedge)
          
        self.maxHegdeSize = 0
        for es in self.HedgeSizeDic:
            if self.HedgeSizeDic[es] > self.maxHegdeSize:
                self.maxHegdeSize = self.HedgeSizeDic[es]
        pass
        
    def calLCCsize_CCfre(self):
        # compute the largest connected component via 2-projection
        G = nx.Graph()
        for node in self.Hnodes:
            for neiedge in self.HneiEdgeDic[node]:
                for neinode in self.HedgeDic[neiedge]:
                    G.add_edge(node, neinode)
        G.remove_edges_from(nx.selfloop_edges(G))
        CCs = list(nx.connected_components(G))
        maxCC = max(CCs, key=len, default=set())
        CCs_size = [len(i) for i in CCs]
        CCs_fre = collections.Counter(CCs_size)

        return len(maxCC), CCs_fre

    def calLSCC_size(self, sLst=[]):
        # compute s-line graph, find all connected components, then map back to the original hypergraph and measure component sizes
        # return a dict: key = s, value = size of the largest connected component in the original hypergraph
        smaxCC = dict()
        if sLst == []:
            sLst = [i+2 for i in range(self.maxHegdeSize-1)]
        for s in sLst:  # s is hyperedge size (order = s-1); minimum size is 2, hence +2 above
            # for each s: build the s-line graph, find connected components; map back to the original hypergraph to compute component sizes
            print(s)
            LG = nx.Graph()
            for hedge in self.HedgeDic:
                if len(self.HedgeDic[hedge]) >= s:
                    neiEdges = set()
                    for node in self.HedgeDic[hedge]:  # find all incident hyperedges
                        neiEdges.update(self.HneiEdgeDic[node])
                else:
                    continue
                for neihdg in neiEdges:
                    if len(set(self.HedgeDic[hedge]) & set(self.HedgeDic[neihdg])) >= s:
                        LG.add_edge(hedge, neihdg)
                LG.add_node(hedge)  # ensure isolated hyperedges are also added
            LSCCs_LineG = list(nx.connected_components(LG))  # list of connected components (each is a set of hyperedge indices)
            szLst = []  # for each component, number of nodes in the corresponding component in the original hypergraph
            for lc in LSCCs_LineG:  # map back to the original hypergraph and count nodes
                if len(lc) > 1:
                    print(lc)
                lcsz = set()
                for hedge in lc:
                    lcsz.update(set(self.HedgeDic[hedge]))
                szLst.append(len(lcsz))
            smaxCC[s] = max(szLst)
        return smaxCC, szLst

    def calEdgeSizeFrequency_total(self):
        # compute hyperedge size frequency in the full hypergraph (zeros are allowed)
        sizeFrq_tt = defaultdict(int)
        for heg in self.HedgeSizeDic:
            sizeFrq_tt[self.HedgeSizeDic[heg]] += 1
        # ineractions_tt = len(self.HedgeSizeDic)  # total number of hyperedges
        # print(ineractions_tt)
        # for f in sizeFrq_tt:
        #     sizeFrq_tt[f] = sizeFrq_tt[f]/ineractions_tt

class SimuFullEDismantling_add_theta():
    def __init__(self, HGraph, theta):
        self.HG = HGraph  # list of hyperedges (list of nodes), as loaded from json
        temp = []
        for i in self.HG:
            temp.extend(i)
        nodeSet = set(temp)
        self.Hnodes = list(nodeSet)
        self.Hnodes.sort()  # list of hypergraph node indices (int), sorted in ascending order
        self.HedgeDic = dict()  # key: hyperedge index (int); value: hyperedge as list of nodes
        self.HedgeSizeDic = dict()  # key: hyperedge index (int); value: hyperedge size (int)
        self.HedgeSizeDic_fix = dict()
        self.maxHegdeSize = 0  # maximum hyperedge size
        self.HneiEdgeDic = defaultdict(list)  # key: node index (int); value: list of incident hyperedges
        self.N = len(self.Hnodes)
        i = 1  # i is the hyperedge index
        for hyperedge in self.HG:
            self.HedgeDic[i] = hyperedge
            self.HedgeSizeDic[i] = len(hyperedge)
            self.HedgeSizeDic_fix[i] = len(hyperedge)
            if len(hyperedge) > self.maxHegdeSize:
                self.maxHegdeSize = len(hyperedge)
            for node in hyperedge:
                self.HneiEdgeDic[node].append(i)
            i += 1
        # number of removed nodes in each hyperedge, counting only nodes removed from the current network
        self.NumofRemovedNodesinEdge = {i:0 for i in self.HedgeDic}
        self.theta = theta

    def fullEdismantling_theta(self, removedNodes):
        for node in removedNodes:
            for Hedge in self.HneiEdgeDic[node]:
                self.NumofRemovedNodesinEdge[Hedge] += 1
                if self.NumofRemovedNodesinEdge[Hedge] >= self.theta*self.HedgeSizeDic_fix[Hedge]:
                    # clear this hyperedge once the fraction of removed nodes exceeds theta
                    self.HedgeDic[Hedge] = []

        self.node = []
        for Hedge in self.HedgeDic:
            self.HedgeSizeDic[Hedge] = len(self.HedgeDic[Hedge])
            self.node.extend(self.HedgeDic[Hedge])
        temp_nodeSet = set(self.node)
        self.Hnodes = list(temp_nodeSet)
        self.Hnodes.sort()
        
        self.HneiEdgeDic = defaultdict(list)
        # self.Hnodes and self.HneiEdgeDic contain only the remaining nodes in the current (residual) network;
        # isolated nodes created solely by edge removal are not kept, which is reasonable when focusing on the largest connected component
        for Hedge in self.HedgeDic:
            for node in self.HedgeDic[Hedge]:
                self.HneiEdgeDic[node].append(Hedge)
          
        self.maxHegdeSize = 0
        for es in self.HedgeSizeDic:
            if self.HedgeSizeDic[es] > self.maxHegdeSize:
                self.maxHegdeSize = self.HedgeSizeDic[es]
        pass
        
    def calLCCsize_CCfre(self):
        # compute the largest connected component via 2-projection
        G = nx.Graph()
        for node in self.Hnodes:
            for neiedge in self.HneiEdgeDic[node]:
                for neinode in self.HedgeDic[neiedge]:
                    G.add_edge(node, neinode)
        G.remove_edges_from(nx.selfloop_edges(G))
        CCs = list(nx.connected_components(G))
        maxCC = max(CCs, key=len, default=set())
        CCs_size = [len(i) for i in CCs]
        CCs_fre = collections.Counter(CCs_size)

        return len(maxCC), CCs_fre

    def calLSCC_size(self, sLst=[]):
        # compute s-line graph, find all connected components, then map back to the original hypergraph and measure component sizes
        # return a dict: key = s, value = size of the largest connected component in the original hypergraph
        smaxCC = dict()
        if sLst == []:
            sLst = [i+2 for i in range(self.maxHegdeSize-1)]
        for s in sLst:  # s is hyperedge size (order = s-1); minimum size is 2, hence +2 above
            # for each s: build the s-line graph, find connected components; map back to the original hypergraph to compute component sizes
            print(s)
            LG = nx.Graph()
            for hedge in self.HedgeDic:
                if len(self.HedgeDic[hedge]) >= s:
                    neiEdges = set()
                    for node in self.HedgeDic[hedge]:  # find all incident hyperedges
                        neiEdges.update(self.HneiEdgeDic[node])
                else:
                    continue
                for neihdg in neiEdges:
                    if len(set(self.HedgeDic[hedge]) & set(self.HedgeDic[neihdg])) >= s:
                        LG.add_edge(hedge, neihdg)
                LG.add_node(hedge)  # ensure isolated hyperedges are also added
            LSCCs_LineG = list(nx.connected_components(LG))  # list of connected components (each is a set of hyperedge indices)
            szLst = []  # for each component, number of nodes in the corresponding component in the original hypergraph
            for lc in LSCCs_LineG:  # map back to the original hypergraph and count nodes
                if len(lc) > 1:
                    print(lc)
                lcsz = set()
                for hedge in lc:
                    lcsz.update(set(self.HedgeDic[hedge]))
                szLst.append(len(lcsz))
            smaxCC[s] = max(szLst)
        return smaxCC, szLst

    def calEdgeSizeFrequency_total(self):
        # compute hyperedge size frequency in the full hypergraph (zeros are allowed)
        sizeFrq_tt = defaultdict(int)
        for heg in self.HedgeSizeDic:
            sizeFrq_tt[self.HedgeSizeDic[heg]] += 1
        # ineractions_tt = len(self.HedgeSizeDic)  # total number of hyperedges
        # print(ineractions_tt)
        # for f in sizeFrq_tt:
        #     sizeFrq_tt[f] = sizeFrq_tt[f]/ineractions_tt


class SimuNamingGame():
    def __init__(self, HGraph):
        self.HG = HGraph  # list of hyperedges (list of nodes), as loaded from json
        temp = []
        for i in self.HG:
            temp.extend(i)
        nodeSet = set(temp)
        self.Hnodes = list(nodeSet)
        self.Hnodes.sort()  # list of hypergraph node indices (int), sorted in ascending order
        self.N = len(self.Hnodes)
        self.HedgeDic = dict()  # key: hyperedge index (int); value: hyperedge as list of nodes
        self.HedgeSizeDic = dict()  # key: hyperedge index (int); value: hyperedge size (int)
        self.maxHegdeSize = 0  # maximum hyperedge size
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

    def initialization(self, committedNodes, initialN_Aratio):
        self.commNodesLst = committedNodes  # list of committed nodes
        self.opinionsDic = dict()
        for node in self.Hnodes:
            if node in self.commNodesLst:
                self.opinionsDic[node] = frozenset(['A'])
        N_Acomm = len(self.commNodesLst)
        N_A = int(self.N*initialN_Aratio)
        N_B = self.N - N_A - N_Acomm  # N_Acomm is not included in N_A
        opinions_to_assign = ['A']*N_A + ['B']*N_B
        random.shuffle(opinions_to_assign)
        uncommittedNodes = set(self.Hnodes) - set(self.commNodesLst)
        for node, opn in zip(uncommittedNodes, opinions_to_assign):
            self.opinionsDic[node] = set(opn)

    def agreeUpdate(self, hyperEdge, saidOP):
        for n in hyperEdge:
            try:
                self.opinionsDic[n].clear()
                self.opinionsDic[n].add(saidOP)
            except AttributeError:  # committed nodes
                pass

    def notagreeUpdate(self, listeners, saidOP):
        for ls in listeners:
            try:
                self.opinionsDic[ls].add(saidOP)
            except AttributeError:  # committed nodes
                pass

    def calDensities(self):
        single_opinion_counter = collections.Counter(
            [list(opinions)[0] for opinions in self.opinionsDic.values() if len(opinions) == 1]
        )
        n_A = single_opinion_counter["A"]/self.N
        n_B = single_opinion_counter["B"]/self.N
        n_AB = 1-n_A-n_B
        return n_A, n_B, n_AB

    def namingGameSimulation(self, beta, tmax, rule, check_every, sampleNum, sampleArg):
        t = 0
        HedgeLst = list(self.HedgeDic.keys())
        n_ALst = []
        n_BLst = []
        n_ABLst = []

        while t <= tmax:
            t += 1
            # if t % 10000 == 0:
            #     print(t)
            hyperedgeGPlatform = self.HedgeDic[random.choice(HedgeLst)]
            random.shuffle(hyperedgeGPlatform)
            speaker = hyperedgeGPlatform[0]
            listeners = hyperedgeGPlatform[1:]

            saidWord = random.choice(list(self.opinionsDic[speaker]))
            listenersWords = [self.opinionsDic[lser] for lser in listeners]

            if rule == 'union':
                ifAgreementWordLst = set.union(*[set(w) for w in listenersWords])
            elif rule == 'unanimity':    
                ifAgreementWordLst = set.intersection(*[set(w) for w in listenersWords])
            if (saidWord in ifAgreementWordLst) and (random.random() <= beta):
                self.agreeUpdate(hyperedgeGPlatform, saidWord)
            else:  # no agreement; the said word is learned by the listeners
                self.notagreeUpdate(listeners, saidWord)
            
            if t % check_every == 0:
                n_A, n_B, n_AB = self.calDensities()
                n_ALst.append(n_A)
                n_BLst.append(n_B)
                n_ABLst.append(n_AB)
                if n_A == 1 or n_B == 1:
                    # print(f'absorbing state reached')
                    return n_A, n_B, n_AB
        # print(f'run out of time, average over {sampleNum} samples in last {sampleArg} returened')
        sampleArg = int(sampleArg/check_every)
        sampledIndex = random.sample([i for i in range(sampleArg)], sampleNum)
        n_ALst = n_ALst[-sampleArg:]
        n_BLst = n_BLst[-sampleArg:]
        n_ABLst = n_ABLst[-sampleArg:]
        n_A = sum([n_ALst[i] for i in sampledIndex])/sampleNum
        n_B = sum([n_BLst[i] for i in sampledIndex])/sampleNum
        n_AB = sum([n_ABLst[i] for i in sampledIndex])/sampleNum
        return n_A, n_B, n_AB

class SimuNC_SIS_QSapproach():
    def __init__(self, HGraph):
        self.HG = HGraph  # list of hyperedges (list of nodes), as loaded from json
        temp = []
        for i in self.HG:
            temp.extend(i)
        nodeSet = set(temp)
        self.Hnodes = list(nodeSet)
        self.Hnodes.sort()  # list of hypergraph node indices (int), sorted in ascending order
        self.N = len(self.Hnodes)
        self.HedgeDic = dict()  # key: hyperedge index (int); value: hyperedge as list of nodes
        self.HedgeSizeDic = dict()  # key: hyperedge index (int); value: hyperedge size (int)
        self.maxHegdeSize = 0  # maximum hyperedge size
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
        self.iAgentSet = set()
        self.sAgentSet = set()
        self.rAgentSet = set()
        self.iNodesNum_inHedge = {}  # key: hyperedge index; value: number of infected nodes in the hyperedge

    def infectAgent(self,agent):
        self.iAgentSet.add(agent)
        self.sAgentSet.remove(agent)
        self.nodeStateDic[agent] = 'I'
        return 1

    def recoverAgent_S(self, agent):
        self.sAgentSet.add(agent)
        self.iAgentSet.remove(agent)
        self.nodeStateDic[agent] = 'S'
        return 0

    def initialization_random(self, randomInfect=0):
        self.nodeStateDic = dict()  # record node states; key: node, value: state
        self.iNodesNum_inHedge = defaultdict(int)
        for n in self.Hnodes:
            self.sAgentSet.add(n)
            self.nodeStateDic[n] = 'S'
        
        self.iAgentSet = set()
            
        if randomInfect > 0:
            self.infect_this_setup = random.sample(self.Hnodes, randomInfect)
        else:
            print("error, initialization unsetted") 
        for to_infect in self.infect_this_setup:  # to_infect is node index
            self.infectAgent(to_infect)
            for neiE in self.HneiEdgeDic[to_infect]:
                self.iNodesNum_inHedge[neiE] += 1

    def initialization_defined(self, definiteInfect=[]):
        self.nodeStateDic = dict()  # record node states; key: node, value: state
        self.iNodesNum_inHedge = defaultdict(int)
        for n in self.Hnodes:
            self.sAgentSet.add(n)
            self.nodeStateDic[n] = 'S'
        
        self.iAgentSet = set()
            
        if definiteInfect != []:
            self.infect_this_setup = definiteInfect
        else:
            print("error, initialization unsetted") 
        for to_infect in self.infect_this_setup:  # to_infect is node index
            self.infectAgent(to_infect)
            for neiE in self.HneiEdgeDic[to_infect]:
                self.iNodesNum_inHedge[neiE] += 1

    def contagionSimu_QSapproach(self, nu, l, mu, tr, ta):
        pass

class Simu_NCsize_SIR():
    def __init__(self, HGraph):
        self.HG = HGraph  # list of hyperedges (list of nodes)
        temp = []
        for i in self.HG:
            temp.extend(i)
        nodeSet = set(temp)
        self.Hnodes = list(nodeSet)
        self.Hnodes.sort()  # list of hypergraph node indices (int), sorted in ascending order
        self.HedgeDic = dict()  # key: hyperedge index (int); value: hyperedge as list of nodes
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

        self.iAgentSet = set()
        self.sAgentSet = set()
        self.rAgentSet = set()

    def infectAgent(self,agent):
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

    def initialization(self, initial_infected, immunilized=[]):  # SIR: initial infected nodes must be specified
        self.nodeStateDic = dict()
        self.sAgentSet = set()
        self.iAgentSet = set()
        self.rAgentSet = set()
        self.rAgentSet_imm = set()
        for n in self.Hnodes:
            self.sAgentSet.add(n)
            self.nodeStateDic[n] = 'S'
            
        self.infect_this_setup = initial_infected
        for to_infect in self.infect_this_setup:  # to_infect is node index
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
        for to_infect in self.infect_this_setup:  # to_infect is node index
            self.infectAgent(to_infect)
    
    def contagionSimulation(self, l, mu, nu, tmax=float("Inf"), timeSeries=False):
        infectionProbDic = dict()
        for sizeE in range(1, self.maxHegdeSize+1):
            # each hyperedge size has its own contagion probability profile
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
                    if self.nodeStateDic[i] == 'I':
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
                p = 1
                for edgeID in self.HneiEdgeDic[node]:  # indices of hyperedges incident to node
                    p = p * infectionProbDic[self.HedgeSizeDic[edgeID]][iNodesNum_inHedge[edgeID]]
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


def chooseTopNodes(hrt, netName, chooseMtd, chooseNum):
    # choose nodes with the largest values of a given measure
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
        with open(f"../MeasureValuesRanking/NodesMeasures/{hrt}_{chooseMtd}_{netName}_rvs.json", "r") as f:
            # file is a dict: key = measure value (float), value = list of nodes (int) with that value
            msNodesDic = json.load(f)

            msList = [float(i) for i in msNodesDic.keys()]
            msList.sort(reverse=True)  # sort in descending order

            aggchoosenNodes = 0
            for ms in msList:
                currN = len(msNodesDic[str(ms)])
                aggchoosenNodes += currN
                if aggchoosenNodes < chooseNum:
                    # if adding all nodes with the current measure value is still not enough, add them all
                    ext = msNodesDic[str(ms)][:]
                    random.shuffle(ext)
                    immediatedNodes.extend(ext)
                else:
                    # randomly sample the remaining number of nodes needed from the current measure group
                    immediatedNodes.extend(
                        random.sample(msNodesDic[str(ms)], chooseNum-aggchoosenNodes+currN)
                    )
                    # return a list of node indices
                    return immediatedNodes
