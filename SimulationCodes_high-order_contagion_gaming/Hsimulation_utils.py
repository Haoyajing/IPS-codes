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
        self.Hnodes.sort()  # list of hypergraph node indices (int), sorted in ascending order
        self.HedgeDic = dict()  # key: hyperedge index (int); value: hyperedge (list of nodes)
        self.HedgeSizeDic = dict()  # key: hyperedge index (int); value: size of the hyperedge
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

    def initialization(self, initial_infected, immunilized=[]):  # for SIR, initial infected nodes are explicitly specified
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
                for edgeID in self.HneiEdgeDic[node]:  # list of hyperedges incident to node
                    p=p*infectionProbLst[iNodesNum_inHedge[edgeID]]
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

    
    def contagionSimulation_details(self, l, mu, nu, tmax=float("Inf")):
        pass


class SimuThresholdContagion_SIR():
    def __init__(self, HGraph):
        self.HG = HGraph  # list of hyperedges, each hyperedge is a list of nodes
        temp = []
        for i in self.HG:
            temp.extend(i)
        nodeSet = set(temp)
        self.Hnodes = list(nodeSet)
        self.Hnodes.sort()  # list of hypergraph node indices (int), sorted in ascending order
        self.HedgeDic = dict()  # key: hyperedge index (int); value: hyperedge (list of nodes)
        self.HedgeSizeDic = dict()  # key: hyperedge index (int); value: size of the hyperedge
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

    def initialization(self, initial_infected):  # for SIR, initial infected nodes are explicitly specified
        self.nodeStateDic = dict()
        self.sAgentSet = set()
        self.iAgentSet = set()
        self.rAgentSet = set()
        for n in self.Hnodes:
            self.sAgentSet.add(n)
            self.nodeStateDic[n] = 'S'
            
        self.infect_this_setup = initial_infected
        for to_infect in self.infect_this_setup:  # to_infect is the node index
            self.infectAgent(to_infect)
            self.nodeStateDic[to_infect] = 'I'

    def initialization_random(self, randomInfect=0):
        self.nodeStateDic = dict()  # record node states; key: node, value: state
        self.sAgentSet = set()
        self.iAgentSet = set()
        self.rAgentSet = set()
        self.iNodesNum_inHedge = defaultdict(int)  # key: hyperedge index; value: number of infected nodes in that hyperedge
        for n in self.Hnodes:
            self.sAgentSet.add(n)
            self.nodeStateDic[n] = 'S'
        
        if randomInfect > 0:
            self.infect_this_setup = random.sample(self.Hnodes, randomInfect)
        else:
            print("error, initialization unsetted") 
        for to_infect in self.infect_this_setup:  # to_infect is the node index
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
                if len(SnodesLst)>0 and (iNodesNum_inHedge[edgeID]>=math.ceil(theta*len(self.HedgeDic[edgeID]))) and (random.random() <= lbd):
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
        self.HedgeDic = dict()  # key: hyperedge index (int); value: hyperedge (list of nodes)
        self.HedgeSizeDic = dict()  # key: hyperedge index (int); value: size of the hyperedge
        self.maxHegdeSize = 0  # maximum hyperedge size
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
            self.opinionsDic[node]=set(opn)

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
        single_opinion_counter = collections.Counter([list(opinions)[0] for opinions in self.opinionsDic.values() if len(opinions)==1])
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
            #if t % 10000 == 0:
            #    print(t)
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
            else:  # no agreement, but the spoken word is learned by the listeners
                self.notagreeUpdate(listeners, saidWord)
            
            if t%check_every==0:
                n_A, n_B, n_AB = self.calDensities()
                n_ALst.append(n_A)
                n_BLst.append(n_B)
                n_ABLst.append(n_AB)
                if n_A==1 or n_B==1:
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
        n_AB =  sum([n_ABLst[i] for i in sampledIndex])/sampleNum
        return n_A, n_B, n_AB

class Simu_NCsize_SIR():
    def __init__(self, HGraph):
        self.HG = HGraph  # list of hyperedges, each hyperedge is a list of nodes
        temp = []
        for i in self.HG:
            temp.extend(i)
        nodeSet = set(temp)
        self.Hnodes = list(nodeSet)
        self.Hnodes.sort()  # list of hypergraph node indices (int), sorted in ascending order
        self.HedgeDic = dict()  # key: hyperedge index (int); value: hyperedge (list of nodes)
        self.HedgeSizeDic = dict()  # key: hyperedge index (int); value: size of the hyperedge
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

    def initialization(self, initial_infected, immunilized=[]):  # for SIR, initial infected nodes are explicitly specified
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
                for edgeID in self.HneiEdgeDic[node]:  # list of hyperedges incident to node
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


def chooseTopNodes(hrt, netName, chooseMtd, chooseNum):  # choose nodes with the highest measure values
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
            # file is a dict: key = measure value (float); value = list of nodes (int) with that value
            msNodesDic = json.load(f)

            msList = [float(i) for i in msNodesDic.keys()]
            msList.sort(reverse=True)  # sort in descending order

            aggchoosenNodes = 0
            for ms in msList:
                currN = len(msNodesDic[str(ms)])
                aggchoosenNodes += currN
                if aggchoosenNodes < chooseNum:  # adding all nodes with current measure value is still not enough; include them all
                    ext = msNodesDic[str(ms)][:]
                    random.shuffle(ext)
                    immediatedNodes.extend(ext)
                else:  # randomly sample the remaining number of nodes needed
                    immediatedNodes.extend((random.sample(msNodesDic[str(ms)], chooseNum-aggchoosenNodes+currN)))
                    return immediatedNodes  # list of node indices
