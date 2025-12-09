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
from scipy.stats import kendalltau

class NetAtrributions():
    def __init__(self, group_list):
        temp = []
        for i in group_list:
            temp.extend(i)
        nodeSet = set(temp)
        self.nodes = list(nodeSet)
        self.nodes.sort()  # ascending order by default
        self.groups = group_list

        self.edgeDic = dict()  # key: hyperedge index (int); value: list of nodes in the hyperedge
        self.edgeSizeDic = dict()  # key: hyperedge index (int); value: size of the hyperedge (int)
        self.neiNodeEdgeDic = defaultdict(list)  # key: node index (int); value: list of tuples (neighbor node, neighboring hyperedge)
        self.neiEdgeDic = defaultdict(list)  # key: node index (int); value: list of incident hyperedges
        self.neiNodeDic = defaultdict(set)  # key: node index (int); value: set of neighboring nodes
        self.maxHegdeSize = 0  # maximum hyperedge size
        self.N = len(self.nodes)
        self.M = len(self.groups)
        self.nodeDegreeDic = defaultdict(int)
        i=1
        for hyperedge in self.groups:
            self.edgeDic[i] = hyperedge
            self.edgeSizeDic[i] = len(hyperedge)
            if len(hyperedge) > self.maxHegdeSize:
                self.maxHegdeSize = len(hyperedge)
            for node in hyperedge:
                self.nodeDegreeDic[node] += 1
                self.neiEdgeDic[node].append(i)
                for neinode in hyperedge:
                    if neinode != node:  # avoid adding node itself to its neighbor list
                        self.neiNodeEdgeDic[node].append((neinode,i))
                        self.neiNodeDic[node].add(neinode)
            i += 1

    def caldistance(self, node1, node2):
        pass


def chooseTopNodes_h_realNet(netName, chooseMtd, chooseNum):
    # choose nodes with the largest values of a given measure
    immediatedNodes = []
    with open(f"../../MeasureValuesRanking/NodesMeasures/h_{chooseMtd}_realNet_{netName}_rvs.json","r") as f:
        # file is a dict: key = measure value (float), value = list of nodes (int) with the same value
        msNodesDic_oral = json.load(f)

        msNodesDic = dict()
        if os.path.exists(f'iniNodes_25_rvs_{chooseMtd}.json'):
            with open(f'iniNodes_25_rvs_{chooseMtd}.json', 'r') as f:
                msNodesDic = json.load(f)
        else:
            with open('iniNodes_25.json', 'r') as f:
                ininodes = json.load(f)
            for ms in msNodesDic_oral:
                msNodesDic[ms] = []
                for node in msNodesDic_oral[ms]:
                    if node in ininodes:
                        msNodesDic[ms].append(node)
            with open(f'iniNodes_25_rvs_{chooseMtd}.json', 'w') as f:
                json.dump(msNodesDic,f)
        msList = [float(i) for i in msNodesDic.keys()]
        msList.sort(reverse=True)  # sort in descending order

        aggchoosenNodes = 0
        for ms in msList:
            currN = len(msNodesDic[str(ms)])
            aggchoosenNodes += currN
            if aggchoosenNodes < chooseNum:
                # if adding all nodes with current measure value is still not enough, add them all
                ext = msNodesDic[str(ms)][:]
                random.shuffle(ext)
                immediatedNodes.extend(ext)
            else:
                # randomly sample the remaining number of nodes needed from the current measure group
                immediatedNodes.extend((random.sample(msNodesDic[str(ms)], chooseNum-aggchoosenNodes+currN)))
                return immediatedNodes  # list of node IDs


def chooseTopNodes_groundTruth(fileName, chooseNum):
    # choose nodes with the largest ground-truth spreading size (simulation-based influence)
    with open(f'{fileName}.json', 'r') as f:
        msNodesDic = json.load(f)
    immediatedNodes = []
    msList = [float(i) for i in msNodesDic.keys()]
    msList.sort(reverse=True)  # sort in descending order

    aggchoosenNodes = 0
    for ms in msList:
        currN = len(msNodesDic[str(ms)])
        aggchoosenNodes += currN
        if aggchoosenNodes < chooseNum:
            # if adding all nodes with current measure value is still not enough, add them all
            ext = msNodesDic[str(ms)][:]
            random.shuffle(ext)
            immediatedNodes.extend(ext)
        else:
            # randomly sample the remaining number of nodes needed from the current measure group
            immediatedNodes.extend((random.sample(msNodesDic[str(ms)], chooseNum-aggchoosenNodes+currN)))
            return immediatedNodes  # list of node IDs

def calMeasuresNodes(nodeMeasureDic, fileName):
    msDic = defaultdict(list)
    for node in nodeMeasureDic:
        msDic[nodeMeasureDic[node]].append(int(node))
    keysLstStr = msDic.keys()
    # check if there are duplicate float values that map to different string keys
    keysLst = [float(i) for i in keysLstStr]
    keysSet = set(keysLst)
    duplicates = []
    if len(keysLst) == len(keysSet):
        print(f"no duplicates, {fileName}_rvs generated")
    else:
        for element in keysLst:
            if keysLst.count(element) > 1 and element not in duplicates:
                duplicates.append(element)
        print(f"duplicates={duplicates}, {fileName}_rvs generated, further processing needed!")
    
    with open(f'{fileName}_rvs.json',"w") as f:
        json.dump(msDic, f)
    return msDic


def caltau(dict1,dict2, type):
    lst1 = []
    lst2 = []
    for item in dict1:
        lst1.append(dict1[item])
        lst2.append(dict2[item])
    tau,p = kendalltau(lst1,lst2,variant=f'{type}')
    return tau
