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
        i = 1
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
                        self.neiNodeEdgeDic[node].append((neinode, i))
                        self.neiNodeDic[node].add(neinode)
            i += 1

    def caldistance(self, node1, node2):
        pass


def chooseTopNodes_h_realNet(netName, chooseMtd, chooseNum):
    # select nodes with the largest values of a given node measure
    immediatedNodes = []
    with open(f"../MeasureValuesRanking/NodesMeasures/h_{chooseMtd}_realNet_{netName}_rvs.json", "r") as f:
        # dictionary: key = measure value (float), value = list of nodes (int) with that value
        msNodesDic = json.load(f)

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
                immediatedNodes.extend(
                    random.sample(msNodesDic[str(ms)], chooseNum - aggchoosenNodes + currN)
                )
                return immediatedNodes


def chooseTopNodes_groundTruth(model, filename, chooseNum):
    # select nodes with the largest simulated spreading size (ground-truth influence)
    immediatedNodes = []
    with open(f"{model}/{filename}_rvs.json", "r") as f:
        # dictionary: key = measure value (float), value = list of nodes (int) with that value
        msNodesDic = json.load(f)

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
                immediatedNodes.extend(
                    random.sample(msNodesDic[str(ms)], chooseNum - aggchoosenNodes + currN)
                )
                return immediatedNodes


def caltau(dict1, dict2, type):
    lst1 = []
    lst2 = []
    for item in dict1:
        lst1.append(dict1[item])
        lst2.append(dict2[item])
    tau, p = kendalltau(lst1, lst2, variant=f'{type}')
    return tau
