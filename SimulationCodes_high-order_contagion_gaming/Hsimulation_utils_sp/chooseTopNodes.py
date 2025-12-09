import networkx as nx
import time
from collections import defaultdict
from collections import Counter
import os
import json
import numpy as np
from scipy import sparse 
import copy
import random
import math

def chooseTopNodes(hrt, netName, chooseMtd, chooseNum, mu, nu, lbd):  # select nodes with the largest measure values
    if chooseMtd == 'random':
        with open(f'../../Networks/networks/{netName}.json', 'r') as fl:
            HyperGraph = json.load(fl)
        Hnodes = []
        for edge in HyperGraph:
            Hnodes.extend(edge)
        HnodesLst = list(set(Hnodes))
        random.shuffle(HnodesLst)
        immediatedNodes = random.sample(HnodesLst,chooseNum)
        return immediatedNodes
    else: 
        immediatedNodes = []
        with open(f"../../MeasureValuesRanking/NodesMeasures/{hrt}_{chooseMtd}_{netName}_rvs.json","r") as f:
            # file is a dict: key = measure value (float); value = list of nodes (int) that share this value
            msNodesDic = json.load(f)

            msList = [float(i) for i in msNodesDic.keys()]
            msList.sort(reverse=True)  # sort in descending order

            aggchoosenNodes = 0
            for ms in msList:
                currN = len(msNodesDic[str(ms)])
                aggchoosenNodes += currN
                if aggchoosenNodes < chooseNum:  # still not enough after adding all nodes with the current measure; include them all
                    ext = msNodesDic[str(ms)][:]
                    random.shuffle(ext)
                    immediatedNodes.extend(ext)
                else:  # randomly add the remaining needed nodes (chooseNum - aggchoosenNodes + currN) into the top-k set
                    immediatedNodes.extend((random.sample(msNodesDic[str(ms)], chooseNum-aggchoosenNodes+currN)))
                    return immediatedNodes  # list of node indices

class CalNodesMeasures():
    def __init__(self, group_list, net):
        self.net = net  # network name
        temp = []
        for i in group_list:
            temp.extend(i)
        nodeSet = set(temp)
        self.nodes = list(nodeSet)
        self.nodes.sort()  # sorted in ascending order
        self.groups = group_list

        self.edgeDic = dict()  # key: hyperedge index (int); value: hyperedge as a list of nodes
        self.edgeSizeDic = dict()  # key: hyperedge index (int); value: hyperedge size (int)
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
                    if neinode != node:  # avoid adding node itself into its neighbor list
                        self.neiNodeEdgeDic[node].append((neinode,i))
                        self.neiNodeDic[node].add(neinode)
            i += 1

    def h_random(self, IFcalTime=True):
        time_sum = None
        if IFcalTime:
            time_start = time.time()
        Hnodes = []
        for edge in self.groups:
            Hnodes.extend(edge)
        HnodesLst = list(set(Hnodes))
        delnode = random.choice(HnodesLst)
        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start
        return delnode, time_sum

    def h_degree(self, IFcalTime=True):
        time_sum = None
        if IFcalTime:
            time_start = time.time()
        
        nodeDegreeDic = defaultdict(float)

        for gp in self.groups:
            for node in gp:
                nodeDegreeDic[node] += 1

        msDic = defaultdict(list)
        for node in nodeDegreeDic:
            msDic[nodeDegreeDic[node]].append(int(node))
        # msList = [float(i) for i in msDic.keys()]
        # msList.sort(reverse=True)
        delnode = random.choice(msDic[max(msDic.keys())])
        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start

        return delnode, time_sum

    def h_neiNodesNum(self, IFcalTime=True):
        time_sum = None
        if IFcalTime:
            time_start = time.time()
        
        nodeDegreeDic = dict()
        neiNodeDic = defaultdict(set)
        for hyperedge in self.groups:
            for node in hyperedge:
                for neinode in hyperedge:
                    if neinode != node:  # avoid adding node itself into its neighbor list
                        neiNodeDic[node].add(neinode)
        for node in neiNodeDic:
            nodeDegreeDic[node] = float(len(neiNodeDic[node]))
        Hnodes = []
        for edge in self.groups:
            Hnodes.extend(edge)
        HnodesSet = (set(Hnodes))
        for node in HnodesSet-set(neiNodeDic.keys()):
            nodeDegreeDic[node] = float(0)
        msDic = defaultdict(list)
        for node in nodeDegreeDic:
            msDic[nodeDegreeDic[node]].append(int(node))
        # msList = [float(i) for i in msDic.keys()]
        # msList.sort(reverse=True)
        delnode = random.choice(msDic[max(msDic.keys())])
        
        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start
        return delnode, time_sum


    def h_clsCloseness(self, IFcalTime=True):
        time_sum = None
        if IFcalTime:
            time_start = time.time()
        temp = []
        for i in self.groups:
            temp.extend(i)
        nodeSet = set(temp)
        nodes = list(nodeSet)
        nodes.sort()  # sorted in ascending order
        G_2proj = nx.Graph()  # compute measures on the 2-projection
        neiEdgeDic = defaultdict(list)
        edgeDic = dict()
        i=1
        for hyperedge in self.groups:
            edgeDic[i] = hyperedge
            for node in hyperedge:
                neiEdgeDic[node].append(i)
            i += 1
        for node in nodes:
            for neiedge in neiEdgeDic[node]:
                for neinode in edgeDic[neiedge]:
                    G_2proj.add_edge(node, neinode)
        G_2proj.remove_edges_from(nx.selfloop_edges(G_2proj))
        nodeclsClosenessDic = nx.closeness_centrality(G_2proj)

        msDic = defaultdict(list)
        for node in nodeclsClosenessDic:
            msDic[nodeclsClosenessDic[node]].append(int(node))
        # msList = [float(i) for i in msDic.keys()]
        # msList.sort(reverse=True)
        delnode = random.choice(msDic[max(msDic.keys())])

        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start
        return delnode, time_sum
    
    def h_clsBetweenness(self, IFcalTime=True):
        time_sum = None
        if IFcalTime:
            time_start = time.time()

        temp = []
        for i in self.groups:
            temp.extend(i)
        nodeSet = set(temp)
        nodes = list(nodeSet)
        nodes.sort()  # sorted in ascending order
        G_2proj = nx.Graph()  # compute measures on the 2-projection
        neiEdgeDic = defaultdict(list)
        edgeDic = dict()
        i=1
        for hyperedge in self.groups:
            edgeDic[i] = hyperedge
            for node in hyperedge:
                neiEdgeDic[node].append(i)
            i += 1
        for node in nodes:
            for neiedge in neiEdgeDic[node]:
                for neinode in edgeDic[neiedge]:
                    G_2proj.add_edge(node, neinode)
        G_2proj.remove_edges_from(nx.selfloop_edges(G_2proj))
        nodeclsBetweennessDic = nx.betweenness_centrality(G_2proj)

        msDic = defaultdict(list)
        for node in nodeclsBetweennessDic:
            msDic[nodeclsBetweennessDic[node]].append(int(node))
        # msList = [float(i) for i in msDic.keys()]
        # msList.sort(reverse=True)
        delnode = random.choice(msDic[max(msDic.keys())])

        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start

        return delnode, time_sum

    def h_clsEigenvector(self, IFcalTime=True):
        time_sum = None
        if IFcalTime:
            time_start = time.time()
        temp = []
        for i in self.groups:
            temp.extend(i)
        nodeSet = set(temp)
        nodes = list(nodeSet)
        nodes.sort()  # sorted in ascending order
        G_2proj = nx.Graph()  # compute measures on the 2-projection
        neiEdgeDic = defaultdict(list)
        edgeDic = dict()
        i=1
        for hyperedge in self.groups:
            edgeDic[i] = hyperedge
            for node in hyperedge:
                neiEdgeDic[node].append(i)
            i += 1
        for node in nodes:
            for neiedge in neiEdgeDic[node]:
                for neinode in edgeDic[neiedge]:
                    G_2proj.add_edge(node, neinode)
        G_2proj.remove_edges_from(nx.selfloop_edges(G_2proj))
        nodemsDic = nx.eigenvector_centrality(G_2proj)
        
        msDic = defaultdict(list)
        for node in nodemsDic:
            msDic[nodemsDic[node]].append(int(node))
        delnode = random.choice(msDic[max(msDic.keys())])

        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start

        return delnode, time_sum
  
    def h_nodeEdgeEigenvector(self, funcGrp='linear', IFcalTime=True):
        time_sum = None
        if IFcalTime:
            time_start = time.time()

        nodeEdgeEigenvectorDic = defaultdict(float)
        if funcGrp == 'linear':
            f = lambda x: x
            g = lambda x: x
            phi = lambda x: x
            psi = lambda x: x
        if funcGrp == 'log_exp':
            f = lambda x: x
            g = lambda x: np.power(x,1/2)
            phi = lambda x: np.log(x)
            psi = lambda x: np.exp(x)
        if funcGrp == 'max':
            f = lambda x: x
            g = lambda x: x
            phi = lambda x: np.power(x,10)
            psi = lambda x: np.power(x,1/10)
        max_iter=1000
        tol=1e-8
        temp = []
        for i in self.groups:
            temp.extend(i)
        nodeSet = set(temp)
        N = len(nodeSet)
        M = len(self.groups)
        backwardDic,neiEdgeDic_new = self.reNumber_ne()

        B = self.geneIncidenceMatrix(neiEdgeDic_new)  # generate incidence matrix B (numpy array)

        x = np.ones(self.N) / self.N
        y = np.ones(self.M) / self.M
        check = np.inf
        for iter in range(max_iter):
            u = np.sqrt(np.multiply(x, g(B @ f(y))))
            v = np.sqrt(np.multiply(y, psi(B.T @ phi(x))))
            # multiply by the sign to try and enforce positivity
            new_x = np.multiply(u, 1/np.linalg.norm(u, 1))
            new_y = np.multiply(v, 1/np.linalg.norm(v, 1))

            check = np.linalg.norm(new_x-x,1) + np.linalg.norm(new_y-y,1)
            if check < tol:
                break
            x = copy.deepcopy(new_x)
            y = copy.deepcopy(new_y)
        else:
            print(f'{self.net} nodeedgeEign not converge')

        if backwardDic:
            # map back to original node labels; current indices are the renumbered consecutive indices
            nodeEdgeEigenvectorDic = dict()
            for id, vl in enumerate(x):
                nodeEdgeEigenvectorDic[backwardDic[id]] = vl

        msDic = defaultdict(list)
        for node in nodeEdgeEigenvectorDic:
            msDic[nodeEdgeEigenvectorDic[node]].append(int(node))
        delnode = random.choice(msDic[max(msDic.keys())])
        
        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start
        return delnode, time_sum

    def h_clsKcore(self, IFcalTime=True):
        time_sum = None
        if IFcalTime:
            time_start = time.time()

        temp = []
        for i in self.groups:
            temp.extend(i)
        nodeSet = set(temp)
        nodes = list(nodeSet)
        nodes.sort()  # sorted in ascending order
        G_2proj = nx.Graph()  # compute measures on the 2-projection
        neiEdgeDic = defaultdict(list)
        edgeDic = dict()
        i=1
        for hyperedge in self.groups:
            edgeDic[i] = hyperedge
            for node in hyperedge:
                neiEdgeDic[node].append(i)
            i += 1
        for node in nodes:
            for neiedge in neiEdgeDic[node]:
                for neinode in edgeDic[neiedge]:
                    G_2proj.add_edge(node, neinode)
        G_2proj.remove_edges_from(nx.selfloop_edges(G_2proj))        
        nodeclsKcoreDic_int = nx.core_number(G_2proj)
        nodeclsKcoreDic = dict()
        for item in nodeclsKcoreDic_int:
            nodeclsKcoreDic[item] = float(nodeclsKcoreDic_int[item])

        msDic = defaultdict(list)
        for node in nodeclsKcoreDic:
            msDic[nodeclsKcoreDic[node]].append(int(node))
        delnode = random.choice(msDic[max(msDic.keys())])

        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start

        return delnode, time_sum

    def h_KMcore_g1(self, IFcalTime=True):
        time_sum = None
        if IFcalTime:
            time_start = time.time()
        temp = []
        for i in self.groups:
            temp.extend(i)
        nodeSet = set(temp)
        nodes = list(nodeSet)
        nodes.sort()  # sorted in ascending order

        edgeDic = dict()  # key: hyperedge index (int); value: hyperedge as a list of nodes
        edgeSizeDic = dict()  # key: hyperedge index (int); value: hyperedge size (int)
        maxHegdeSize = 0  # maximum hyperedge size
        nodeDegreeDic = defaultdict(int)
        i=1
        for hyperedge in self.groups:
            edgeDic[i] = hyperedge
            edgeSizeDic[i] = len(hyperedge)
            if len(hyperedge) > maxHegdeSize:
                maxHegdeSize = len(hyperedge)
            for node in hyperedge:
                nodeDegreeDic[node] += 1
            i += 1

        maxK = max([nodeDegreeDic[_] for _ in nodeDegreeDic])
        nodeKMcore_g1Dic = defaultdict(float)

        nodePosation = defaultdict(dict)
        mshells = defaultdict(dict)
        for m in range(2, maxHegdeSize+1):
            currentHGraph = set()
            for edge in edgeSizeDic:  # first select all hyperedges with size ≥ m to form sub-hypergraph currentHGraph
                if edgeSizeDic[edge] >= m:
                    currentHGraph.add(frozenset(edgeDic[edge]))
            
            for k in range(1, maxK+1):
                degDic = defaultdict(int)
                for hedge in currentHGraph:  # compute node degrees in the sub-hypergraph and find nodes with degree < k
                    for node in hedge:
                        degDic[node] += 1
                nodes_prevShell = list(degDic.keys())  # nodes in the previous shell

                nodeDelSet = ([_ for _ in degDic if degDic[_] < k])
                tempHG_nodeDeled = set([frozenset(set(e) - set(nodeDelSet)) for e in currentHGraph])  # iteratively remove nodes from the hypergraph
                edgeDelSet = set([e for e in tempHG_nodeDeled if len(e) < m])
                tempHG_edgeDeled = set([e for e in tempHG_nodeDeled if len(e) >= m])
                currentHGraph = copy.deepcopy(tempHG_edgeDeled)

                if len(nodeDelSet) != 0 or len(edgeDelSet) != 0:
                    while len(nodeDelSet) != 0 or len(edgeDelSet) != 0:
                        degDic = defaultdict(int)
                        for hedge in currentHGraph:  # recompute node degrees and remove nodes with degree < k
                            for node in hedge:
                                degDic[node] += 1
                        nodeDelSet = ([_ for _ in degDic if degDic[_] < k])
                        tempHG_nodeDeled = set([frozenset(set(e) - set(nodeDelSet)) for e in currentHGraph])
                        edgeDelSet = set([e for e in tempHG_nodeDeled if len(e) < m])
                        tempHG_edgeDeled = set([e for e in tempHG_nodeDeled if len(e) >= m])
                        currentHGraph = copy.deepcopy(tempHG_edgeDeled)
                    tpNodes = []
                    for _ in currentHGraph:
                        tpNodes.extend(list(_))
                    shell_k_nodes = set(nodes_prevShell) - set(tpNodes)
                    mshells[m][k] = []
                    for node in shell_k_nodes:
                        nodePosation[node][m] = k
                        mshells[m][k].append(node)
                elif len(currentHGraph) != 0:  # hypergraph not fully dismantled yet; increase k and continue
                    continue
                else:  # fully dismantled
                    break
        KMmax = {m:max(list(mshells[m].keys())) for m in mshells}
        sizeFrequence = defaultdict(int)  # key: hyperedge size; value: count
        for e in edgeSizeDic:
            sizeFrequence[edgeSizeDic[e]] += 1
        for node in nodes:
            nodeKMcore_g1Dic[node] = sum([nodePosation[node][m]/KMmax[m] for m in nodePosation[node]])

        msDic = defaultdict(list)
        for node in nodeKMcore_g1Dic:
            msDic[nodeKMcore_g1Dic[node]].append(int(node))
        delnode = random.choice(msDic[max(msDic.keys())])

        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start
            
        return delnode, time_sum

    def h_KMcore_gf(self, IFcalTime=True):
        time_sum = None
        if IFcalTime:
            time_start = time.time()
        temp = []
        for i in self.groups:
            temp.extend(i)
        nodeSet = set(temp)
        nodes = list(nodeSet)
        nodes.sort()  # sorted in ascending order

        edgeDic = dict()  # key: hyperedge index (int); value: hyperedge as a list of nodes
        edgeSizeDic = dict()  # key: hyperedge index (int); value: hyperedge size (int)
        maxHegdeSize = 0  # maximum hyperedge size
        nodeDegreeDic = defaultdict(int)
        i=1
        for hyperedge in self.groups:
            edgeDic[i] = hyperedge
            edgeSizeDic[i] = len(hyperedge)
            if len(hyperedge) > maxHegdeSize:
                maxHegdeSize = len(hyperedge)
            for node in hyperedge:
                nodeDegreeDic[node] += 1
            i += 1
        maxK = max([nodeDegreeDic[_] for _ in nodeDegreeDic])
        nodeKMcore_gfDic = defaultdict(float)

        nodePosation = defaultdict(dict)
        mshells = defaultdict(dict)
        for m in range(2, maxHegdeSize+1):
            currentHGraph = set()
            for edge in edgeSizeDic:  # first select all hyperedges with size ≥ m to form sub-hypergraph currentHGraph
                if edgeSizeDic[edge] >= m:
                    currentHGraph.add(frozenset(edgeDic[edge]))
            
            for k in range(1, maxK+1):
                degDic = defaultdict(int)
                for hedge in currentHGraph:  # compute node degrees in the sub-hypergraph and find nodes with degree < k
                    for node in hedge:
                        degDic[node] += 1
                nodes_prevShell = list(degDic.keys())  # nodes in the previous shell

                nodeDelSet = ([_ for _ in degDic if degDic[_] < k])
                tempHG_nodeDeled = set([frozenset(set(e) - set(nodeDelSet)) for e in currentHGraph])  # iteratively remove nodes from the hypergraph
                edgeDelSet = set([e for e in tempHG_nodeDeled if len(e) < m])
                tempHG_edgeDeled = set([e for e in tempHG_nodeDeled if len(e) >= m])
                currentHGraph = copy.deepcopy(tempHG_edgeDeled)

                if len(nodeDelSet) != 0 or len(edgeDelSet) != 0:
                    while len(nodeDelSet) != 0 or len(edgeDelSet) != 0:
                        degDic = defaultdict(int)
                        for hedge in currentHGraph:  # recompute node degrees and remove nodes with degree < k
                            for node in hedge:
                                degDic[node] += 1
                        nodeDelSet = ([_ for _ in degDic if degDic[_] < k])
                        tempHG_nodeDeled = set([frozenset(set(e) - set(nodeDelSet)) for e in currentHGraph])
                        edgeDelSet = set([e for e in tempHG_nodeDeled if len(e) < m])
                        tempHG_edgeDeled = set([e for e in tempHG_nodeDeled if len(e) >= m])
                        currentHGraph = copy.deepcopy(tempHG_edgeDeled)
                    tpNodes = []
                    for _ in currentHGraph:
                        tpNodes.extend(list(_))
                    shell_k_nodes = set(nodes_prevShell) - set(tpNodes)
                    mshells[m][k] = []
                    for node in shell_k_nodes:
                        nodePosation[node][m] = k
                        mshells[m][k].append(node)
                elif len(currentHGraph) != 0:  # hypergraph not fully dismantled yet; increase k and continue
                    continue
                else:  # fully dismantled
                    break
        KMmax = {m:max(list(mshells[m].keys())) for m in mshells}
        sizeFrequence = defaultdict(int)  # key: hyperedge size; value: count
        for e in edgeSizeDic:
            sizeFrequence[edgeSizeDic[e]] += 1
        edgeNum = len(edgeDic)
        for node in nodes:
            nodeKMcore_gfDic[node] = sum([sizeFrequence[m]/edgeNum*nodePosation[node][m]/KMmax[m] for m in nodePosation[node]])

        msDic = defaultdict(list)
        for node in nodeKMcore_gfDic:
            msDic[nodeKMcore_gfDic[node]].append(int(node))
        delnode = random.choice(msDic[max(msDic.keys())])

        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start

        return delnode, time_sum

    def gene2Projection(self, nodes, neiEdgeDic, edgeDic):
        G_2proj = nx.Graph()  # compute measures on the 2-projection
        for node in nodes:
            for neiedge in neiEdgeDic[node]:
                for neinode in edgeDic[neiedge]:
                    G_2proj.add_edge(node, neinode)
        G_2proj.remove_edges_from(nx.selfloop_edges(G_2proj))
        return G_2proj

    def geneLineGraph(self, edgeDic):
        LineG = nx.Graph()
        for edge1 in edgeDic:
            for edge2 in edgeDic:
                if len(set(edgeDic[edge1])&set(edgeDic[edge2]))!=0 and edge1!=edge2:
                    LineG.add_edge(edge1, edge2)
        return LineG

    def geneLineGraph_weighted_selfloops(self, edgeDic):
        LineG = nx.Graph()
        for edge1 in edgeDic:
            for edge2 in edgeDic:
                if len(set(edgeDic[edge1])&set(edgeDic[edge2]))!=0:
                    w = (1/3)*(len(set(edgeDic[edge1])|set(edgeDic[edge2])) + 
                               len(set(edgeDic[edge1])|set(edgeDic[edge2]))/len(set(edgeDic[edge1])&set(edgeDic[edge2]))) - 1
                    LineG.add_edge(edge1, edge2, weight=w)
        return LineG

    def reNumber(self):
        node_raw = set()
        for gp in self.groups:
            node_raw.update(set(gp))

        numMap = dict()
        backward = dict()
        node_raw_lst = list(node_raw)
        node_raw_lst.sort()
        for i,node in enumerate(node_raw_lst):
            numMap[node] = i  # start indexing from 0
            backward[i] = node
        groups_new = []
        for gp in self.groups:
            groups_new.append([numMap[i] for i in gp])
            
        temp = []
        for i in groups_new:
            temp.extend(i)
        nodeSet_new = set(temp)
        nodes_new = list(nodeSet_new)
        nodes_new.sort()  # sorted in ascending order
        edgeDic_new = dict()  # key: hyperedge index (int); value: hyperedge as a list of nodes
        edgeSizeDic_new = dict()  # key: hyperedge index (int); value: hyperedge size (int)
        neiNodeEdgeDic_new = defaultdict(list)  # key: node index (int); value: list of tuples (neighbor node, neighboring hyperedge)
        neiEdgeDic_new = defaultdict(list)  # key: node index (int); value: list of incident hyperedges
        neiNodeDic_new = defaultdict(set)  # key: node index (int); value: set of neighboring nodes
        maxHegdeSize_new = 0  # maximum hyperedge size
        nodeDegreeDic_new = defaultdict(int)
        i=0
        for hyperedge in groups_new:
            edgeDic_new[i] = hyperedge
            edgeSizeDic_new[i] = len(hyperedge)
            if len(hyperedge) > maxHegdeSize_new:
                maxHegdeSize_new = len(hyperedge)
            for node in hyperedge:
                nodeDegreeDic_new[node] += 1
                neiEdgeDic_new[node].append(i)
                for neinode in hyperedge:
                    if neinode != node:  # avoid adding node itself into its neighbor list
                        neiNodeEdgeDic_new[node].append((neinode,i))  
                        neiNodeDic_new[node].add(neinode)              
            i += 1
        return numMap, backward, groups_new, edgeDic_new, edgeSizeDic_new, neiNodeEdgeDic_new, maxHegdeSize_new, neiEdgeDic_new, neiNodeDic_new

    def reNumber_pg(self):
        node_raw = set()
        for gp in self.groups:
            node_raw.update(set(gp))

        numMap = dict()
        backward = dict()
        node_raw_lst = list(node_raw)
        node_raw_lst.sort()
        for i,node in enumerate(node_raw_lst):
            numMap[node] = i  # start indexing from 0
            backward[i] = node
        groups_new = []
        for gp in self.groups:
            groups_new.append([numMap[i] for i in gp])
            
        edgeSizeDic_new = dict()  # key: hyperedge index (int); value: hyperedge size (int)
        neiNodeEdgeDic_new = defaultdict(list)  # key: node index (int); value: list of tuples (neighbor node, neighboring hyperedge)
        neiNodeDic_new = defaultdict(set)  # key: node index (int); value: set of neighboring nodes
        i=0
        for hyperedge in groups_new:
            edgeSizeDic_new[i] = len(hyperedge)
            for node in hyperedge:
                for neinode in hyperedge:
                    if neinode != node:  # avoid adding node itself into its neighbor list
                        neiNodeEdgeDic_new[node].append((neinode,i))  
                        neiNodeDic_new[node].add(neinode)              
            i += 1
        return backward, groups_new, edgeSizeDic_new, neiNodeEdgeDic_new, neiNodeDic_new

    def reNumber_ne(self):
        node_raw = set()
        for gp in self.groups:
            node_raw.update(set(gp))

        numMap = dict()
        backward = dict()
        node_raw_lst = list(node_raw)
        node_raw_lst.sort()
        for i,node in enumerate(node_raw_lst):
            numMap[node] = i  # start indexing from 0
            backward[i] = node
        groups_new = []
        for gp in self.groups:
            groups_new.append([numMap[i] for i in gp])
            
        neiEdgeDic_new = defaultdict(list)  # key: node index (int); value: list of incident hyperedges
        i=0
        for hyperedge in groups_new:
            for node in hyperedge:
                neiEdgeDic_new[node].append(i)
            i += 1
        return backward, neiEdgeDic_new

    def geneIncidenceMatrix(self, neiEdgeDic):
        B = np.zeros((self.N, self.M))
        for i in range(self.N):
            for j in neiEdgeDic[i]:
                B[i][j] = 1
        return B
       
    def h_new_neiNodeSum_1(self, IFcalTime=True):
        time_sum = None
        if IFcalTime:
            time_start = time.time()
        temp = []
        for i in self.groups:
            temp.extend(i)
        nodeSet = set(temp)
        nodes = list(nodeSet)
        nodes.sort()  # sorted in ascending order

        edgeSizeDic = dict()  # key: hyperedge index (int); value: hyperedge size (int)
        neiEdgeDic = defaultdict(list)  # key: node index (int); value: list of incident hyperedges
        i=1
        for hyperedge in self.groups:
            edgeSizeDic[i] = len(hyperedge)
            for node in hyperedge:
                neiEdgeDic[node].append(i)
            i += 1

        nodemsDic = dict()
        for node in nodes:
            nodemsDic[node] = float(sum([edgeSizeDic[edge]-1 for edge in neiEdgeDic[node]]))
        
        
        msDic = defaultdict(list)
        for node in nodemsDic:
            msDic[nodemsDic[node]].append(int(node))
        delnode = random.choice(msDic[max(msDic.keys())])

        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start

        return delnode, time_sum
