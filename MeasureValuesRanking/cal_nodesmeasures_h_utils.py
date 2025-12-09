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

class CalNodesMeasures():
    def __init__(self, group_list, net):
        self.net = net  # network name
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
        self.neiEdgeDic = defaultdict(list)  # key: node index (int); value: list of adjacent hyperedges
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
        
    def h_degree(self, IFcalTime=False, IFcover=False):
        if os.path.exists(f'NodesMeasures/h_degree_{self.net}.json') and not IFcover:
            print(f"h_degree_{self.net}.json existed, check parameter: \"IFcover\"")
            return None
        
        time_sum = None
        if IFcalTime:
            time_start = time.time()
        
        nodeDegreeDic = defaultdict(float)

        for gp in self.groups:
            for node in gp:
                nodeDegreeDic[node] += 1

        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start

        filename = f'h_degree_{self.net}'
        if os.path.exists(f'NodesMeasures/{filename}.json') and IFcover:
            print(f"{filename}.json covered")
            # recompute corresponding rvs
            self.calMeasuresNodes(nodeDegreeDic, filename)
        else:
            print(f"{filename}.json generated")
            # compute corresponding rvs
            self.calMeasuresNodes(nodeDegreeDic, filename)
        with open(f'NodesMeasures/{filename}.json',"w") as f:
            json.dump(nodeDegreeDic,f)
        
        return time_sum

    def h_neiNodesNum(self, IFcalTime=False, IFcover=False):
        filename = f'h_neiNodesNum_{self.net}'
        if os.path.exists(f'NodesMeasures/{filename}.json') and not IFcover:
            print(f"{filename}.json existed, check parameter: \"IFcover\"")
            return None
        
        time_sum = None
        if IFcalTime:
            time_start = time.time()
        
        nodeDegreeDic = dict()

        for node in self.neiNodeDic:
            nodeDegreeDic[node] = float(len(self.neiNodeDic[node]))

        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start

        if os.path.exists(f'NodesMeasures/{filename}.json') and IFcover:
            print(f"{filename}.json covered")
            # recompute corresponding rvs
            self.calMeasuresNodes(nodeDegreeDic, filename)
        else:
            print(f"{filename}.json generated")
            # compute corresponding rvs
            self.calMeasuresNodes(nodeDegreeDic, filename)
        with open(f'NodesMeasures/{filename}.json',"w") as f:
            json.dump(nodeDegreeDic,f)
        
        return time_sum

    def h_clsCloseness(self, IFcalTime=False, IFcover=False):
        if os.path.exists(f'NodesMeasures/h_clsCloseness_{self.net}.json') and not IFcover:
            print(f"h_clsCloseness_{self.net}.json existed, check parameter: \"IFcover\"")
            return None
        
        time_sum = None
        if IFcalTime:
            time_start = time.time()
        
        G_2proj  =self.gene2Projection(self.nodes, self.neiEdgeDic, self.edgeDic)  # generate 2-projection graph
        nodeclsClosenessDic = nx.closeness_centrality(G_2proj)
        
        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start

        filename = f'h_clsCloseness_{self.net}'
        if os.path.exists(f'NodesMeasures/{filename}.json') and IFcover:
            print(f"{filename}.json covered")
            # recompute corresponding rvs
            self.calMeasuresNodes(nodeclsClosenessDic, filename)
        else:
            print(f"{filename}.json generated")
            # compute corresponding rvs
            self.calMeasuresNodes(nodeclsClosenessDic, filename)
        with open(f'NodesMeasures/{filename}.json',"w") as f:
            json.dump(nodeclsClosenessDic,f)
        
        return time_sum
    
    def h_clsBetweenness(self, IFcalTime=False, IFcover=False):
        if os.path.exists(f'NodesMeasures/h_clsBetweenness_{self.net}.json') and not IFcover:
            print(f"h_clsBetweenness_{self.net}.json existed, check parameter: \"IFcover\"")
            return None
        
        time_sum = None
        if IFcalTime:
            time_start = time.time()
        
        G_2proj = self.gene2Projection(self.nodes, self.neiEdgeDic, self.edgeDic)  # generate 2-projection graph
        nodeclsBetweennessDic = nx.betweenness_centrality(G_2proj)
        
        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start

        filename = f'h_clsBetweenness_{self.net}'
        if os.path.exists(f'NodesMeasures/{filename}.json') and IFcover:
            print(f"{filename}.json covered")
            # recompute corresponding rvs
            self.calMeasuresNodes(nodeclsBetweennessDic, filename)
        else:
            print(f"{filename}.json generated")
            # compute corresponding rvs
            self.calMeasuresNodes(nodeclsBetweennessDic, filename)
        with open(f'NodesMeasures/{filename}.json',"w") as f:
            json.dump(nodeclsBetweennessDic,f)
        
        return time_sum 

    def h_clsEigenvector(self, IFcalTime=False, IFcover=False):
        filename = f'h_clsEigenvector_{self.net}'
        if os.path.exists(f'NodesMeasures/{filename}.json') and not IFcover:
            print(f"{filename}.json existed, check parameter: \"IFcover\"")
            return None
        
        time_sum = None
        if IFcalTime:
            time_start = time.time()

        G_2proj = self.gene2Projection(self.nodes, self.neiEdgeDic, self.edgeDic)  # generate 2-projection graph
        nodemsDic = nx.eigenvector_centrality(G_2proj)
        
        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start
        
        if os.path.exists(f'NodesMeasures/{filename}.json') and IFcover:
            print(f"{filename}.json covered")
            # recompute corresponding rvs
            self.calMeasuresNodes(nodemsDic, filename)
        else:
            print(f"{filename}.json generated")
            # compute corresponding rvs
            self.calMeasuresNodes(nodemsDic, filename)
        with open(f'NodesMeasures/{filename}.json',"w") as f:
            json.dump(nodemsDic,f)
        
        return time_sum
    
    def h_nodeEdgeEigenvector(self, funcGrp='linear', IFcalTime=False, IFcover=False):
        filename = f'h_nodeEdgeEigenvector_{funcGrp}_{self.net}'
        if os.path.exists(f'NodesMeasures/{filename}.json') and not IFcover:
            print(f"{filename}.json existed, check parameter: \"IFcover\"")
            return None

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

        renumDic, backwardDic, groups_new, edgeDic_new, edgeSizeDic_new, neiNodeEdgeDic_new, maxHegdeSize_new, neiEdgeDic_new, neiNodeDic_new = self.reNumber()

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
            print(f'{self.net} {filename} not converge')
        
        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start

        if backwardDic:
            # map back to original node indices
            nodeEdgeEigenvectorDic = dict()
            for id, vl in enumerate(x):
                nodeEdgeEigenvectorDic[backwardDic[id]] = vl
        
        if os.path.exists(f'NodesMeasures/{filename}.json') and IFcover:
            print(f"{filename}.json covered")
            # recompute corresponding rvs
            self.calMeasuresNodes(nodeEdgeEigenvectorDic, filename)
        else:
            print(f"{filename}.json generated")
            # compute corresponding rvs
            self.calMeasuresNodes(nodeEdgeEigenvectorDic, filename)
        with open(f'NodesMeasures/{filename}.json',"w") as f:
            json.dump(nodeEdgeEigenvectorDic,f)

        return time_sum

    def h_clsKcore(self, IFcalTime=False, IFcover=False):
        if os.path.exists(f'NodesMeasures/h_clsKcore_{self.net}.json') and not IFcover:
            print(f"h_clsKcore_{self.net}.json existed, check parameter: \"IFcover\"")
            return None
        
        time_sum = None
        if IFcalTime:
            time_start = time.time()
        
        G_2proj = self.gene2Projection(self.nodes, self.neiEdgeDic, self.edgeDic)  # generate 2-projection graph
        nodeclsKcoreDic_int = nx.core_number(G_2proj)
        nodeclsKcoreDic = dict()
        for item in nodeclsKcoreDic_int:
            nodeclsKcoreDic[item] = float(nodeclsKcoreDic_int[item])
        
        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start

        filename = f'h_clsKcore_{self.net}'
        if os.path.exists(f'NodesMeasures/{filename}.json') and IFcover:
            print(f"{filename}.json covered")
            # recompute corresponding rvs
            self.calMeasuresNodes(nodeclsKcoreDic, filename)
        else:
            print(f"{filename}.json generated")
            # compute corresponding rvs
            self.calMeasuresNodes(nodeclsKcoreDic, filename)
        with open(f'NodesMeasures/{filename}.json',"w") as f:
            json.dump(nodeclsKcoreDic,f)
        
        return time_sum

    def h_KMcore(self, IFcalTime=False, IFcover=False):
        if os.path.exists(f'NodesMeasures/h_KMcore_gf_{self.net}.json') and not IFcover:
            print(f"h_KMcore_gf_{self.net}.json existed, check parameter: \"IFcover\"")
            return None

        time_sum = None
        if IFcalTime:
            time_start = time.time()

        maxK = max([self.nodeDegreeDic[_] for _ in self.nodeDegreeDic])
        nodeKMcore_g1Dic = defaultdict(float)
        nodeKMcore_gfDic = defaultdict(float)

        nodePosation = defaultdict(dict)
        mshells = defaultdict(dict)
        for m in range(2, self.maxHegdeSize+1):
            currentHGraph = set()
            for edge in self.edgeSizeDic:  # extract all hyperedges with size >= m to form sub-hypergraph currentHGraph
                if self.edgeSizeDic[edge] >= m:
                    currentHGraph.add(frozenset(self.edgeDic[edge]))
            
            for k in range(1, maxK+1):
                degDic = defaultdict(int)
                for hedge in currentHGraph:  # compute node degrees and find nodes with degree < k in the subgraph
                    for node in hedge:
                        degDic[node] += 1
                nodes_prevShell = list(degDic.keys())

                nodeDelSet = ([_ for _ in degDic if degDic[_] < k])
                tempHG_nodeDeled = set([frozenset(set(e) - set(nodeDelSet)) for e in currentHGraph])
                edgeDelSet = set([e for e in tempHG_nodeDeled if len(e) < m])
                tempHG_edgeDeled = set([e for e in tempHG_nodeDeled if len(e) >= m])
                currentHGraph = copy.deepcopy(tempHG_edgeDeled)

                if len(nodeDelSet) != 0 or len(edgeDelSet) != 0:
                    while len(nodeDelSet) != 0 or len(edgeDelSet) != 0:
                        degDic = defaultdict(int)
                        for hedge in currentHGraph:  # iteratively delete nodes with degree < k
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
                elif len(currentHGraph) != 0:  # hypergraph not fully decomposed, increase k
                    continue
                else:  # decomposition finished
                    break
        KMmax = {m:max(list(mshells[m].keys())) for m in mshells}
        sizeFrequence = defaultdict(int)  # size: count
        for e in self.edgeSizeDic:
            sizeFrequence[self.edgeSizeDic[e]] += 1
        edgeNum = len(self.edgeDic)
        for node in self.nodes:
            nodeKMcore_gfDic[node] = sum([sizeFrequence[m]/edgeNum*nodePosation[node][m]/KMmax[m] for m in nodePosation[node]])

        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start

        for node in self.nodes:
            nodeKMcore_g1Dic[node] = sum([nodePosation[node][m]/KMmax[m] for m in nodePosation[node]])

        filename = f'h_KMcore_g1_{self.net}'
        if os.path.exists(f'NodesMeasures/{filename}.json') and IFcover:
            print(f"{filename}.json covered")
            # recompute corresponding rvs
            self.calMeasuresNodes(nodeKMcore_g1Dic, filename)
        else:
            print(f"{filename}.json generated")
            # compute corresponding rvs
            self.calMeasuresNodes(nodeKMcore_g1Dic, filename)
        with open(f'NodesMeasures/{filename}.json',"w") as f:
            json.dump(nodeKMcore_g1Dic,f)

        filename = f'h_KMcore_gf_{self.net}'
        if os.path.exists(f'NodesMeasures/{filename}.json') and IFcover:
            print(f"{filename}.json covered")
            # recompute corresponding rvs
            self.calMeasuresNodes(nodeKMcore_gfDic, filename)
        else:
            print(f"{filename}.json generated")
            # compute corresponding rvs
            self.calMeasuresNodes(nodeKMcore_gfDic, filename)
        with open(f'NodesMeasures/{filename}.json',"w") as f:
            json.dump(nodeKMcore_gfDic,f)
        
        return time_sum

    def gene2Projection(self, nodes, neiEdgeDic, edgeDic):
        G_2proj = nx.Graph()  # 2-projection graph
        for node in nodes:
            for neiedge in neiEdgeDic[node]:
                for neinode in edgeDic[neiedge]:
                    G_2proj.add_edge(node, neinode)
        G_2proj.remove_edges_from(nx.selfloop_edges(G_2proj))
        return G_2proj

    def calMeasuresNodes(self, nodeMeasureDic, fileName):
        msDic = defaultdict(list)
        for node in nodeMeasureDic:
            msDic[nodeMeasureDic[node]].append(int(node))
        keysLstStr = msDic.keys()
        keysLst = [float(i) for i in keysLstStr]  # check if there are duplicate float keys due to string representation
        keysSet = set(keysLst)
        duplicates = []
        if len(keysLst) == len(keysSet):
            print(f"no duplicates, {fileName}_rvs generated")
        else:
            for element in keysLst:
                if keysLst.count(element) > 1 and element not in duplicates:
                    duplicates.append(element)
            print(f"duplicates={duplicates}, {fileName}_rvs generated, further processing needed!")
        
        with open(f'NodesMeasures/{fileName}_rvs.json',"w") as f:
            json.dump(msDic, f)

    def reNumber(self):
        node_raw = set()
        for gp in self.groups:
            node_raw.update(set(gp))

        numMap = dict()
        backward = dict()
        node_raw_lst = list(node_raw)
        node_raw_lst.sort()
        for i,node in enumerate(node_raw_lst):
            numMap[node] = i  # reindex from 0
            backward[i] = node
        groups_new = []
        for gp in self.groups:
            groups_new.append([numMap[i] for i in gp])
            
        temp = []
        for i in groups_new:
            temp.extend(i)
        nodeSet_new = set(temp)
        nodes_new = list(nodeSet_new)
        nodes_new.sort()  # ascending order by default
        edgeDic_new = dict()  # key: hyperedge index (int); value: list of nodes in the hyperedge
        edgeSizeDic_new = dict()  # key: hyperedge index (int); value: size of the hyperedge (int)
        neiNodeEdgeDic_new = defaultdict(list)  # key: node index (int); value: list of tuples (neighbor node, neighboring hyperedge)
        neiEdgeDic_new = defaultdict(list)  # key: node index (int); value: list of adjacent hyperedges
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
                    if neinode != node:  # avoid adding node itself to its neighbor list
                        neiNodeEdgeDic_new[node].append((neinode,i))  
                        neiNodeDic_new[node].add(neinode)              
            i += 1
        return numMap, backward, groups_new, edgeDic_new, edgeSizeDic_new, neiNodeEdgeDic_new, maxHegdeSize_new, neiEdgeDic_new, neiNodeDic_new

    def geneIncidenceMatrix(self, neiEdgeDic):
        B = np.zeros((self.N, self.M))
        for i in range(self.N):
            for j in neiEdgeDic[i]:
                B[i][j] = 1
        return B

    def h_new_neiNodeSum_1(self, IFcalTime=False, IFcover=False):
        filename = f'h_new_neiNodeSum_1_{self.net}'
        if os.path.exists(f'NodesMeasures/{filename}.json') and not IFcover:
            print(f"{filename}.json existed, check parameter: \"IFcover\"")
            return None
        
        time_sum = None
        if IFcalTime:
            time_start = time.time()
        
        nodemsDic = dict()
        for node in self.nodes:
            nodemsDic[node] = float(sum([self.edgeSizeDic[edge]-1 for edge in self.neiEdgeDic[node]]))
        
        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start
        
        if os.path.exists(f'NodesMeasures/{filename}.json') and IFcover:
            print(f"{filename}.json covered")
            # recompute corresponding rvs
            self.calMeasuresNodes(nodemsDic, filename)
        else:
            print(f"{filename}.json generated")
            # compute corresponding rvs
            self.calMeasuresNodes(nodemsDic, filename)
        with open(f'NodesMeasures/{filename}.json',"w") as f:
            json.dump(nodemsDic,f)
        
        return time_sum
    
    def h_new_precise_prob_t2_iNum_p(self, parasClass, nu, l, mu, IFcalTime=False, IFcover=False):
        filename = f'h_new_precise_prob_t2_iNum_p{parasClass}_{self.net}'
        if os.path.exists(f'NodesMeasures/{filename}.json') and not IFcover:
            print(f"{filename}.json existed, check parameter: \"IFcover\"")
            return None
        
        time_sum = None
        if IFcalTime:
            time_start = time.time()

        t_1 = dict()
        t_2_1 = dict()
        t_2_2 = dict()
        nodeneiSum_aw_i = dict()
        nodeneiSum_aw_ir = dict()
        for node in self.nodes:
            allneiN = []
            neinode_edge = defaultdict(list)  # for each first-order neighbor, list of incident first-order hyperedges
            for neiE in self.neiEdgeDic[node]:  # neiE is a hyperedge incident to node
                allneiN.extend(self.edgeDic[neiE])
                for neiN in self.edgeDic[neiE]:
                    neinode_edge[neiN].append(neiE)
            neiFreq = Counter(allneiN)
            del neiFreq[node]  # remove node itself from neighbor multiplicity
            del neinode_edge[node]
            t_1[node] = sum([1-pow(math.e, -1*neiFreq[neiN]*l) for neiN in neiFreq])
            
            # compute second-order spreading in the source's neighborhood caused by infected neighbors
            # dictionary: for each second-order neighbor, list of incident second-order hyperedges
            neinode_edge_2 = defaultdict(list)
            neinode_edge_1 = defaultdict(set)
            neinode_edge_1_ = defaultdict(set)
            for neiN in neiFreq:
                for neiN_neiE in self.neiEdgeDic[neiN]:
                    if neiN_neiE not in self.neiEdgeDic[node]:  # second-order hyperedge
                        # handle second-order hyperedges fully contained in the first-order neighborhood
                        if set(self.edgeDic[neiN_neiE]).issubset(self.neiNodeDic[node]):
                            neinode_edge_1_[neiN].add(neiN_neiE)
                            continue
                        for nei2 in self.edgeDic[neiN_neiE]:
                            if nei2 not in neiFreq:  # second-order neighbor node
                                neinode_edge_1[neiN].add(neiN_neiE)  # second-order hyperedges incident to first-order neighbor neiN
                                neinode_edge_2[nei2].append((neiN_neiE, neiN))  # second-order neighbor nei2 is connected to neiN via neiN_neiE
            # expected number of infected nodes in each second-order hyperedge for each second-order neighbor
            infNum_2 = dict()
            for nei2 in neinode_edge_2:  # loop over second-order neighbors
                infNum_2[nei2] = defaultdict(float)
                for pairs in neinode_edge_2[nei2]:
                    infNum_2[nei2][pairs[0]] += 1-pow(math.e, -1*neiFreq[pairs[1]]*l)
            # aggregate expected infected counts in each second-order hyperedge
            infNum_2_1 = dict()
            for nei2 in infNum_2:
                for E2 in infNum_2[nei2]:
                    if E2 not in infNum_2_1:
                        infNum_2_1[E2]=infNum_2[nei2][E2]
                    elif infNum_2_1[E2] != infNum_2[nei2][E2]:
                        print('wrong')
            for nei1 in neiFreq:
                for e2_ in neinode_edge_1_[nei1]:
                    if e2_ not in infNum_2_1:
                        infNum_2_1[e2_] = sum([1-pow(math.e, -1*neiFreq[_]*l) for _ in self.edgeDic[e2_]])
            # compute higher-/lower-order spreading in the first-order neighborhood
            # dictionary of expected number of infected nodes in each first-order hyperedge
            neiE1_infNum = dict()
            for neiE in self.neiEdgeDic[node]:
                neiE1_infNum[neiE] = 0  # expected number of infected nodes in each first-order hyperedge (excluding source node)
                for n1 in self.edgeDic[neiE]:
                    if n1 != node:
                        neiE1_infNum[neiE] += 1-pow(math.e, -1*neiFreq[n1]*l)
            neiE1_infNum_now = defaultdict(float)
            for neiN1 in neiFreq:  # sum infection probabilities of first-order neighbors at t2
                temp1 = 1  # probability of not being infected across all hyperedges
                for neiE1 in neinode_edge[neiN1]:
                    # expected number of infected nodes in neiE1 excluding neiN1 itself
                    neiE1_iNum_other = neiE1_infNum[neiE1]-(1-pow(math.e, -1*neiFreq[neiN1]*l))
                    if neiE1_iNum_other >= 1:  # higher-order transmission in neiE1
                        temp1 *= ((1-mu)*pow(math.e, -1*l*pow(neiE1_iNum_other+1,nu)) + mu*pow(math.e, -1*l*pow(neiE1_iNum_other,nu)))
                    else:   # lower-order transmission in neiE1
                        temp1 *= ((1-mu)*pow(math.e, -1*l*pow(neiE1_iNum_other+1,nu)) + mu*(1-neiE1_iNum_other*(1-pow(math.e, -1*l*1))))
                temp2 = 0
                tempp2 = 1
                e2_nei1 = neinode_edge_1[neiN1]|neinode_edge_1_[neiN1]
                for neiE2 in e2_nei1:
                    other = infNum_2_1[neiE2]-(1-pow(math.e, -1*neiFreq[neiN1]*l))
                    if other >= 1:
                        temp2 += pow(other,nu)
                    else:
                        tempp2 *= (1-other*(1-pow(math.e, -1*l*1)))
                neiE1_infNum_now[neiN1] = (1-temp1*pow(math.e, -1*l*temp2)*tempp2)*(pow(math.e, -1*l*neiFreq[neiN1]))
            t_2_1[node] = sum([neiE1_infNum_now[_] for _ in neiE1_infNum_now])

            
            neiE2_infNum_now = defaultdict(float)
            for nei2 in infNum_2:  # loop over second-order neighbors and compute infection probabilities
                temp = 0
                tempp = 1
                for neiE2 in infNum_2[nei2]:
                    if infNum_2[nei2][neiE2] >= 1:
                        temp += pow(infNum_2[nei2][neiE2],nu)
                    else:
                        tempp *= (1-infNum_2[nei2][neiE2]*(1-pow(math.e, -1*l*1)))
                neiE2_infNum_now[nei2] = (1-pow(math.e, -1*l*temp)*tempp)
            t_2_2[node] = sum([neiE2_infNum_now[_] for _ in neiE2_infNum_now])

            nodeneiSum_aw_i[node] = (1-mu)*t_1[node] + t_2_1[node] + t_2_2[node]
            nodeneiSum_aw_ir[node] = t_1[node] + t_2_1[node] + t_2_2[node]

        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start
        
        filename = f'h_new_precise_prob_t2_iNum_p{parasClass}_{self.net}'
        if os.path.exists(f'NodesMeasures/{filename}.json') and IFcover:
            print(f"{filename}.json covered")
            # recompute corresponding rvs
            self.calMeasuresNodes(nodeneiSum_aw_i, filename)
        else:
            print(f"{filename}.json generated")
            # compute corresponding rvs
            self.calMeasuresNodes(nodeneiSum_aw_i, filename)
        with open(f'NodesMeasures/{filename}.json',"w") as f:
            json.dump(nodeneiSum_aw_i,f)

        filename = f'h_new_precise_prob_t2_irNum_p{parasClass}_{self.net}'
        if os.path.exists(f'NodesMeasures/{filename}.json') and IFcover:
            print(f"{filename}.json covered")
            # recompute corresponding rvs
            self.calMeasuresNodes(nodeneiSum_aw_ir, filename)
        else:
            print(f"{filename}.json generated")
            # compute corresponding rvs
            self.calMeasuresNodes(nodeneiSum_aw_ir, filename)
        with open(f'NodesMeasures/{filename}.json',"w") as f:
            json.dump(nodeneiSum_aw_ir,f)
        
        return time_sum

    def h_new_neiNodeSum_1_HNCsize(self, IFcalTime=False, IFcover=False):
        # for higher-order naming game
        filename = f'h_new_neiNodeSum_1_HNCsize_{self.net}'
        if os.path.exists(f'NodesMeasures/{filename}.json') and not IFcover:
            print(f"{filename}.json existed, check parameter: \"IFcover\"")
            return None
        
        time_sum = None
        if IFcalTime:
            time_start = time.time()
        
        nodemsDic = dict()
        for node in self.nodes:
            nodemsDic[node] = float(sum([(1-1/self.edgeSizeDic[edge]) for edge in self.neiEdgeDic[node]]))
        
        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start
        
        if os.path.exists(f'NodesMeasures/{filename}.json') and IFcover:
            print(f"{filename}.json covered")
            # recompute corresponding rvs
            self.calMeasuresNodes(nodemsDic, filename)
        else:
            print(f"{filename}.json generated")
            # compute corresponding rvs
            self.calMeasuresNodes(nodemsDic, filename)
        with open(f'NodesMeasures/{filename}.json',"w") as f:
            json.dump(nodemsDic,f)
        
        return time_sum

    def h_degree_random(self, IFcalTime=False, IFcover=False):
        filename = f'h_degree_random_{self.net}'
        if os.path.exists(f'NodesMeasures/{filename}.json') and not IFcover:
            print(f"{filename}.json existed, check parameter: \"IFcover\"")
            return None
        
        time_sum = None
        if IFcalTime:
            time_start = time.time()
        
        nodeDegreeDic = defaultdict(float)

        for gp in self.groups:
            for node in gp:
                nodeDegreeDic[node] += 1
        for item in nodeDegreeDic:
            nodeDegreeDic[item] += random.random()

        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start
        
        if os.path.exists(f'NodesMeasures/{filename}.json') and IFcover:
            print(f"{filename}.json covered")
            # recompute corresponding rvs
            self.calMeasuresNodes(nodeDegreeDic, filename)
        else:
            print(f"{filename}.json generated")
            # compute corresponding rvs
            self.calMeasuresNodes(nodeDegreeDic, filename)
        with open(f'NodesMeasures/{filename}.json',"w") as f:
            json.dump(nodeDegreeDic,f)
        
        return time_sum

    def h_tc4_sum_1(self, IFcalTime=False, IFcover=False):
        filename = f'h_tc4_sum_1_{self.net}'
        if os.path.exists(f'NodesMeasures/{filename}.json') and not IFcover:
            print(f"{filename}.json existed, check parameter: \"IFcover\"")
            return None
        
        time_sum = None
        if IFcalTime:
            time_start = time.time()
    
        tc4Dic = defaultdict(float)

        for node in self.nodes:
            tc4Dic[node] = 0.0
            for edg in self.neiEdgeDic[node]:
                if self.edgeSizeDic[edg]<=4:
                    tc4Dic[node] += self.edgeSizeDic[edg]-1

        if IFcalTime:
            time_end = time.time() 
            time_sum = time_end - time_start
        
        if os.path.exists(f'NodesMeasures/{filename}.json') and IFcover:
            print(f"{filename}.json covered")
            self.calMeasuresNodes(tc4Dic, filename)
        else:
            print(f"{filename}.json generated")
            self.calMeasuresNodes(tc4Dic, filename)
        with open(f'NodesMeasures/{filename}.json',"w") as f:
            json.dump(tc4Dic,f)
        
        return time_sum

    def h_tc2_sum_1(self, IFcalTime=False, IFcover=False):
        filename = f'h_tc2_sum_1_{self.net}'
        if os.path.exists(f'NodesMeasures/{filename}.json') and not IFcover:
            print(f"{filename}.json existed, check parameter: \"IFcover\"")
            return None
        
        time_sum = None
        if IFcalTime:
            time_start = time.time()
    
        tc2Dic = defaultdict(float)

        for node in self.nodes:
            tc2Dic[node] = 0.0
            for edg in self.neiEdgeDic[node]:
                if self.edgeSizeDic[edg]<=2:
                    tc2Dic[node] += self.edgeSizeDic[edg]-1

        if IFcalTime:
            time_end = time.time() 
            time_sum = time_end - time_start
        
        if os.path.exists(f'NodesMeasures/{filename}.json') and IFcover:
            print(f"{filename}.json covered")
            self.calMeasuresNodes(tc2Dic, filename)
        else:
            print(f"{filename}.json generated")
            self.calMeasuresNodes(tc2Dic, filename)
        with open(f'NodesMeasures/{filename}.json',"w") as f:
            json.dump(tc2Dic,f)
        
        return time_sum

