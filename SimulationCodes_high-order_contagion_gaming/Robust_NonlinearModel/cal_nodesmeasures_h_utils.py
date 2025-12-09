import networkx as nx
import time
from collections import defaultdict
from collections import Counter
import os
import json
import numpy as np
from scipy import sparse 
import copy
import math
# import xgi

"""

"""

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
                    if neinode != node:  # avoid adding node itself to its neighbor list
                        self.neiNodeEdgeDic[node].append((neinode,i))
                        self.neiNodeDic[node].add(neinode)
            i += 1

        
    def gene2Projection(self, nodes, neiEdgeDic, edgeDic):
        G_2proj = nx.Graph()  # use 2-projection to build a simple graph
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

    def calMeasuresNodes(self, nodeMeasureDic, fileName):
        msDic = defaultdict(list)
        for node in nodeMeasureDic:
            msDic[nodeMeasureDic[node]].append(int(node))
        keysLstStr = msDic.keys()
        # check whether there exist floats that are equal as numbers but stored as different string keys
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
        # msDic['duplicates'] = duplicates
        
        with open(f'measures_t2_robust/{fileName}_rvs.json',"w") as f:
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
            numMap[node] = i  # reindex starting from 0
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


    def h_new_new_new_rough_t2_iNum(self, parasClass, nu, l, mu, IFcalTime=False, IFcover=False):
        def controlSize(e,s):
            if e <= s:
                return e
            else:
                return s
        def keepNoneNegtive(n):
            if n>0:
                return n
            else:
                return 0
        filename = f'h_new_new_new_rough_t2_iNum{parasClass}_{self.net}'
        if os.path.exists(f'measures_t2_robust/measure_t2_robust_rough/{filename}.json') and not IFcover:
            print(f"{filename}.json existed, check parameter: \"IFcover\"")
            return None
        
        time_sum = None
        if IFcalTime:
            time_start = time.time()
        
        t_1 = dict()
        nodeneiSum_aw = dict()
        for node in self.nodes:
            allneiN = []
            for neiE in self.neiEdgeDic[node]:
                allneiN.extend(self.edgeDic[neiE])
            neiFreq = Counter(allneiN)
            del neiFreq[node]  # multiplicity of all neighboring nodes, with the node itself removed
            t_1[node] = l*sum([neiFreq[neiN] for neiN in neiFreq])
            psum = sum([neiFreq[_] for _ in neiFreq])

            # compute higher-/lower-order spreading within the first-order neighborhood of the source
            firstNei = 0
            for neiE in self.neiEdgeDic[node]:
                Einfec = sum([neiFreq[nn] for nn in self.edgeDic[neiE]])  # estimated number of infected nodes in this first-order hyperedge
                Einfec *= l
                # Einfec = controlSize(Einfec,self.edgeSizeDic[neiE]-1)  # optionally cap the estimated number of infected nodes in the hyperedge
                if Einfec >= 1:
                    firstNei += (1-mu)*controlSize(l*pow(Einfec+1,nu), 1)*keepNoneNegtive(self.edgeSizeDic[neiE]-Einfec-1) +\
                        mu*controlSize(l*pow(Einfec,nu), 1)*keepNoneNegtive(self.edgeSizeDic[neiE]-Einfec-1)
                else:
                    firstNei += (1-mu)*controlSize(l*pow(Einfec+1,nu), 1)*keepNoneNegtive(self.edgeSizeDic[neiE]-Einfec-1) +\
                        mu*controlSize(l*Einfec, 1)*keepNoneNegtive(self.edgeSizeDic[neiE]-Einfec-1)
                # if Einfec >= 1:
                #     firstNei += (1-mu)*(1-pow(math.e, -1*l*pow(Einfec+1,nu)))*keepNoneNegtive(self.edgeSizeDic[neiE]-Einfec-1) +\
                #         mu*(1-pow(math.e, -1*l*pow(Einfec,nu)))*keepNoneNegtive(self.edgeSizeDic[neiE]-Einfec-1)
                # else:
                #     firstNei += (1-mu)*(1-pow(math.e, -1*l*pow(Einfec+1,nu)))*keepNoneNegtive(self.edgeSizeDic[neiE]-Einfec-1) +\
                #         mu*Einfec*(1-pow(math.e, -1*l*1))*keepNoneNegtive(self.edgeSizeDic[neiE]-Einfec-1)
            
            # compute spreading in second-order neighbors caused by infected neighbors of the source
            secondNeiEdic = defaultdict(int)  # dictionary of all second-order hyperedges
            oneNeiNum = defaultdict(int)  # intersection size between second-order hyperedges and the first-order neighborhood
            for neiN in neiFreq:
                for neiN_neiE in self.neiEdgeDic[neiN]:
                    if neiN_neiE not in self.neiEdgeDic[node]:  # hyperedges incident to neighbors but not to the source are second-order hyperedges
                        # neiFreq[neiN]/psum is the infection probability of neiN; summed and multiplied by l gives the expected number of infection sources in this second-order hyperedge
                        secondNeiEdic[neiN_neiE] += neiFreq[neiN]
                        # oneNeiNum[neiN_neiE] += 1
            secondNei = 0
            for sneiE in secondNeiEdic:
                Einfec = l*secondNeiEdic[sneiE]  # expected number of infection sources in the current second-order hyperedge
                # (1-pow(math.e,-1*l*secondNeiEdic[sneiE]))
                if Einfec >= 1:
                    # secondNei += (1-pow(math.e, -1*l*pow(Einfec,nu)))*(self.edgeSizeDic[sneiE]-oneNeiNum[neiN_neiE])
                    secondNei += controlSize(l*pow(Einfec,nu), 1)*keepNoneNegtive(self.edgeSizeDic[sneiE]-Einfec)
                else:
                    secondNei += l*Einfec*keepNoneNegtive(self.edgeSizeDic[sneiE]-Einfec)

            # nodeneiSum_aw[node] = firstNei + (1-mu)*nodeneiSum_1[node] + secondNei
            nodeneiSum_aw[node] = [t_1[node], firstNei, secondNei, t_1[node]+firstNei+secondNei]
        print(self.net,'\t', Einfec, '\t', l*secondNeiEdic[sneiE])

        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start
        
        # if os.path.exists(f'measures_t2_robust/{filename}.json') and IFcover:
        #     print(f"{filename}.json c")
        #     # recompute corresponding rvs
        #     self.calMeasuresNodes(nodeneiSum_aw, filename)
        # else:
        #     print(f"{filename}.json generated")
        #     # compute corresponding rvs
        #     self.calMeasuresNodes(nodeneiSum_aw, filename)
        with open(f'measures_t2_robust/measure_t2_robust_rough/{filename}.json',"w") as f:
            json.dump(nodeneiSum_aw,f)
        
        return time_sum
    

    def h_new_precise_prob_t2_iNum(self, parasClass, nu, l, mu, IFcalTime=False, IFcover=False):
        filename = f'h_new_precise_prob_t2_iNum{parasClass}_{self.net}'
        if os.path.exists(f'measures_t2_robust/{filename}.json') and not IFcover:
            print(f"{filename}.json existed, check parameter: \"IFcover\"")
            return None
        
        time_sum = None
        if IFcalTime:
            time_start = time.time()

        t_1 = dict()
        t_2_1 = dict()
        t_2_2 = dict()
        nodeneiSum_aw = dict()
        for node in self.nodes:
        # for node in [3]:
            allneiN = []
            neinode_edge = defaultdict(list)  # for each first-order neighbor, list of first-order hyperedges it belongs to
            for neiE in self.neiEdgeDic[node]:  # neiE is a hyperedge incident to node
                allneiN.extend(self.edgeDic[neiE])
                for neiN in self.edgeDic[neiE]:
                    neinode_edge[neiN].append(neiE)
            neiFreq = Counter(allneiN)
            del neiFreq[node]  # multiplicity of all neighboring nodes, with the node itself removed
            del neinode_edge[node]
            t_1[node] = sum([1-pow(math.e, -1*neiFreq[neiN]*l) for neiN in neiFreq])
            
            # compute spreading in second-order neighbors caused by infected neighbors of the source
            # dictionary: for each second-order neighbor, list of second-order hyperedges
            neinode_edge_2 = defaultdict(list)
            neinode_edge_1 = defaultdict(set)
            neinode_edge_1_ = defaultdict(set)
            for neiN in neiFreq:
                for neiN_neiE in self.neiEdgeDic[neiN]:
                    if neiN_neiE not in self.neiEdgeDic[node]:  # found a second-order hyperedge
                        # handle the case where a second-order hyperedge is fully contained in the first-order neighborhood
                        if set(self.edgeDic[neiN_neiE]).issubset(self.neiNodeDic[node]):
                            neinode_edge_1_[neiN].add(neiN_neiE)
                            continue
                            # print(node,neiN_neiE)
                        for nei2 in self.edgeDic[neiN_neiE]:
                            if nei2 not in neiFreq:  # found a second-order neighbor
                                neinode_edge_1[neiN].add(neiN_neiE)  # second-order hyperedges incident to first-order neighbor neiN
                                neinode_edge_2[nei2].append((neiN_neiE, neiN))  # second-order neighbor nei2 is connected to neiN via neiN_neiE
            # expected number of infected nodes in each second-order hyperedge for each second-order neighbor
            infNum_2 = dict()
            for nei2 in neinode_edge_2:  # iterate over second-order neighbors and compute expected number of infected nodes in each second-order hyperedge
                infNum_2[nei2] = defaultdict(float)
                for pairs in neinode_edge_2[nei2]:
                    infNum_2[nei2][pairs[0]] += 1-pow(math.e, -1*neiFreq[pairs[1]]*l)
            # aggregate expected infected counts within each second-order hyperedge
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
            # compute higher-/lower-order spreading within the first-order neighborhood of the source
            # dictionary of infection levels in first-order hyperedges; key = hyperedge index, value = expected number of infected nodes
            neiE1_infNum = dict()
            for neiE in self.neiEdgeDic[node]:
                neiE1_infNum[neiE] = 0  # expected number of infected nodes in each first-order hyperedge (excluding the source node)
                for n1 in self.edgeDic[neiE]:
                    if n1 != node:
                        neiE1_infNum[neiE] += 1-pow(math.e, -1*neiFreq[n1]*l)
            neiE1_infNum_now = defaultdict(float)
            for neiN1 in neiFreq:  # the sum over first-order neighbors of their infection probability at t2 gives expected number of newly infected nodes at distance 1
                temp1 = 1  # probability of not being infected in all hyperedges
                for neiE1 in neinode_edge[neiN1]:
                    # estimate number of infected nodes in neiE1 excluding neiN1 itself
                    neiE1_iNum_other = neiE1_infNum[neiE1]-(1-pow(math.e, -1*neiFreq[neiN1]*l))
                    if neiE1_iNum_other >= 1:  # higher-order transmission in neiE1 while remaining uninfected
                        temp1 *= ((1-mu)*pow(math.e, -1*l*pow(neiE1_iNum_other+1,nu)) + mu*pow(math.e, -1*l*pow(neiE1_iNum_other,nu)))
                    else:   # lower-order transmission in neiE1 while remaining uninfected
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
            for nei2 in infNum_2:  # iterate over all second-order neighbors and compute infection probability for each
                temp = 0
                tempp = 1
                for neiE2 in infNum_2[nei2]:  # neinode_edge_2[nei2] is the list of hyperedges incident to nei2
                    if infNum_2[nei2][neiE2] >= 1:  # infNum_2[nei2][neiE2] is expected number of infected nodes in this second-order hyperedge
                        temp += pow(infNum_2[nei2][neiE2],nu)
                    else:
                        tempp *= (1-infNum_2[nei2][neiE2]*(1-pow(math.e, -1*l*1)))
                neiE2_infNum_now[nei2] = (1-pow(math.e, -1*l*temp)*tempp)
            t_2_2[node] = sum([neiE2_infNum_now[_] for _ in neiE2_infNum_now])

            nodeneiSum_aw[node] = [t_1[node], t_2_1[node], t_2_2[node]]

        if IFcalTime:
            time_end = time.time()  # record end time
            time_sum = time_end - time_start
        
        # if os.path.exists(f'measures_t2_robust/{filename}.json') and IFcover:
        #     print(f"{filename}.json covered")
        #     # recompute corresponding rvs
        #     self.calMeasuresNodes(nodeneiSum_aw, filename)
        # else:
        #     print(f"{filename}.json generated")
        #     # compute corresponding rvs
        #     self.calMeasuresNodes(nodeneiSum_aw, filename)
        with open(f'measures_t2_robust/{filename}.json',"w") as f:
            json.dump(nodeneiSum_aw,f)
        
        return time_sum
