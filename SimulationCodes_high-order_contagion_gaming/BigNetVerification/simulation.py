import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Hsimulation_utils_sp.nonlinearContagion_SIR_spdup import *
from pathos.multiprocessing import ProcessingPool as Pool
import time
import tqdm
import pandas as pd
import json
import random
import math
import numpy as np
from copy import deepcopy
# import psutil

mu = 1
processNum = 40
netType = "realNet"
mctimes = 1000
tMax = float("Inf")
sanmpleRatio = 0.1  # randomly select 10% of nodes as initial seeds instead of traversing all nodes
net = 25
nuLst = [1]
sp=24  # split the initial nodes into sp parts
muName = int(mu*10)
filename = f'SIR_{net}_robust_range_mu{muName}_add_nu1_1.json'
if mu == 1:
    nets_paras = {
        #25:{'net':25, 'l':[5e-2, 1e-1]},  # nu=3
        25:{'net':25, 'l':[1e-1]},  # nu = 1, case 1
    }
# elif mu == 0.1:
#     nets_paras = {
#     }
memory_threshold = 100 * 1024 * 1024 * 1024  # memory threshold = 100 GB

def main():
    args = []  # each element corresponds to one process
    with open(f'iniNodes_25.json', 'r') as fl:
        iniNodes = json.load(fl)
    iniNodes.sort()  # list of hypergraph node IDs (int) in ascending order
    # split iniNodes into sp parts
    spnodes = []
    onepiece = int(len(iniNodes)/sp)
    for i in range(sp-1):
        spnodes.append(iniNodes[i*onepiece: (i+1)*onepiece])
    spnodes.append(iniNodes[(sp-1)*onepiece:])
    tqdmposition = 1
    for nu in nuLst:
        for l in nets_paras[net]['l']:
            for initialInfectedNodes in spnodes:
                args.append([net, nu, l, initialInfectedNodes, tqdmposition])
                tqdmposition+=1

    strat_time = time.time()
    pool = Pool(processes=processNum)  # processNum is the maximum number of concurrent worker processes (typically set to the number of CPU cores)
    results = list(tqdm.tqdm(iterable=(pool.imap(run_one_simulation, args)), total=len(args), desc='all'))
    # results is a list of dictionaries, one per run
    end_time = time.time()
    print('simulation time = ', (end_time-strat_time)/(60), 'min')
    
    results_dict = dict()
    for smalldic in results:
        for item in smalldic:
            results_dict[item] = smalldic[item]
    with open(f'{filename}','w') as f:
        json.dump(results_dict,f)

# def check_memory_usage():
#     process = psutil.Process(os.getpid())
#     memory_info = process.memory_info()
#     memory_usage = memory_info.rss  # Resident Set Size (RSS) in bytes
#     # print(f"Current memory usage: {memory_usage / (1024 * 1024):.2f} MB")
#
#     if memory_usage > memory_threshold:
#         print(f"Memory usage exceeded threshold ({memory_threshold / (1024 * 1024):.2f} MB). Exiting process...")
#         sys.exit(1)

def run_one_simulation(args):
    # one call of run_one_simulation corresponds to one run with fixed parameters and mctimes Monte Carlo trials
    # check_memory_usage()
    net, nu, l, initialInfectedNodes, tqdmposition = args

    with open(f'../../Networks/networks/{netType}_{net}.json', 'r') as fl:
        HyperGraph = json.load(fl)  # HyperGraph is a list; each element is a list of node IDs
    
    scTSIR = SimuNonlinearContagion_SIR(HyperGraph)
    infectedNodesNum_dic = dict()
    # sampledNodes = random.sample(scTSIR.Hnodes, math.ceil(len(scTSIR.Hnodes)*sanmpleRatio))
    # sampledNodes.sort()  # list of node IDs (int) in ascending order
    for node in tqdm.tqdm(initialInfectedNodes, desc=f'lid {tqdmposition}'):
        iniNode=node

        # ===========================
        # Speedup 1/2: precompute data needed for the initialization part
        # ===========================
        scTSIR.initialization(initial_infected=[iniNode])
        scTSIR.prepare_for_numba()  # generate self.nodeStates, self.iNodesNum_inHedge, self.nb_HneiEdgeDic, self.nb_HedgeDic
        infectionProbLst = np.exp(-l * np.arange(scTSIR.maxHegdeSize) ** nu)

        RNodesNum_allmc = []
        for mc in range(mctimes):  # Monte Carlo trials with identical parameters
            
            # ===========================
            # Speedup 2/2: use numba-accelerated contagion simulation
            # ===========================
            RNodesNum, propagationLength, InodesNum, SnodesNum = contagion_simulation_numba(
                deepcopy(scTSIR.nodeStates), 
                deepcopy(scTSIR.iNodesNum_inHedge), 
                scTSIR.nb_HneiEdgeDic, 
                scTSIR.nb_HedgeDic, 
                infectionProbLst, mu, tMax
            )
             

            if RNodesNum+SnodesNum != scTSIR.N:
                print(iniNode, 'add error')
            if InodesNum !=0 :
                print(iniNode, '0 error')
            # RNodesNum_allmc stores the final number of recovered nodes in each Monte Carlo trial (length = mctimes)
            RNodesNum_allmc.append(RNodesNum)

        # key encodes the parameter setting and initial node, value is the average outbreak size over mctimes trials
        infectedNodesNum_dic[f'n{net}_nu{nu}_l{l}_mu{mu}_ini{iniNode}_range']= sum(RNodesNum_allmc)/mctimes
    
    return infectedNodesNum_dic

if __name__ == '__main__':
    main()
