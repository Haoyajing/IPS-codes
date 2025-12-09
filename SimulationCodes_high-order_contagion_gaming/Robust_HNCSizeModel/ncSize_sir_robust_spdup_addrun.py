import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Hsimulation_utils_sp.ncSize_SIR_spdup import *
from pathos.multiprocessing import ProcessingPool as Pool
import time
import tqdm
import pandas as pd
import json
import random
import math
import numpy as np
from copy import deepcopy

mu = 1
runNets = [2,5,7,17]
# runNets = [i for i in range(1,21)]
processNum = 47
netType = "realNet"
if mu ==  1:
    mctimes = 1000
elif mu == 0.1:
    mctimes = 300
tMax = float("Inf")
#Traversing all nodes is too slow. 
# Here, we randomly select 50% of the nodes as initial nodes for propagation simulation
sanmpleRatio = 0.5  
muName = int(mu*10)
if len(runNets) == 1:
    netName = runNets[0]
    filename = f'addrun_ncSize_SIR_{netName}_robust_range_mu{muName}.json'
elif len(runNets)==20:
     filename = f'addrun_ncSize_SIR_all_robust_range_mu{muName}.json'
else:
    filename = f'addrun_ncSize_SIR_multi_robust_range_mu{muName}.json'
if mu == 1:
    nets_paras = {
    1:{'net':1, 'l':[6e-3, 2e-3, 1e-3, 7e-4]},
    2:{'net':2, 'l':[2e-1, 6e-2, 5e-2, 4e-2]},
    3:{'net':3, 'l':[9e-2, 5e-2, 3e-2, 2e-2]},
    4:{'net':4, 'l':[9e-2, 3e-2, 2e-2, 1e-2]},
    5:{'net':5, 'l':[3e-1, 7e-2, 5e-2, 3e-2]},
    6:{'net':6, 'l':[2e-3, 1e-3, 7e-4, 6e-4]},
    7:{'net':7, 'l':[1e-1, 4e-2, 3e-2, 2e-2]},
    8:{'net':8, 'l':[8e-2, 9e-2, 4e-2, 3e-2]},
    9:{'net':9, 'l':[3e-2, 3e-2, 1e-2, 8e-3]},
    10:{'net':10, 'l':[6e-3, 5e-3, 4e-3, 3e-3]},
    11:{'net':11, 'l':[4e-3, 4e-3, 4e-3, 3e-3]},
    12:{'net':12, 'l':[6e-2, 5e-2, 6e-2, 5e-2]},
    13:{'net':13, 'l':[4e-2, 4e-2, 4e-2, 3e-2]},
    14:{'net':14, 'l':[1e-2, 1e-2, 9e-3, 6e-3]},
    15:{'net':15, 'l':[5e-2, 5e-2, 6e-2, 4e-2]},
    16:{'net':16, 'l':[5e-2, 4e-2, 4e-2, 4e-2]},
    17:{'net':17, 'l':[3e-1, 1e-1, 7e-2, 6e-2]},
    18:{'net':18, 'l':[9e-2, 5e-2, 3e-2, 3e-2]},
    19:{'net':19, 'l':[9e-2, 4e-2, 2e-2, 2e-2]},
    20:{'net':20, 'l':[5e-2, 3e-2, 2e-2, 2e-2]},
    }
elif mu == 0.1:
    nets_paras = {
        1:{'net':1, 'l':[1e-3, 3e-4, 9e-5, 8e-5]},
        2:{'net':2, 'l':[2e-2, 6e-3, 3e-3, 2e-3]},
        3:{'net':3, 'l':[2e-2, 8e-3, 2e-3, 2e-3]},
        4:{'net':4, 'l':[1e-2, 4e-3, 2e-3, 8e-4]},
        5:{'net':5, 'l':[4e-2, 7e-3, 4e-3, 2e-3]},
        6:{'net':6, 'l':[3e-4, 1e-4, 5e-5, 4e-5]},
        7:{'net':7, 'l':[9e-3, 4e-3, 2e-3, 2e-3]},
        8:{'net':8, 'l':[9e-3, 9e-3, 4e-3, 2e-3]},
        9:{'net':9, 'l':[3e-3, 3e-3, 1e-3, 7e-4]},
        10:{'net':10, 'l':[9e-4, 7e-4, 4e-4, 2e-4]},
        11:{'net':11, 'l':[6e-4, 6e-4, 4e-4, 2e-4]},
        12:{'net':12, 'l':[7e-3, 6e-3, 9e-3, 5e-3]},
        13:{'net':13, 'l':[6e-3, 6e-3, 4e-3, 4e-3]},
        14:{'net':14, 'l':[1e-3, 1e-3, 8e-4, 5e-4]},
        15:{'net':15, 'l':[9e-3, 5e-3, 5e-3, 4e-3]},
        16:{'net':16, 'l':[6e-3, 6e-3, 5e-3, 5e-3]},
        17:{'net':17, 'l':[5e-2, 1e-2, 5e-3, 4e-3]},
        18:{'net':18, 'l':[1e-2, 5e-3, 3e-3, 2e-3]},
        19:{'net':19, 'l':[1e-2, 6e-3, 2e-3, 8e-4]},
        20:{'net':20, 'l':[5e-3, 3e-3, 2e-3, 8e-4]},
        }

def main():
    args = []  
    for net in runNets:
        for nu in [1,2]:
            lThres = nets_paras[net]['l'][nu-1]
            l_up = lThres*10
            for l in [l_up]:
                args.append([net, nu, l])
    
    strat_time = time.time()
    pool = Pool(processes=processNum)  
    results = list(tqdm.tqdm(iterable=(pool.imap(run_one_simulation, args)), total=len(args), desc='Layer0'))
    # results = [([k1,v1],[ke1,va1]), ([k2,v2],[ke2,va2]), ...]
    end_time = time.time()
    print('simulation time = ', (end_time-strat_time)/(60), 'min') 
    
    results_dict = dict()
    for smalldic in results:
        for item in smalldic:
            results_dict[item] = smalldic[item]
    with open(f'{filename}','w') as f:
        json.dump(results_dict,f)

def run_one_simulation(args):  
    net, nu, l = args

    with open(f'../../Networks/networks/{netType}_{net}.json', 'r') as fl:
        HyperGraph = json.load(fl)  
    
    scTSIR = Simu_NCsize_SIR(HyperGraph)
    infectedNodesNum_dic = dict()
    sampledNodes = random.sample(scTSIR.Hnodes, math.ceil(len(scTSIR.Hnodes)*sanmpleRatio))
    sampledNodes.sort()  
    for node in tqdm.tqdm(sampledNodes, desc='Layer1'):
        iniNode=node

        # ===========================
        # initialize
        # ===========================
        scTSIR.initialization(initial_infected=[iniNode])
        scTSIR.prepare_for_numba()  # generate self.nodeStates，self.iNodesNum_inHedge， self.nb_HneiEdgeDic， self.nb_HedgeDic，self.nb_HedgeSize
        infectionProbDic = dict()
        for sizeE in range(1, scTSIR.maxHegdeSize+1):
            infectionProbDic[sizeE] = np.exp((-l/sizeE) * np.arange(0, sizeE+1) ** nu)
        infectionProbDic_numba = dict2numba_float(infectionProbDic)
        
        RNodesNum_allmc = []
        for mc in range(mctimes): #tqdm.tqdm(range(mctimes), desc='Layer2'):  
            
            # ===========================
            # contagion Simulation
            # ===========================
            RNodesNum, propagationLength, InodesNum, SnodesNum = contagion_simulation_numba(
                deepcopy(scTSIR.nodeStates), 
                deepcopy(scTSIR.iNodesNum_inHedge), 
                scTSIR.nb_HneiEdgeDic, 
                scTSIR.nb_HedgeDic, 
                scTSIR.nb_HedgeSize,
                infectionProbDic_numba, mu, tMax
            )


            if RNodesNum+SnodesNum != scTSIR.N:
                print(iniNode, 'add error')
            if InodesNum !=0 :
                print(iniNode, '0 error')
            RNodesNum_allmc.append(RNodesNum)  

        infectedNodesNum_dic[f'n{net}_nu{nu}_l{l}_mu{mu}_ini{iniNode}_range']= sum(RNodesNum_allmc)/mctimes  
        
    return infectedNodesNum_dic

if __name__ == '__main__':
    main()