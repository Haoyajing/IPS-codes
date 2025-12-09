import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Hsimulation_utils_sp.nonlinearContagion_SIR_spdup_cmpr import *
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
processNum = 24
netType = "realNet"
mctimes = 10000
tMax = float("Inf")
runNets = [8]
muName = int(mu*10)
if len(runNets) == 1:
    netName = runNets[0]
    filename = f'nc_SIR_{netName}_range_mu{muName}.json'
elif len(runNets)==20:
     filename = f'nc_SIR_all_range_mu{muName}.json'
else:
    filename = f'nc_SIR_multi_range_mu{muName}.json'
if mu == 1:
    nets_paras = {
        1:{'net':1, 'l':[3e-4, 2e-4, 7e-5, 6e-5]},
        2:{'net':2, 'l':[4e-3, 1e-3, 9e-4, 7e-4]},
        3:{'net':3, 'l':[7e-3, 2e-3, 2e-3, 1e-3]},
        4:{'net':4, 'l':[5e-3, 1e-3, 5e-4, 3e-4]},
        5:{'net':5, 'l':[6e-3, 2e-3, 2e-3, 1e-3]},
        6:{'net':6, 'l':[8e-5, 4e-5, 2e-5, 2e-5]},
        7:{'net':7, 'l':[3e-3, 2e-3, 2e-3, 1e-3]},
        8:{'net':8, 'l':[1e-2, 9e-3, 5e-3, 4e-3]},
        9:{'net':9, 'l':[7e-3, 2e-3, 2e-3, 7e-4]},
        10:{'net':10, 'l':[1e-3, 1e-3, 8e-4, 5e-4]},
        11:{'net':11, 'l':[1e-3, 9e-4, 7e-4, 5e-4]},
        12:{'net':12, 'l':[2e-2, 2e-2, 2e-2, 2e-2]},
        13:{'net':13, 'l':[1e-2, 1e-2, 1e-2, 8e-3]},
        14:{'net':14, 'l':[3e-3, 2e-3, 2e-3, 2e-3]},
        15:{'net':15, 'l':[2e-2, 2e-2, 1e-2, 1e-2]},
        16:{'net':16, 'l':[2e-2, 2e-2, 1e-2, 9e-3]},
        17:{'net':17, 'l':[9e-3, 3e-3, 2e-3, 2e-3]},
        18:{'net':18, 'l':[5e-3, 2e-3, 2e-3, 9e-4]},
        19:{'net':19, 'l':[4e-3, 1e-3, 6e-4, 5e-4]},
        20:{'net':20, 'l':[1e-3, 6e-4, 4e-4, 3e-4]},
    }
elif mu == 0.1:
    nets_paras = {
        1:{'net':1, 'l':[3e-5, 2e-5, 6e-6, 4e-6]},
        2:{'net':2, 'l':[4e-4, 1e-4, 5e-5, 5e-5]},
        3:{'net':3, 'l':[1e-3, 2e-4, 2e-4, 8e-5]},
        4:{'net':4, 'l':[4e-4, 9e-5, 4e-5, 2e-5]},
        5:{'net':5, 'l':[9e-4, 2e-4, 9e-5, 6e-5]},
        6:{'net':6, 'l':[1e-5, 4e-6, 2e-6, 9e-7]},
        7:{'net':7, 'l':[6e-4, 2e-4, 8e-5, 7e-5]},
        8:{'net':8, 'l':[2e-3, 9e-4, 5e-4, 4e-4]},
        9:{'net':9, 'l':[6e-4, 5e-4, 1e-4, 6e-5]},
        10:{'net':10, 'l':[2e-4, 1e-4, 8e-5, 4e-5]},
        11:{'net':11, 'l':[1e-4, 1e-4, 8e-5, 4e-5]},
        12:{'net':12, 'l':[3e-3, 2e-3, 2e-3, 2e-3]},
        13:{'net':13, 'l':[4e-3, 9e-4, 7e-4, 6e-4]},
        14:{'net':14, 'l':[3e-4, 3e-4, 2e-4, 1e-4]},
        15:{'net':15, 'l':[2e-3, 2e-3, 2e-3, 1e-3]},
        16:{'net':16, 'l':[2e-3, 2e-3, 1e-3, 1e-3]},
        17:{'net':17, 'l':[1e-3, 4e-4, 2e-4, 9e-5]},
        18:{'net':18, 'l':[5e-4, 2e-4, 9e-5, 9e-5]},
        19:{'net':19, 'l':[6e-4, 1e-4, 4e-5, 4e-5]},
        20:{'net':20, 'l':[1e-4, 5e-5, 3e-5, 2e-5]},
    }

def main():
    args = [] 
    for net in runNets:
        for nu in [1,2,3,4]:
            lThres = nets_paras[net]['l'][nu-1]
            for l in [lThres*1.05, lThres*1.1, lThres*1.15, lThres*1.2, lThres*1.5, lThres*1.7, lThres*2, lThres*2.5,]:
                # [lThres*1.2, lThres*1.5, lThres*1.7, lThres*2, lThres*2.5,]:
                # [lThres*0.1, lThres, lThres*3, lThres*5, lThres*7, lThres*10]:# [lThres*0.1, lThres, lThres*10]:
                args.append([net, nu, l])
    
    strat_time = time.time()
    pool = Pool(processes=processNum)  # ProcessNum is the maximum number of processes that can run simultaneously
    results = list(tqdm.tqdm(iterable=(pool.imap(run_one_simulation, args)), total=len(args), desc='Layer0'))
    # results = [([k1,v1],[ke1,va1]), ([k2,v2],[ke2,va2]), ...] len(args) items
    end_time = time.time()
    print('simulation time = ', (end_time-strat_time)/(60), 'min')  # minute
    
    results_dict = dict()
    for smalldic in results:
        for item in smalldic:
            results_dict[item] = smalldic[item]
    with open(f'{filename}','w') as f:
        json.dump(results_dict,f)

def run_one_simulation(args):
    net, nu, l = args

    with open(f'../../Networks/networks/{netType}_{net}.json', 'r') as fl:
        HyperGraph = json.load(fl)  # HyperGraph is a list where each element is a list of numbers (i.e., a hyperedge)
    
    scTSIR = SimuNonlinearContagion_SIR(HyperGraph)
    infectedNodesNum_dic = dict()
    for node in tqdm.tqdm(scTSIR.Hnodes, desc='Layer1'):
        iniNode=node

        # ===========================
        # initialize
        # ===========================
        scTSIR.initialization(initial_infected=[iniNode])
        scTSIR.prepare_for_numba()  # generate self.nodeStates，self.iNodesNum_inHedge， self.nb_HneiEdgeDic， self.nb_HedgeDic
        infectionProbLst = np.exp(-l * np.arange(scTSIR.maxHegdeSize) ** nu)

        save_stats = np.zeros((mctimes, 3))  # 可以删除

        RNodesNum_allmc = []
        I1_allmc = []
        for mc in range(mctimes): #tqdm.tqdm(range(mctimes), desc='Layer2'):  # Monte Carlo experiment with identical propagation conditions
            RNodesNum, propagationLength, InodesNum, SnodesNum, I1 = contagion_simulation_numba(
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
            # RNodesNum_allmc is a cumulative list of infected nodes from these mctimes Monte Carlo experiments, 
            # with a length of mctimes
            RNodesNum_allmc.append(RNodesNum)  
            I1_allmc.append(I1)

        infectedNodesNum_dic[f'n{net}_nu{nu}_l{l}_mu{mu}_ini{iniNode}_range']= sum(RNodesNum_allmc)/mctimes 
        infectedNodesNum_dic[f'n{net}_nu{nu}_l{l}_mu{mu}_ini{iniNode}_I1'] = sum(I1_allmc)/mctimes
    return infectedNodesNum_dic

if __name__ == '__main__':
    main()