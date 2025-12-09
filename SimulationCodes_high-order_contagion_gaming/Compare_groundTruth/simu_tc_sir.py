import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Hsimulation_utils_sp.thresholdContagion_SIR_spdup_cmpr import *
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
processNum = 18
netType = "realNet"
mctimes = 10000
tMax = float("Inf")
runNets = [8]
muName = int(mu*10)
if len(runNets) == 1:
    netName = runNets[0]
    filename = f'tc_SIR_{netName}_range_mu{muName}.json'
elif len(runNets)==20:
     filename = f'tc_SIR_all_range_mu{muName}.json'
else:
    filename = f'tc_SIR_multi_range_mu{muName}.json'
if mu == 1:
    nets_paras = {
        1:{'net':1, 'lbd':[0.0009,  0.009,  0.09, ]},
        6:{'net':6, 'lbd':[0.0009,  0.005,  0.05,]},
        8:{'net':8, 'lbd':[0.03,  0.05,  0.1,]},
        9:{'net':9, 'lbd':[0.009,  0.02,  0.09,]},
        10:{'net':10, 'lbd':[0.002,  0.006,  0.2,]},
        11:{'net':11, 'lbd':[0.002,  0.003,  0.05,]},
        12:{'net':12, 'lbd':[0.02,  0.03,  0.1,]},
        13:{'net':13, 'lbd':[0.02,  0.02,  0.2,]},
        14:{'net':14, 'lbd':[0.003,  0.006,  0.2,]},
        15:{'net':15, 'lbd':[0.02,  0.02,  0.07,]},
        16:{'net':16, 'lbd':[0.02,  0.02,  0.2, ]},
    }
elif mu == 0.1:
    nets_paras = {
        1:{'net':1, 'lbd':[0.0002,  0.001,  0.007, ]},
        6:{'net':6, 'lbd':[7e-05,  0.0006,  0.005,]},
        8:{'net':8, 'lbd':[0.004,  0.005,  0.02, ]},
        9:{'net':9, 'lbd':[0.0009,  0.002,  0.006,]},
        10:{'net':10, 'lbd':[0.0003,  0.0007,  0.01,]},
        11:{'net':11, 'lbd':[0.0002,  0.0003,  0.005, ]},
        12:{'net':12, 'lbd':[0.004,  0.003,  0.01,]},
        13:{'net':13, 'lbd':[0.002,  0.004,  0.02,]},
        14:{'net':14, 'lbd':[0.0004,  0.001,  0.02, ]},
        15:{'net':15, 'lbd':[0.002,  0.003,  0.01,]},
        16:{'net':16, 'lbd':[0.003,  0.003,  0.02,]},
    }
maxEsize = {
    1:25, 2:81, 3:107, 4:230, 5:83, 6:99, 7:31, 8:37, 9:25, 10:16,
    11:13, 12:10, 13:7, 14:10, 15:10, 16:7, 17:124, 18:104, 19:58, 20:157
}

def main():
    args = []  # The number of elements is the total process
    for net in runNets:
        for id,theta in enumerate([1/maxEsize[net], 1/4, 1/2]):
            lThres = nets_paras[net]['lbd'][id]
            if lThres*10 >= 1:
                l_up = lThres*2 
            else:
                l_up = lThres*10
            for lbd in [lThres*0.7, lThres*0.8, lThres*0.9, lThres, lThres*1.1, lThres*1.2]:
                args.append([net, theta, lbd])
    
    strat_time = time.time()
    pool = Pool(processes=processNum)  # ProcessNum is the maximum number of processes that can run simultaneously
    results = list(tqdm.tqdm(iterable=(pool.imap(run_one_simulation, args)), total=len(args), desc='Layer0'))
    # results = [([k1,v1],[ke1,va1]), ([k2,v2],[ke2,va2]), ...] len(args) items
    end_time = time.time()
    print('simulation time = ', (end_time-strat_time)/(60), 'min')  # minutes
    
    results_dict = dict()
    for smalldic in results:
        for item in smalldic:
            results_dict[item] = smalldic[item]
    with open(f'{filename}','w') as f:
        json.dump(results_dict,f)

def run_one_simulation(args): 
    net,theta, lbd = args

    with open(f'../../Networks/networks/{netType}_{net}.json', 'r') as fl:
        HyperGraph = json.load(fl)  # list of hyperedges
    
    scTSIR = SimuThresholdContagion_SIR(HyperGraph)
    infectedNodesNum_dic = dict()
    for node in tqdm.tqdm(scTSIR.Hnodes, desc='Layer1'):
        iniNode=node
        # ===========================
        # initialize
        # ===========================
        scTSIR.initialization(initial_infected=[iniNode])
        scTSIR.prepare_for_numba()  # generate self.nodeStates，self.iNodesNum_inHedge, self.nb_HedgeDic, self.nb_HedgeSize
        
        RNodesNum_allmc = []
        I1_allmc = []
        for mc in range(mctimes): #tqdm.tqdm(range(mctimes), desc='Layer2'):  # Monte Carlo experiment
            
            RNodesNum, propagationLength, InodesNum, SnodesNum, I1 = contagion_simulation_numba(
                deepcopy(scTSIR.nodeStates), 
                deepcopy(scTSIR.iNodesNum_inHedge), 
                scTSIR.nb_HedgeDic, 
                scTSIR.nb_HedgeSize,
                theta, lbd, mu, tMax
            )


            if RNodesNum+SnodesNum != scTSIR.N:
                print(iniNode, 'add error')
            if InodesNum !=0 :
                print(iniNode, '0 error')
            # RNodesNum_allmc is a cumulative list of infected nodes from these mctimes Monte Carlo experiments, 
            # with a length of mctimes
            RNodesNum_allmc.append(RNodesNum)  
            I1_allmc.append(I1)
        
        infectedNodesNum_dic[f'n{net}_theta{theta}_lbd{lbd}_mu{mu}_ini{iniNode}_range']= sum(RNodesNum_allmc)/mctimes
        infectedNodesNum_dic[f'n{net}_theta{theta}_lbd{lbd}_mu{mu}_ini{iniNode}_I1'] = sum(I1_allmc)/mctimes
    return infectedNodesNum_dic

if __name__ == '__main__':
    main()