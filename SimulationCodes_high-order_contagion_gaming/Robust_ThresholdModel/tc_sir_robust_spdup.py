import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Hsimulation_utils_sp.thresholdContagion_SIR_spdup import *
from pathos.multiprocessing import ProcessingPool as Pool
import time
import tqdm
import pandas as pd
import json
import random
import math
import numpy as np
from copy import deepcopy

mu = 0.1
processNum = 43
netType = "realNet"
if mu==0.1:
    mctimes = 300
elif mu==1:
    mctimes = 1000
tMax = float("Inf")
# Traversing all nodes is too slow. 
# Here, we randomly select 50% of the nodes as initial nodes for propagation simulation
sanmpleRatio = 0.5 
runNets = [1,6,8,9,10,11,12,13,14,15,16]
muName = int(mu*10)
if len(runNets) == 1:
    netName = runNets[0]
    filename = f'tc_SIR_{netName}_robust_range_mu{muName}.json'
elif len(runNets)==20:
     filename = f'tc_SIR_all_robust_range_mu{muName}.json'
else:
    filename = f'tc_SIR_multi_robust_range_mu{muName}.json'
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
    args = []  
    for net in runNets:
        for id,theta in enumerate([1/maxEsize[net], 1/4, 1/2]):
            lThres = nets_paras[net]['lbd'][id]
            if lThres*10 >= 1:
                l_up = lThres*2 
            else:
                l_up = lThres*10
            for lbd in [lThres*0.1, lThres, l_up]:
                args.append([net, theta, lbd])
    
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
    net,theta, lbd = args

    with open(f'../../Networks/networks/{netType}_{net}.json', 'r') as fl:
        HyperGraph = json.load(fl)  
    
    scTSIR = SimuThresholdContagion_SIR(HyperGraph)
    infectedNodesNum_dic = dict()
    sampledNodes = random.sample(scTSIR.Hnodes, math.ceil(len(scTSIR.Hnodes)*sanmpleRatio))
    sampledNodes.sort()  
    for node in tqdm.tqdm(sampledNodes, desc='Layer1'):
        iniNode=node
        # ===========================
        # initialize
        # ===========================
        scTSIR.initialization(initial_infected=[iniNode])
        scTSIR.prepare_for_numba()  # generate self.nodeStates，self.iNodesNum_inHedge, self.nb_HedgeDic, self.nb_HedgeSize
        
        RNodesNum_allmc = []
        for mc in range(mctimes): #tqdm.tqdm(range(mctimes), desc='Layer2'): 
            
            # ===========================
            # contagion Simulation
            # ===========================
            RNodesNum, propagationLength, InodesNum, SnodesNum = contagion_simulation_numba(
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
            RNodesNum_allmc.append(RNodesNum) 

        infectedNodesNum_dic[f'n{net}_theta{theta}_lbd{lbd}_mu{mu}_ini{iniNode}_range']= sum(RNodesNum_allmc)/mctimes 
    
    return infectedNodesNum_dic

if __name__ == '__main__':
    main()