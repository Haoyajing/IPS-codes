import sys
sys.setrecursionlimit(10000) 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Hsimulation_utils_sp.immunization_ncsis_mc_spdup import *
from Hsimulation_utils_sp.chooseTopNodes import *
from pathos.multiprocessing import ProcessingPool as Pool
import time
import tqdm
import pandas as pd
import json
import random
import math
import numpy as np
from copy import deepcopy

netType = "realNet"
immunizationMtd = ['random', 'degree', 'clsBetweenness', 'clsCloseness', 'clsKcore', 
                    'KMcore_g1', 'KMcore_gf', 'neiNodesNum', 'new_neiNodeSum_1', 'clsEigenvector',
                    'nodeEdgeEigenvector_linear', 'nodeEdgeEigenvector_max']
nets_paras = dict()
nets_paras[1] = {
    1:{'net':1, 'l':[8e-05,  4e-05,  2e-05,  9e-06,]},
    2:{'net':2, 'l':[0.002,  0.0005,  0.0001,  3e-05,]},
    3:{'net':3, 'l':[0.002,  0.0006,  0.0002,  8e-05, ]},
    4:{'net':4, 'l':[0.0006,  0.0002,  5e-05,  1e-05,]},
    5:{'net':5, 'l':[0.002,  0.0007,  0.0003,  8e-05, ]},
    6:{'net':6, 'l':[5e-05,  2e-05,  7e-06,  3e-06, ]},
    7:{'net':7, 'l':[0.003,  0.0009,  0.0004,  0.0002, ]},
    8:{'net':8, 'l':[0.007,  0.003,  0.002,  0.0008, ]},
    9:{'net':9, 'l':[0.0009,  0.0005,  0.0002,  9e-05, ]},
    10:{'net':10, 'l':[0.0006,  0.0004,  0.0002,  0.0001,]},
    11:{'net':11, 'l':[0.0006,  0.0004,  0.0003,  0.0001,]},
    12:{'net':12, 'l':[0.007,  0.005,  0.002,  0.001, ]},
    13:{'net':13, 'l':[0.006,  0.004,  0.003,  0.002,]},
    14:{'net':14, 'l':[0.002,  0.002,  0.0007,  0.0004, ]},
    15:{'net':15, 'l':[0.008,  0.007,  0.005,  0.003, ]},
    16:{'net':16, 'l':[0.008,  0.006,  0.004,  0.003,  ]},
    17:{'net':17, 'l':[0.004,  0.0009,  0.0003,  7e-05, ]},
    18:{'net':18, 'l':[0.004,  0.0009,  0.0004,  0.0002,  ]},
    19:{'net':19, 'l':[0.0009,  0.0003,  0.0002,  6e-05,]},
    20:{'net':20, 'l':[0.0008,  0.0003,  9e-05,  2e-05, ]},
}
nets_paras[0.1] = {
    1:{'net':1, 'l':[8e-06,  3e-06,  2e-06,  5e-07,]},
    2:{'net':2, 'l':[0.0002,  4e-05,  7e-06,  2e-06,]},
    3:{'net':3, 'l':[0.0002,  5e-05,  8e-06,  5e-06, ]},
    4:{'net':4, 'l':[6e-05,  1e-05,  2e-06,  6e-07, ]},
    5:{'net':5, 'l':[0.0002,  6e-05,  2e-05,  4e-06,]},
    6:{'net':6, 'l':[5e-06,  1e-06,  4e-07,  2e-07,]},
    7:{'net':7, 'l':[0.0002,  7e-05,  2e-05,  7e-06,]},
    8:{'net':8, 'l':[0.0007,  0.0002,  6e-05,  3e-05,]},
    9:{'net':9, 'l':[9e-05,  4e-05,  1e-05,  4e-06, ]},
    10:{'net':10, 'l':[6e-05,  3e-05,  2e-05,  7e-06,]},
    11:{'net':11, 'l':[6e-05,  4e-05,  2e-05,  9e-06,]},
    12:{'net':12, 'l':[0.0006,  0.0004,  0.0002,  6e-05,]},
    13:{'net':13, 'l':[0.0005,  0.0003,  0.0002,  8e-05, ]},
    14:{'net':14, 'l':[0.0002,  0.0001,  5e-05,  2e-05, ]},
    15:{'net':15, 'l':[0.0007,  0.0006,  0.0004,  0.0002,]},
    16:{'net':16, 'l':[0.0008,  0.0006,  0.0004,  0.0002,]},
    17:{'net':17, 'l':[0.0003,  8e-05,  2e-05,  3e-06,]},
    18:{'net':18, 'l':[0.0004,  6e-05,  2e-05,  6e-06,]},
    19:{'net':19, 'l':[8e-05,  3e-05,  1e-05,  3e-06,]},
    20:{'net':20, 'l':[8e-05,  2e-05,  6e-06,  2e-06, ]},
}
processNum = 46
######################################
net = 1
#######################################

parasClass = 2  # inirho = 0.2
exprClass = 1
mctimes = 100
tWait=500
tMax=1500
nodeNumDic = {1: 1718, 2: 1290, 3: 420, 4: 580, 5: 1104, 
              6: 294, 7: 282, 8: 143, 9: 979, 10: 339, 
              11: 591, 12: 217, 13: 76, 14: 242, 15: 403, 
              16: 327, 17: 663, 18: 130, 19: 1044, 20: 456}
def main():
    args = [] 
    for mu in [0.1,1]:
        for nu in [1,3]:
            lThres = nets_paras[mu][net]['l'][nu-1]
            l_5 = lThres*5
            l_10 = lThres*10
            for lid,l in enumerate([l_5, l_10]): 
                for mtd in immunizationMtd:
                    args.append([mu, net, nu, lid, l, mtd])
    print(len(args))
    strat_time = time.time()
    pool = Pool(processes=processNum)  
    results = list(tqdm.tqdm(iterable=(pool.imap(run_one_simulation, args)), total=len(args)))
    end_time = time.time()
    print(f'net{net}', 'simulation time = ', (end_time-strat_time)/(60), 'min')  

    resultsDic = dict()
    for item in results:
        resultsDic[item[0]] = item[1]
    filename1 = f'mc_{net}_para{parasClass}_expr{exprClass}_delnode_rho_static.json'
    with open(f'Results/Results_add/{filename1}','w') as f:
        json.dump(resultsDic,f)

def run_one_simulation(args): 
    mu, net, nu, lid, l, mtd = args
    muName = int(mu*10)
    with open(f'Iniconfig/initialAvgINum_mu_net_nu_lid_add1.json', 'r') as f:
        inirhofile = json.load(f)
    rho_ini = inirhofile[f'{mu}_{net}_{nu}_{lid}']
    rhoLstall = []
    for mc in tqdm.tqdm(range(mctimes)):
        # print(mtd,mc)
        rhoLst = [] # output
        with open(f'../../Networks/networks/{netType}_{net}.json', 'r') as fl:
            HyperGraph = json.load(fl)
        nodeOrder = chooseTopNodes('h', f'{netType}_{net}', mtd, nodeNumDic[net])
        
        with open(f'Iniconfig/initialConfigration_mu{muName}_net{net}_nu{nu}_lid{lid}.json', 'r') as f:
            initialConfigration = json.load(f)  
        rho = rho_ini
        for i in range(1,nodeNumDic[net]+1):# tqdm.tqdm(range(1,nodeNumDic[net]+1)):
            if rho > 0:
                delnode = nodeOrder[i-1]
                # print(mtd,'No.', i ,delnode)
                initialConfigrationL = list(initialConfigration)
                if delnode in initialConfigrationL:
                    initialConfigrationL.remove(delnode)
                if len(initialConfigrationL) == 0:  # The final state of the previous time step was only delnode in an infected state
                    HyperGraph = generate_new_net(HyperGraph, delnode)  # Generate a new network
                    immsis = Immunization_SIS(HyperGraph)
                    ccDic = immsis.cal_connected_component()  # Calculate branch
                    rhoLst.append((delnode, rho, ccDic))
                    break
                HyperGraph = generate_new_net(HyperGraph, delnode)  
                immsis = Immunization_SIS(HyperGraph)
                ccDic = immsis.cal_connected_component()  
                infectionProbLst = np.exp(-l * np.arange(immsis.maxHegdeSize) ** nu)
                immsis.prepare_for_numba()
                rho, initialConfigration = calrho_numba_mc(nodeNumDic[net], immsis.nb_Hnodes, immsis.edgeIDs, immsis.nb_HneiEdgeDic, immsis.nb_HedgeDic,
                                                infectionProbLst, initialConfigrationL, mu, tWait=tWait, tMax=tMax)
                rhoLst.append((delnode, rho, ccDic))
            else:
                rhoLst.append((delnode, rho, ccDic))
                break
        rhoLstall.append(rhoLst)
    return [f'mu{muName}_net{net}_nu{nu}_lid{lid}_mtd{mtd}', rhoLstall]

if __name__ == '__main__':
    main()


