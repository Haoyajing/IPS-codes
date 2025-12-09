import sys
from Hsimulation_utils import *
from pathos.multiprocessing import ProcessingPool as Pool
import time
import tqdm
import pandas as pd

netType = 'realNet'
# processNum = 16
nets = [i for i in range(1,21)]
mctimes = 300
processNum = 46
mu = 0.1
tMax = float("Inf")
iniI = 1
nuLst = [1,2,3,4]
lLst = [
    1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5,
    1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4,
    1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 
    1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 
    1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1,
    ]

# lLst = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2,]
# lLst = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3,]
# lLst = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4,]
# lLst = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 1e-5,]
# lLst = [1e-7, 2e-7, 3e-7, 4e-7, 5e-7, 6e-7, 7e-7, 8e-7, 9e-7, 1e-6,]
# lLst = [1e-8, 2e-8, 3e-8, 4e-8, 5e-8, 6e-8, 7e-8, 8e-8, 9e-8, 1e-7,]

def main():
    args = []
    for n in nets:
        for nu in nuLst:
            for l in lLst:
                args.append([n, nu, l])

    strat_time = time.time()
    pool = Pool(processes=processNum)  
    results = list(tqdm.tqdm(iterable=(pool.imap(run_one_simulation, args)), total=len(args)))
    end_time = time.time()
    print('simulation time = ', (end_time-strat_time)/(60), 'min')  # 分钟
    susceptibility = dict(results)
    muName = int(mu*10)
    with open(f'HNCSize_ini1_result_all_mu{muName}.json', 'w') as f:
        json.dump(susceptibility, f)


def run_one_simulation(args):
    net, nu, l = args
    with open(f'../../../Networks/networks/{netType}_{net}.json', 'r') as fl:
        HyperGraph = json.load(fl)
    simuSIR = Simu_NCsize_SIR(HyperGraph)
    RNodesNum_allmc = []
    initialInfectedNodes = iniI # int(iniI*simuSIR.N)+1
    for mc in range(mctimes):
        # print(mc, end=' ')
        simuSIR.initialization_random(randomInfect=initialInfectedNodes)
        RNodesNum, propagationLength, InodesNum, SnodesNum = simuSIR.contagionSimulation(l, mu, nu, tMax)
        RNodesNum_allmc.append(RNodesNum)
    n_avg = sum(RNodesNum_allmc)/mctimes
    if n_avg != 0:
        susceptibility = (sum([x*x for x in RNodesNum_allmc])/mctimes - pow(n_avg,2))/n_avg
    else:
        susceptibility = 0
    return [f'net{net}_nu{nu}_l{l}', susceptibility]

if __name__ == '__main__':
    main()