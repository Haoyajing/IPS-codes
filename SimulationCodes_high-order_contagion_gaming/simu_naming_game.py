from cmath import sqrt
from multiprocessing import pool
import numpy as np
from Hsimulation_utils import *
from pathos.multiprocessing import ProcessingPool as Pool
import pickle
import tqdm
import time

netType = "realNet"
net = 1
agreementRule = 'unanimity'
immunizationMtd = ['degree', 'clsBetweenness', 'clsCloseness', 'clsKcore', 'clsEigenvector', 'KMcore_g1', 'KMcore_gf', 
           'nodeEdgeEigenvector_linear', 'nodeEdgeEigenvector_max', 'PageRank', 'random', 'neiNodesNum', 'new_neiNodeSum_1_HNCsize', 'new_neiNodeSum_1']

processNum = 46
#dynamic parameters
parasClass = 1
betaLst_ = np.linspace(0, 1, 31)
betaLst = betaLst_[1:]
if agreementRule == 'unanimity':
    if net == 1:
        staratp=0.1*1e-2
        endp=2.8*1e-2
        pieceNum=50
        tmax = 1*1e6
        T = int(1*1e5)
    elif net == 2:
        staratp=0.1*1e-2
        endp=1.8*1e-2
        pieceNum=22
        tmax = 1*1e6 # 5*1e5
        T = int(1*1e5) # int(5*1e4)
    elif net == 9:
        staratp=0.1*1e-2
        endp=2.8*1e-2
        pieceNum=27
        tmax = 1*1e6 # 5*1e5
        T = int(1*1e5) # int(5*1e4)
    elif net == 5:
        staratp=0.1*1e-2
        endp=2.8*1e-2
        pieceNum=31
        tmax = 1*1e6 # 1*1e5
        T = int(1*1e5) # int(1*1e4)
    elif net == 11:
        staratp=0.2*1e-2
        endp=8.8*1e-2
        pieceNum=50
        tmax = 1*1e6 # 1*1e5
        T = int(1*1e5) # int(1*1e4)
    elif net == 12:
        staratp=0.5*1e-2
        endp=9.4*1e-2
        pieceNum=20
        tmax = 1*1e6 # 1*1e5
        T = int(1*1e5) # int(1*1e4)
    elif net == 19:
        staratp=0.1*1e-2
        endp=2.8*1e-2
        pieceNum=50
        tmax = 1*1e6
        T = int(1*1e5)
elif agreementRule == 'union':
    if net == 1:
        staratp=0.1*1e-2
        endp=2.8*1e-2
        pieceNum=50
        tmax = 1*1e6
        T = int(1*1e5)
    elif net == 2:
        staratp=0.1*1e-2
        endp=1.8*1e-2
        pieceNum=22
        tmax = 1*1e6 # 5*1e5
        T = int(1*1e5) # int(5*1e4)
    elif net == 9:
        staratp=0.1*1e-2
        endp=2.8*1e-2
        pieceNum=27
        tmax = 1*1e6 # 5*1e5
        T = int(1*1e5) # int(5*1e4)
    elif net == 5:
        staratp=0.1*1e-2
        endp=2.8*1e-2
        pieceNum=31
        tmax = 1*1e6 # 1*1e5
        T = int(1*1e5) # int(1*1e4)
    elif net == 11:
        staratp=0.2*1e-2
        endp=2.9*1e-2
        pieceNum=17
        tmax = 1*1e6 # 1*1e5
        T = int(1*1e5) # int(1*1e4)
    elif net == 12:
        staratp=0.5*1e-2
        endp=3.2*1e-2
        pieceNum=7
        tmax = 1*1e6 # 1*1e5
        T = int(1*1e5) # int(1*1e4)
    elif net == 19:
        staratp=0.1*1e-2
        endp=2.8*1e-2
        pieceNum=50
        tmax = 1*1e6
        T = int(1*1e5)
pLst = np.linspace(staratp, endp, pieceNum, endpoint=False)
#experiment parameters
exprClass = 3
mctimes = 100 
check_every = 100
sampleNum = 100

def main():
    args = []  
    for mtd in immunizationMtd:
        for idb, beta in enumerate(betaLst):
            for idp, p in enumerate(pLst):
                args.append([mtd, idb+1, beta, idp+1, p])
    
    strat_time = time.time()
    pool = Pool(processes=processNum) 
    results = list(tqdm.tqdm(iterable=(pool.imap(run_one_simulation, args)), total=len(args)))
    end_time = time.time()
    print(f'net{net}', 'simulation time = ', (end_time-strat_time)/(60), 'min') 
    
    nameRatios_results = [item for item in results]
    nameRatios_results_dict = dict(nameRatios_results)

    out_dir1 = f'../SimulationResults/NamingGame/'
    filename1 = f'{agreementRule}_{net}_para{parasClass}_expr{exprClass}_nameRatios.json'
    with open(f'{out_dir1+filename1}','w') as f:
        json.dump(nameRatios_results_dict,f)


def run_one_simulation(args): 
    mtd, idb, beta, idp, p = args

    with open(f'../Networks/networks/{netType}_{net}.json', 'r') as fl:
        HyperGraph = json.load(fl)
    
    simuNG = SimuNamingGame(HyperGraph)
    if p!=0 and int(simuNG.N*p) == 0:
        commedNodes = chooseTopNodes('h', f'{netType}_{net}', mtd, 1)
    else:
        commedNodes = chooseTopNodes('h', f'{netType}_{net}', mtd, int(simuNG.N*p))

    nameRatios_allmc = []
    for mc in range(mctimes):
        simuNG.initialization(committedNodes=commedNodes, initialN_Aratio=0)        
        n_A, n_B, n_AB = simuNG.namingGameSimulation(beta, tmax, agreementRule, check_every, sampleNum, T)
        nameRatios_allmc.append((n_A, n_B, n_AB))  
    nameRatios_dic = [f'{agreementRule}_n{net}_p{parasClass}_e{exprClass}_betaID{idb}_pID{idp}_mtd{mtd}_range', nameRatios_allmc] 
    
    
    return nameRatios_dic

if __name__ == '__main__':
    main()
