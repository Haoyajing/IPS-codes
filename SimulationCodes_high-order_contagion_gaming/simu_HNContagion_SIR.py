from cmath import sqrt
from multiprocessing import pool
from numpy import tri
# from choose_initialNodes import*
from Hsimulation_utils import*
from pathos.multiprocessing import ProcessingPool as Pool
import pickle
import time
import tqdm

netType = "realNet"
net = 1
parasClass = 1
exprClass = 2
processNum = 46
#Infection parameters
nu = 1.5
l = 0.00005
mu = 0.1
#experiment parameters
mctimes = 300
tmax = float("Inf")


def main():
    with open(f'../Networks/networks/{netType}_{net}.json', 'r') as fl:
        HyperGraph = json.load(fl)

    temp = []
    for i in HyperGraph:
        temp.extend(i)
    nodeSet = set(temp)
    Hnodes = list(nodeSet)
    Hnodes.sort() 
    
    args = [] 
    for node in Hnodes:
        initialInfectedNodes = [int(node)]
        args.append(initialInfectedNodes)
    
    
    strat_time = time.time()
    pool = Pool(processes=processNum)  
    results = list(tqdm.tqdm(iterable=(pool.imap(run_one_simulation, args)), total=len(args)))
    # results = [([k1,v1],[ke1,va1]), ([k2,v2],[ke2,va2]), ...]
    end_time = time.time()
    print(f'net{net}', 'simulation time = ', (end_time-strat_time)/(60), 'min') 
    range_results = [item[0] for item in results]
    range_results_dict = dict(range_results)
    length_results = [item[1] for item in results]
    length_results_dict = dict(length_results)

    out_dir1 = f'../SimulationResults/NonlinearHC_SIR/'
    filename1 = f'{net}_para{parasClass}_expr{exprClass}_range.json'
    with open(f'{out_dir1+filename1}','w') as f:
        json.dump(range_results_dict,f)

    out_dir2 = f'../SimulationResults/NonlinearHC_SIR/'
    filename2 = f'{net}_para{parasClass}_expr{exprClass}_length.json'
    with open(f'{out_dir2+filename2}','w') as g:
        json.dump(length_results_dict,g)


def run_one_simulation(args):  
    initialInfectedNodes= args
    with open(f'../Networks/networks/{netType}_{net}.json', 'r') as fl:
        HyperGraph = json.load(fl)
    iniNode = initialInfectedNodes[0]

    scSIR = SimuNonlinearContagion_SIR(HyperGraph)

    RNodesNum_allmc = []
    propagationLength_allmc = []

    for mc in range(mctimes):  # Monte Carlo experiment
        scSIR.initialization(initial_infected=initialInfectedNodes)
        RNodesNum, propagationLength, InodesNum, SnodesNum = scSIR.contagionSimulation(l, mu, nu, tmax)
        if RNodesNum+SnodesNum != scSIR.N:
            print(iniNode, 'add error')
        if InodesNum !=0 :
            print(iniNode, '0 error')

        RNodesNum_allmc.append(RNodesNum)  
        propagationLength_allmc.append(propagationLength)  

    infectedNodesNum_dic = [f'n{net}_p{parasClass}_e{exprClass}_ini{iniNode}_range', RNodesNum_allmc]  
    propagationLength_dic = [f'n{net}_p{parasClass}_e{exprClass}_ini{iniNode}_length', propagationLength_allmc]
    
    return infectedNodesNum_dic, propagationLength_dic

if __name__ == '__main__':
    main()
