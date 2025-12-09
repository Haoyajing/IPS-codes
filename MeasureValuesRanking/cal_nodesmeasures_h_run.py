from cal_nodesmeasures_h_utils import*
import json
from pathos.multiprocessing import ProcessingPool as Pool
import tqdm

dataType = "realNet"
def main():
    nets = [i+1 for i in range (20)] # [i+1 for i in range (20)]

    pool = Pool(processes=20)
    results = list(tqdm.tqdm(iterable=(pool.imap(calOneNet, nets)), total=len(nets)))

def calOneNet(nt):
    with open(f"../Networks/networks/{dataType}_{nt}.json","r") as f:
        network = json.load(f)

    calTool = CalNodesMeasures(network, f'{dataType}_{nt}')
    calTool.h_neiNodesNum()   # 2-degree
    calTool.h_clsBetweenness()   # 2-betweenness
    calTool.h_clsCloseness()   # 2-closeness
    calTool.h_clsEigenvector()   # 2-eigenvector
    calTool.h_clsKcore()   # 2-k-coreness
    calTool.h_nodeEdgeEigenvector(funcGrp='max')   # eigenvector-max
    calTool.h_nodeEdgeEigenvector(funcGrp='linear')   # eigenvector-linear
    calTool.h_KMcore()  # hyper-coreness-R and hyper-coreness-Rw
    calTool.h_degree()  # hyper-degree
    calTool.h_degree_random()   # hyper-degree-random

    calTool.h_new_neiNodeSum_1_HNCsize() # $IPS_1^{HNG}, $IPS_{1}^{HCSA}
    calTool.h_new_neiNodeSum_1()  # $IPS_{1}^{HCP}$
    calTool.h_tc2_sum_1()   # $IPS_{1}^{HTC} (\theta=0.5)$
    calTool.h_tc4_sum_1()   # $IPS_{1}^{HTC} (\theta=0.25)

    

if __name__ == '__main__':
    main()