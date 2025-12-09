from cal_nodesmeasures_h_utils import*
import json
from pathos.multiprocessing import ProcessingPool as Pool
import tqdm

dataType = "realNet" 

def main():
    mu1 = 0.1
    nets_paras_1 = {
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
    mu5 = 0.5
    nets_paras_5 = {
        1:{'net':1, 'l':[ 2e-4,  7e-5,  3e-5,  2e-5]},
        2:{'net':2, 'l':[ 2e-3,  6e-4,  3e-4,  2e-4]},
        3:{'net':3, 'l':[ 4e-3,  1e-3,  8e-4,  5e-4]},
        4:{'net':4, 'l':[ 1e-3,  4e-4,  2e-4,  1e-4]},
        5:{'net':5, 'l':[ 4e-3,  9e-4,  6e-4,  4e-4]},
        6:{'net':6, 'l':[ 5e-5,  2e-5,  1e-5,  8e-6]},
        7:{'net':7, 'l':[ 2e-3,  9e-4,  5e-4,  4e-4]},
        8:{'net':8, 'l':[ 7e-3,  4e-3,  3e-3,  2e-3]},
        9:{'net':9, 'l':[ 4e-3,  1e-3,  7e-4,  4e-4]},
        10:{'net':10, 'l':[ 7e-4,  5e-4,  4e-4,  2e-4]},
        11:{'net':11, 'l':[ 6e-4,  5e-4,  4e-4,  2e-4]},
        12:{'net':12, 'l':[ 9e-3,  1e-2,  1e-2,  7e-3]},
        13:{'net':13, 'l':[ 8e-3,  5e-3,  4e-3,  3e-3]},
        14:{'net':14, 'l':[ 1e-3,  1e-3,  9e-4,  6e-4]},
        15:{'net':15, 'l':[ 9e-3,  9e-3,  8e-3,  6e-3]},
        16:{'net':16, 'l':[ 8e-3,  8e-3,  6e-3,  5e-3]},
        17:{'net':17, 'l':[ 6e-3,  2e-3,  8e-4,  5e-4]},
        18:{'net':18, 'l':[ 3e-3,  1e-3,  7e-4,  6e-4]},
        19:{'net':19, 'l':[ 2e-3,  5e-4,  3e-4,  2e-4]},
        20:{'net':20, 'l':[ 5e-4,  3e-4,  2e-4,  1e-4]},
    }
    mu10 = 1
    nets_paras_10 = {
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
    args = []
    print('right')
    nets = [i for i in range(1,21)]
    # args.append([nets[0], 1, 10, nets_paras_1[nets[0]]['l'][1-1], mu10])
    """
    for net in nets:
        for nu in [1,2,3,4]:
            lThres = nets_paras_1[net]['l'][nu-1]
            for lRatio in [0.1, 1, 10]:
                args.append([net, nu, lRatio, lThres, mu1])
    """
    for net in nets:
        for nu in [1,2,3,4]:
            lThres = nets_paras_5[net]['l'][nu-1]
            for lRatio in [0.1, 1, 10]:
                args.append([net, nu, lRatio, lThres, mu5])

    """
    for net in nets: # # # 
        for nu in [1,2,3,4]:
            lThres = nets_paras_10[net]['l'][nu-1]
            for lRatio in [0.1, 1, 10]:
                args.append([net, nu, lRatio, lThres, mu10])
    """
    pool = Pool(processes=46)
    results = list(tqdm.tqdm(iterable=(pool.imap(calOneNet, args)), total=len(args)))

def calOneNet(args):
    nt, nu, lRatio, lThres, mu = args
    with open(f"../../Networks/networks/{dataType}_{nt}.json","r") as f:
        network = json.load(f)

    calTool = CalNodesMeasures(network, f'{dataType}_{nt}')
    lName = int(lRatio*10)
    muName = int(mu*10)
    #   IPS2
    """"""
    if mu==1:
        calTool.h_new_precise_prob_t2_iNum(parasClass=f'n{nt}_nu{nu}_l{lName}_mu{muName}',nu=nu ,l=lRatio*lThres, mu=mu)
    else:
        calTool.h_new_precise_prob_t2_iNum(parasClass=f'n{nt}_nu{nu}_l{lName}_mu{muName}',nu=nu ,l=lRatio*lThres, mu=mu)        
        calTool.h_new_precise_prob_t2_iNum(parasClass=f'n{nt}_nu{nu}_l{lName}_mu{muName}_rescale',nu=nu ,l=lRatio*lThres/mu, mu=1)
    
    # IPS2r
    # if mu==1:
    #     calTool.h_new_new_new_rough_t2_iNum(parasClass=f'n{nt}_nu{nu}_l{lName}_mu{muName}',nu=nu ,l=lRatio*lThres, mu=mu)
    # else:
    #     calTool.h_new_new_new_rough_t2_iNum(parasClass=f'n{nt}_nu{nu}_l{lName}_mu{muName}',nu=nu ,l=lRatio*lThres, mu=mu)        
    #     calTool.h_new_new_new_rough_t2_iNum(parasClass=f'n{nt}_nu{nu}_l{lName}_mu{muName}_rescale',nu=nu ,l=lRatio*lThres/mu, mu=1)    
   

if __name__ == '__main__':
    main()