**README**

This folder contains code and results for computing node measures in higher-order networks, including:

**Folder:**
- `NodesMeasures`: Computational results

**Files:**
- `cal_nodesmeasures_h_utils.py`: Implementation code for various node measures
- `cal_nodesmeasures_h_run.py`: Computes measures, and outputs are save in `NodesMeasures`.

In `NodesMeasures`, files are named by the measure and the hypergraph:
measures:
2-degree  :  neiNodesNum 
2-betweenness  :  clsBetweenness
2-closeness  :  clsCloseness
2-eigenvector  :  clsEigenvector
2-k-coreness  :  clsKcore
eigenvector-max  :  nodeEdgeEigenvector_max
eigenvector-linear  :  nodeEdgeEigenvector_linear
KMcore_g1  :  hyper-coreness-R 
KMcore_gf  :  hyper-coreness-Rw
hyper-degree  :  degree
hyper-degree-random  :  degree_random
new_neiNodeSum_1  :  $IPS_{1}^{HCP}$
new_neiNodeSum_1_HNCsize()  :  $IPS_1^{HNG}, $IPS_{1}^{HCSA}
tc2_sum_1  :  $IPS_{1}^{HTC} (\theta=0.5)$
tc4_sum_1  :  $IPS_{1}^{HTC} (\theta=0.25)$

hypergraphs:
realNet_1 : congress-bills 
realNet_2 : house-committees 
realNet_3 : algebra-questions 
realNet_4 : geometry-questions 
realNet_5 : music-review 
realNet_6 : senate-bills 
realNet_7 : senate-committees 
realNet_8 : email-Enron 
realNet_9 : email-EU 
realNet_10 : Elem1 
realNet_11 : Mid1 
realNet_12 : InVS15 
realNet_13 : LH10 
realNet_14 : LyonSchool 
realNet_15 : SFHH 
realNet_16 : Thiers13 
realNet_17 : M_PL_015_ins 
realNet_18 : M_PL_015_pl 
realNet_19 : M_PL_062_ins 
realNet_20 : M_PL_062_pl 
realNet_25 : threads-math-sx