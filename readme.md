# README

We provide the complete code necessary to reproduce the main results of our paper  
**“A unified framework for identifying influential nodes in hypergraphs.”**  
The purpose and functionality of each folder (module) are summarized below.

---

## **Folder structure**

- **Networks**  
  Hypergraph datasets used in the paper.

- **MeasureValuesRanking**  
  Code and results for computing various centrality measures.

- **SimulationCodes_high-order_contagion_gaming**  
  Code and results for evaluating robustness and transferability, including HCP, HCSA, and HTC (corresponding to Figs. 1, 4, and 5).  
  Simulation code for HCP (Figs. 2–3) and HNG (Fig. 6) is also included here; their results are stored in following `SimulationResults`, and further analyses in `AnalysisResults`.

- **SimulationResults**  
  Intermediate simulation outputs used for figures and analyses.

- **AnalysisResults**  
  Results of analysis and figure-generation scripts.

---

## **Environment**

- `hyperG.yml` — Conda environment file for reproducing the experimental environment.

More detailed README files are provided within the corresponding subfolders.

---

## **Notes on intermediate results**

Due to the large volume of experiments, storing all intermediate results would make the repository excessively large.  
We therefore provide the complete codebase and include only a subset of representative intermediate results to facilitate the review process. 
Additional intermediate results can be provided upon request.

# The represents of measures in codes:
2-degree  :  neiNodesNum; 
2-betweenness  :  clsBetweenness; 
2-closeness  :  clsCloseness; 
2-eigenvector  :  clsEigenvector; 
2-k-coreness  :  clsKcore; 
eigenvector-max  :  nodeEdgeEigenvector_max; 
eigenvector-linear  :  nodeEdgeEigenvector_linear; 
KMcore_g1  :  hyper-coreness-R; 
KMcore_gf  :  hyper-coreness-Rw; 
hyper-degree  :  degree; 
hyper-degree-random  :  degree_random; 
new_neiNodeSum_1  :  $IPS_{1}^{HCP}$; 
new_neiNodeSum_1_HNCsize()  :  $IPS_1^{HNG}$, $IPS_{1}^{HCSA}$; 
tc2_sum_1  :  $IPS_{1}^{HTC} (\theta=0.5)$; 
tc4_sum_1  :  $IPS_{1}^{HTC} (\theta=0.25)$; 

# The represents of hypergraphs in codes:
realNet_1 : congress-bills; 
realNet_2 : house-committees; 
realNet_3 : algebra-questions; 
realNet_4 : geometry-questions; 
realNet_5 : music-review; 
realNet_6 : senate-bills; 
realNet_7 : senate-committees; 
realNet_8 : email-Enron; 
realNet_9 : email-EU; 
realNet_10 : Elem1; 
realNet_11 : Mid1; 
realNet_12 : InVS15; 
realNet_13 : LH10; 
realNet_14 : LyonSchool; 
realNet_15 : SFHH; 
realNet_16 : Thiers13; 
realNet_17 : M_PL_015_ins; 
realNet_18 : M_PL_015_pl; 
realNet_19 : M_PL_062_ins; 
realNet_20 : M_PL_062_pl; 

