# README

# Results of HNG
Results in `NamingGame` is the output of `../SimulationCodes_high-order_contagion_gaming/simu_naming_game.py` (simulations of the HNG model).
Here we only retain the results of the hypergraph congress-bills (numbered 1), under both the unanimity rule and the union rule. Then, the Fig.6 can be drawn by  `../AnalysisResults/ana_namingGame.ipynb`.

# A part Results of HCP
This file keeps the output of `../SimulationCodes_high-order_contagion_gaming/simu_HNContagion_SIR.py` (simulations of the HCP model, corresponding to Fig.2 and Fig.3)
By running `../AnalysisResults/cal_average_rvs.ipynb`, one can calculate the average propagation of these output, i.e., the ground-truth of node influence under HCP (saved in `../AnalysisResults/NonlinearHC_SIR_avg`). Then, the Fig.2 can be drawn by  `../AnalysisResults/analysis_hcp_jac_err.ipynb`, and Fig.3 can be drawn by  `../AnalysisResults/ana_scatters.ipynb`.

Here we only retain the entire results in `../AnalysisResults/NonlinearHC_SIR_avg`, for the correct running of finally plotting codes. 
One can run codes in above procedure to generate these intermediate results.
