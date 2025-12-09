# README

# analysis of HNG
`ana_namingGame.ipynb` utilizes results in `../SimulationResults/NamingGame` (simulations results of the HNG model) to plot Fig.6., where the output plot are saved in `savefig_si_ng`.
Here we only retain the results of the hypergraph congress-bills (numbered 1), under both the unanimity rule and the union rule. 

# analysis of HCP (Fig.2 and Fig.3)
`analysis_hcp_jac_err.ipynb` : Fig.2, outputs are saved in `savefig_all` and `savefig_all/main` (outputs not retained). 

`ana_scatters.ipynb` : Fig.3, outputs are saved in `savefig_all/scatters` (outputs not retained).

`analysis_utils.py` : utils for plotting

`cal_average_rvs.ipynb` : calculate the average propagation of each seed (based on results in `../SimulationResults/NonlinearHC_SIR`), and outputs are saved in `NonlinearHC_SIR_avg`. 

Here we retain the entire results in `/NonlinearHC_SIR_avg`, for the correct running of `analysis_hcp_jac_err.ipynb` and `ana_scatters.ipynb`. 

