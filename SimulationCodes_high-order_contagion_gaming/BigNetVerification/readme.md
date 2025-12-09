**README**

This file analysis the performance of $IPS_1^{HCP}$ on the large-size hypergraph threads-math-sx

**Files:**
- `simulation` : simulations of the node influence (generate `SIR_25_robust_range_mu10_add_nu1_1.json`. For lightweight, we only retain the results when mu=1, nu=1, l=0.1 here)
- `big_analysis.ipynb` : plot imprecise functions and Jaccard coefficient (output figures are saved in `savefig_big`)
- `correlation_scatter.ipynb` : plot subgraph in Fig.3 (output figures are saved in `savefig_big`)
- `analysis_utils.py` : utils for plotting

-`iniNodes_25.json` : randomly sampled seed nodes.
remaining json files are intermediate files for analysis.
