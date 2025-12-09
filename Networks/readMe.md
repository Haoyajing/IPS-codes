`remove_duplication.ipynb` processes the data in `RawData` to obtain the real-world hypergraphs in `networks`, which are used for subsequent method performance evaluation.
`threads-math-sx` provides the raw data of the large-size hypergraph threads-math-sx.
`RawData` comes from the reference [1], and `threads-math-sx` comes from the reference [2].
Based on RawData, we remove duplicates in hyperedges for each hypergraph. Specially, in realNet_2 the largest edge contains duplicate nodes, and we deleted the duplicated nodes; realNet_8 has 0-order hyperedges, and we deleted these edges;
realNet_3, 5, and 17 each have two branches, and the small branch are deleted.

The correspondence among the indices of the higher-order networks in `networks`, the data names, and the entries in `RawData` is as follows:
realNet_1 : congress-bills : bills_comm_online/congress-bills_simplices.json
realNet_2 : house-committees : bills_comm_online/house-committees_simplices.json
realNet_3 : algebra-questions :  bills_comm_online/hyperedges-cat-edge-algebra-questions_simplices.json
realNet_4 : geometry-questions : bills_comm_online/hyperedges-cat-edge-geometry-questions_simplices.json
realNet_5 : music-review : bills_comm_online/hyperedges-cat-edge-music-blues-reviews.json
realNet_6 : senate-bills : bills_comm_online/senate-bills_simplices.json
realNet_7 : senate-committees : bills_comm_online/senate-committees_simplices.json
realNet_8 : email-Enron : email-Enron/email-Enron_simplices.json
realNet_9 : email-EU : email-EU/email-Eu_simplices.json
realNet_10 : Elem1 : UtahSchools/aggr_15min_cliques_thr1_Elem1.json
realNet_11 : Mid1 : UtahSchools/aggr_15min_cliques_thr1_Mid1.json
realNet_12 : InVS15 : SP/aggr_15min_cliques_thr1_InVS15.json
realNet_13 : LH10 : SP/aggr_15min_cliques_thr1_LH10.json
realNet_14 : LyonSchool : SP/aggr_15min_cliques_thr1_LyonSchool.json
realNet_15 : SFHH : SP/aggr_15min_cliques_thr1_SFHH.json
realNet_16 : Thiers13 : SP/aggr_15min_cliques_thr1_Thiers13.json
realNet_17 : M_PL_015_ins : ECO/M_PL_015_ECO_ins.json
realNet_18 : M_PL_015_pl : ECO/M_PL_015_ECO_pl.json
realNet_19 : M_PL_062_ins : ECO/M_PL_062_ECO_ins.json
realNet_20 : M_PL_062_pl : ECO/M_PL_062_ECO_pl.json

`threads-math-sx` is referred to as realNet_25.

References:
[1] Mancastroppa M, Iacopini I, Petri G, et al. Hyper-cores promote localization and efficient seeding in higher-order processes[J]. Nature communications, 2023, 14(1): 6223.
[2] Benson A R, Abebe R, Schaub M T, et al. Simplicial closure and higher-order link prediction[J]. Proceedings of the National Academy of Sciences, 2018, 115(48): E11221-E11230.


