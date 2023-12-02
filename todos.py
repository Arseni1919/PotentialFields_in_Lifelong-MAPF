"""
##########################################
##########################################
DONE:

- save heuristics
- create LL-MAPF env
- implement LL-PrP
- implement PF-LL-PrP

- create ParObs-LL-MAPF
- implement ParObs-LL-PrP
- implement ParObs-PF-LL-PrP

- understand the paper of  LNS2
- h,w for alg, not for env
- unite 3 PrP files

- For PrP:
    - instead to use restart -> use persist
    - if finished, wait to the next k-step iteration
    - for those who did not have a path -> use IStay
- correct the PrP and PF-PrP
- implement LNS2 with Collision-Based Neighborhoods (CBN)
- improve PF (agent creates all PF)
- execute big experiments
    - set the limit for calculations
    - examine different PF variations
- save the data
- improve _implement_istay()
- implement single-shot MAPF in the simulator
- BUG: additional step
- implement SDS

##########################################
##########################################
TODO:

- write a paper
- submit to a conference

- maybe implement the full LNS2
- examine parameters on different maps
- maybe implement learning for the PF parameters with RL

##########################################
##########################################
"""