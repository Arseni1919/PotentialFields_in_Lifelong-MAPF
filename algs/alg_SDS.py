from typing import List, Dict

import numpy as np

from tools_for_graph import *
from tools_for_heuristics import *
from tools_for_plotting import *
from algs.test_single_alg import test_single_alg
from algs.alg_a_star_space_time import a_star_xyt
from environments.env_LL_MAPF import EnvLifelongMAPF
from algs.alg_ParObs_PF_PrP_seq import ParObsPFPrPAgent, AlgParObsPFPrPSeq


class SDSAgent(ParObsPFPrPAgent):
    def __init__(self, num: int, start_node, next_goal_node, **kwargs):
        super().__init__(num, start_node, next_goal_node, **kwargs)


class AlgSDS(AlgParObsPFPrPSeq):
    """
    Public methods:
    .first_init(env)
    .reset()
    .get_actions(observations)
    """
    def __init__(self, params, alg_name):
        super().__init__(params, alg_name)
        self.big_N = params['big_N']
        self.conf_matrix = None
        self.conf_agents_names_list = None
        self.conf_vv_random_walk = None
        self.conf_neighbourhood = None

    def reset(self):
        self.agents: List[ParObsPFPrPAgent] = []
        for env_agent in self.env.agents:
            new_agent = SDSAgent(
                num=env_agent.num, start_node=env_agent.start_node, next_goal_node=env_agent.next_goal_node,
                nodes=self.nodes, nodes_dict=self.nodes_dict, h_func=self.h_func, h_dict=self.h_dict,
                map_dim=self.map_dim, params=self.params
            )
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent
            self.curr_iteration = 0

    def _build_plans(self):
        # init path
        pass

        # while there are conflicts
        while True:
            # exchange with neighbours
            pass

            # replan
            pass


@use_profiler(save_dir='../stats/alg_sds.pstat')
def main():
    # Alg params
    big_N = 5
    # alg_name = 'LNS2'
    # alg_name = 'PF-LNS2'
    # alg_name = 'ParObs-LNS2'
    alg_name = 'ParObs-PF-LNS2'

    params_dict = {
        'SDS': {},
        'PF-SDS': {
            # For PF
            'pf_weight': 0.5,
            'pf_size': 3,
            'pf_shape': 2,
        },
        'ParObs-SDS': {
            # For RHCR
            'h': 5,  # my step
            'w': 5,  # my planning
        },
        'ParObs-PF-SDS': {
            # For PF
            'pf_weight': 0.5,
            'pf_size': 3,
            'pf_shape': 2,
            # For RHCR
            'h': 5,  # my step
            'w': 5,  # my planning
        },
    }

    alg = AlgSDS(params=params_dict[alg_name], alg_name=alg_name)
    test_single_alg(
        alg,

        # GENERAL
        # random_seed=True,
        random_seed=False,
        seed=123,
        PLOT_PER=1,
        # PLOT_PER=20,
        PLOT_RATE=0.001,
        PLOT_FROM=50,
        middle_plot=True,
        # middle_plot=False,
        final_plot=True,
        # final_plot=False,

        # FOR ENV
        iterations=200,  # !!!
        # iterations=100,
        n_agents=250,
        n_problems=1,

        # Map
        img_dir='empty-32-32.map',  # 32-32
        # img_dir='random-32-32-10.map',  # 32-32          | LNS | Up to 400 agents with w=5, h=2, lim=1min.
        # img_dir='random-32-32-20.map',  # 32-32
        # img_dir='room-32-32-4.map',  # 32-32
        # img_dir='maze-32-32-2.map',  # 32-32
    )


if __name__ == '__main__':
    main()

