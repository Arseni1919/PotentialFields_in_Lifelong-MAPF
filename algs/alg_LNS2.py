from typing import List, Dict
from tools_for_graph import *
from tools_for_heuristics import *
from tools_for_plotting import *
from algs.test_single_alg import test_single_alg
from algs.alg_a_star_space_time import a_star_xyt
from environments.env_LL_MAPF import EnvLifelongMAPF
from algs.alg_ParObs_PF_PrP_seq import ParObsPotentialFieldsPrPAgent, AlgParObsPotentialFieldsPrPSeq


class AlgLNS2Seq(AlgParObsPotentialFieldsPrPSeq):
    """
    Public methods:
    .first_init(env)
    .reset()
    .get_actions(observations)
    """
    def __init__(self, params, alg_name):
        super().__init__(params, alg_name)
        self.big_N = params['big_N']

    def _build_G_c(self):
        num_of_confs = 0
        for agent1, agent2 in combinations(self.agents, 2):
            if not two_plans_have_no_vc(agent1.plan, agent2.plan):
                num_of_confs += 1
        return num_of_confs

    def _build_plans(self):
        """
        LNS2:
        - build PrP
        - Build G_c
        - v <- select a random vertex v from Vc with deg(v) > 0
        - find the connected component that has v
        - If |V_v| ≤ N:
            - put all inside
            - select random agent from V_v and do the random walt until fulfill V_v up to N
        - Otherwise:
            - Do the random walk from v in G_c until you have N agents
        - Solve the neighbourhood V_v with PrP (random priority)
        - Replace old plans with the new plans iff the number of colliding pairs (CP) of the paths in the new plan
          is no larger than that of the old plan
        """
        # init PrP solution
        pass

        while (num_of_confs := self._build_G_c()) > 0:
            V_v, v = self._select_random_v()
            if len(V_v) <= self.big_N:
                self._fill_the_neighbourhood()
            else:
                self._cut_the_neighbourhood()
            self._solve_with_PrP(V_v)
            self.replace_olp_plans()
            print(num_of_confs)


@use_profiler(save_dir='../stats/alg_lns2_seq.pstat')
def main():
    # Alg params
    alg_name = 'LNS2'
    params_dict = {
        'LNS2': {'big_N': 8},
    }

    alg = AlgLNS2Seq(params=params_dict[alg_name], alg_name=alg_name)
    test_single_alg(
        alg,

        # GENERAL
        # random_seed=True,
        random_seed=False,
        seed=123,
        PLOT_PER=1,
        PLOT_RATE=0.001,
        PLOT_FROM=0,
        middle_plot=True,
        # middle_plot=False,
        final_plot=True,
        # final_plot=False,

        # FOR ENV
        iterations=200,
        n_agents=170,
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

