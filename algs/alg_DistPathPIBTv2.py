import random
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
        self.a_names_in_conf_list = []
        self.a_names_in_conf_list_prev = []
        self.lower_agents_processed = []
        self.higher_agents_processed_prev = []
        self.order = None
        self.team_leader = None
        self.team_queue = []
        self.has_conf = True

    def set_order(self, order):
        self.order = order

    def secondary_init(self):
        self.a_names_in_conf_list = []
        self.lower_agents_processed = []
        self.plan = None
        self.plan_succeeded = True

    def dppibt_build_plan(self, h_agents=None, goal=None):
        self.plan = None
        self.build_plan(h_agents, goal=goal)


class AlgSDS(AlgParObsPFPrPSeq):
    """
    Public methods:
    .first_init(env)
    .reset()
    .get_actions(observations)
    """
    def __init__(self, params, alg_name):
        super().__init__(params, alg_name)
        self.agents_names = []
        self.u_leader = None

    def reset(self):
        self.agents: List[SDSAgent] = []
        for env_agent in self.env.agents:
            new_agent = SDSAgent(
                num=env_agent.num, start_node=env_agent.start_node, next_goal_node=env_agent.next_goal_node,
                nodes=self.nodes, nodes_dict=self.nodes_dict, h_func=self.h_func, h_dict=self.h_dict,
                map_dim=self.map_dim, params=self.params
            )
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent
            self.curr_iteration = 0
        self.i_agent = self.agents_dict['agent_0']

    def _build_plans(self):
        # one_master = [agent for agent in self.agents if agent.master == agent.master_init][0]
        if self.h and self.curr_iteration % self.h != 0 and self.curr_iteration != 0:
            return self._get_info_to_send()
        distr_time = 0

        # set the masters + build a global order
        self._all_shuffle(to_shuffle=False)
        # self._all_shuffle(to_shuffle=True)
        self._all_set_order()

        # set the team leaders
        time_limit_crossed, distr_time = self._all_secondary_init(distr_time)
        if time_limit_crossed:
            return self._get_info_to_send()

        # build initial leader's path
        self.u_leader.dppibt_build_plan()

        # find free locations
        to_continue, infected_agents, free_nodes_to_fill = self._find_free_spots()

        if to_continue:
            self._plan_ways_aside(infected_agents, free_nodes_to_fill)

        # find collisions
        there_are_collisions, distr_time = self._all_find_conf_agents(distr_time)
        assert not there_are_collisions

        return self._get_info_to_send()

    def _get_info_to_send(self):
        orders_dict = {}
        if self.curr_iteration != 0:
            orders_dict = {agent.name: agent.order for agent in self.agents}
            leaders_orders = list(set(orders_dict.values()))
            leaders_orders.sort()
            orders_dict = {k: leaders_orders.index(v) for k, v in orders_dict.items()}
        info_to_send = {'orders_dict': orders_dict, 'one_master': self.i_agent}
        return info_to_send

    def _all_shuffle(self, to_shuffle=True):
        unsucc_list = [a for a in self.agents if a.curr_node.xy_name != a.next_goal_node.xy_name]
        succ_list = [a for a in self.agents if a.curr_node.xy_name == a.next_goal_node.xy_name]
        if to_shuffle:
            random.shuffle(unsucc_list)
            random.shuffle(succ_list)
        unsucc_list.extend(succ_list)
        # if len(unsucc_list) > 0 and self.i_agent not in unsucc_list:
        #     self.i_agent = unsucc_list[0]
        self.agents = unsucc_list
        self.u_leader = self.agents[0]
        self.i_agent = self.u_leader
        self.agents_names = [a.name for a in self.agents]

    def _all_set_order(self):
        for i, agent in enumerate(self.agents):
            agent.set_order(i)
        self.u_leader.set_order(0)
        pos_to_agent_dict = {a.curr_node.xy_name: a for a in self.agents[1:]}
        i = 1
        nei_nodes_dict = {}
        open_list = [self.u_leader.curr_node]
        while len(open_list) > 0:
            i_node = open_list.pop()
            nei_nodes_dict[i_node.xy_name] = i_node
            neighbours = i_node.neighbours
            random.shuffle(neighbours)
            for node_nei_name in neighbours:
                if node_nei_name not in nei_nodes_dict:
                    node_nei = self.nodes_dict[node_nei_name]
                    open_list.append(node_nei)
                    if node_nei_name in pos_to_agent_dict:
                        nei_agent = pos_to_agent_dict[node_nei_name]
                        nei_agent.set_order(i)
                        i += 1
        self.agents.sort(key=lambda x: x.order)

    def _all_secondary_init(self, distr_time):
        parallel_times = [0]
        for i, agent in enumerate(self.agents):

            local_start_time = time.time()

            agent.secondary_init()

            # limit check
            passed_time = time.time() - local_start_time
            parallel_times.append(passed_time)
            if distr_time + passed_time > self.time_to_think_limit:
                self._cut_up_to_the_limit(i)
                return True, distr_time + max(parallel_times)

        return False, distr_time + max(parallel_times)

    def _find_free_spots(self):
        node_name_to_agent_dict = {a.curr_node.xy_name: a for a in self.agents[1:]}
        node_name_of_agents = list(node_name_to_agent_dict.keys())
        node_names_of_plan = [n.xy_name for n in self.u_leader.plan]
        assert self.u_leader.curr_node.xy_name not in node_name_to_agent_dict
        conflicts = np.intersect1d(node_name_of_agents, node_names_of_plan)
        n_of_conflicts = len(conflicts)

        if n_of_conflicts == 0:
            # all stay on place
            for a in self.agents[1:]:
                a.dppibt_build_plan(goal=a.curr_node)
            return False, None, None

        closed_node_names = [self.u_leader.curr_node.xy_name]
        nodes_to_open = self.u_leader.plan[:]
        closed_node_names.extend(node_names_of_plan)
        free_nodes_to_fill = []
        infected_agents = [node_name_to_agent_dict[c] for c in conflicts]
        while len(nodes_to_open) > 0:
            next_node = nodes_to_open.pop(0)
            closed_node_names.append(next_node.xy_name)

            for nei_name in next_node.neighbours:
                if nei_name in closed_node_names:
                    continue
                curr_node = self.nodes_dict[nei_name]
                if nei_name in node_name_to_agent_dict:
                    nei_agent = node_name_to_agent_dict[nei_name]
                    if nei_agent not in infected_agents:
                        infected_agents.append(nei_agent)
                    # n_of_conflicts += 1
                else:
                    free_nodes_to_fill.append(curr_node)
                    if len(free_nodes_to_fill) == n_of_conflicts:
                        return True, infected_agents, free_nodes_to_fill
                nodes_to_open.append(curr_node)

        # all stay on place
        for a in self.agents:
            a.dppibt_build_plan(goal=a.curr_node)
        return False, None, None

    def _plan_ways_aside(self, infected_agents, free_nodes_to_fill):
        # h_func(from_node, to_node)
        h_agents = []
        for a in self.agents[1:]:
            if a not in infected_agents:
                a.dppibt_build_plan(goal=a.curr_node)
                h_agents.append(a)
        node_names_of_plan = list(set([n.xy_name for n in self.u_leader.plan]))
        while len(free_nodes_to_fill) > 0:
            free_nodes_to_fill.sort(key=lambda n: self.h_func(self.u_leader.curr_node, n), reverse=True)
            next_node = free_nodes_to_fill.pop(0)
            infected_agents.sort(key=lambda a: self.h_func(a.curr_node, next_node))
            nearest_agent = infected_agents.pop(0)
            if nearest_agent.curr_node.xy_name not in node_names_of_plan:
                free_nodes_to_fill.append(nearest_agent.curr_node)
            nearest_agent.dppibt_build_plan(h_agents=h_agents, goal=next_node)
            assert nearest_agent.plan_succeeded
            h_agents.append(nearest_agent)
        self.u_leader.dppibt_build_plan(h_agents=h_agents)

    def _all_find_conf_agents(self, distr_time):
        there_are_collisions, n_collisions = False, 0
        col_str = ''
        for agent1, agent2 in combinations(self.agents, 2):
            if agent1.name not in agent2.nei_dict:
                continue
            plan1 = agent1.get_full_plan()
            plan2 = agent2.get_full_plan()
            have_confs, conf_index = two_plans_have_confs_at(plan1, plan2)
            if have_confs:
                there_are_collisions = True
                agent1.a_names_in_conf_list.append((agent2.name, conf_index))
                agent2.a_names_in_conf_list.append((agent1.name, conf_index))
                n_collisions += 1
                col_str = f'{agent1.name} <-> {agent2.name}'
        print(f'\r>>>> {self.curr_iteration=}, {n_collisions} collisions, last one: {col_str}', end='')
        return there_are_collisions, distr_time


@use_profiler(save_dir='../stats/alg_dppibt_v2.pstat')
def main():
    # Alg params
    # mem_weight = 1
    mem_weight = 2
    h = 5
    w = h
    # alg_name = 'SDS'
    # alg_name = 'PF-SDS'
    alg_name = 'ParObs-SDS'
    # alg_name = 'ParObs-Memory-SDS'

    params_dict = {
        'SDS': {},
        'PF-SDS': {'pf_weight': mem_weight},
        'ParObs-SDS': {'h': h, 'w': w},
        'ParObs-Memory-SDS': {'h': h, 'w': w, 'mem_weight': mem_weight},
    }

    alg = AlgSDS(params=params_dict[alg_name], alg_name=alg_name)
    test_single_alg(
        alg,

        # GENERAL
        # random_seed=True,
        random_seed=False,
        seed=321,
        PLOT_PER=1,
        # PLOT_PER=20,
        PLOT_RATE=0.001,
        # PLOT_RATE=0.5,
        PLOT_FROM=1,
        middle_plot=True,
        # middle_plot=False,
        final_plot=True,
        # final_plot=False,

        # FOR ENV
        # iterations=200,  # !!!
        iterations=100,
        # iterations=50,
        n_agents=300,
        n_problems=1,
        classical_rhcr_mapf=True,
        # classical_rhcr_mapf=False,
        time_to_think_limit=100000,  # seconds
        rhcr_mapf_limit=10000,
        global_time_limit=6000,  # seconds

        # Map
        # img_dir='empty-32-32.map',  # 32-32
        # img_dir='random-32-32-10.map',  # 32-32          | LNS | Up to 400 agents with w=5, h=2, lim=1min.
        # img_dir='random-32-32-20.map',  # 32-32
        # img_dir='room-32-32-4.map',  # 32-32
        # img_dir='maze-32-32-2.map',  # 32-32

        # img_dir='tree.map', predefined_nodes=True, scen_name='tree',  # yes
        # img_dir='corners.map', predefined_nodes=True, scen_name='corners',  # yes
        img_dir='tunnel.map', predefined_nodes=True, scen_name='tunnel',  # yes (slow)
        # img_dir='string.map', predefined_nodes=True, scen_name='string',  # yes
        # img_dir='loop_chain.map', predefined_nodes=True, scen_name='loop_chain',  # no
        # img_dir='connector.map', predefined_nodes=True, scen_name='connector',  # yes
        # img_dir='10_10_my_rand.map',  # 32-32
        # img_dir='random-64-64-20.map',  # 64-64
        # img_dir='warehouse-10-20-10-2-1.map',  # 32-32
    )


if __name__ == '__main__':
    main()