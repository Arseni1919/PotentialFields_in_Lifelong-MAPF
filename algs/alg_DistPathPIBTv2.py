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
        self.a_names_in_conf_list = []
        self.lower_agents_processed = []
        self.plan = None
        self.plan_succeeded = True

    def dppibt_build_plan(self, h_agents=None, goal=None, nodes=None, nodes_dict=None):
        self.plan = None
        self.build_plan(h_agents, goal=goal, nodes=nodes, nodes_dict=nodes_dict)


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

        # set the masters + build a global order
        self._all_shuffle(to_shuffle=False)
        # self._all_shuffle(to_shuffle=True)
        self._all_set_order()

        # build initial leader's path
        self.u_leader.dppibt_build_plan()

        # find free locations
        to_continue, infected_agents, free_nodes_to_fill, directed_graph = self._find_free_spots()

        # if there are agents on the track and the plan is possible
        if to_continue:
            self._plan_ways_aside(infected_agents, free_nodes_to_fill, directed_graph)

        # find collisions
        there_are_collisions = self._all_find_conf_agents()
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

    def _find_free_spots(self):
        rest_of_agents = self.agents[1:]
        node_name_to_agent_dict = {a.curr_node.xy_name: a for a in rest_of_agents}
        node_name_of_agents = list(node_name_to_agent_dict.keys())
        assert self.u_leader.curr_node.xy_name not in node_name_to_agent_dict
        node_names_of_plan = list(set([n.xy_name for n in self.u_leader.plan]))
        conflicts = np.intersect1d(node_name_of_agents, node_names_of_plan)
        n_of_conflicts = len(conflicts)

        # there are no agents on the path
        if n_of_conflicts == 0:
            # all stay on place
            for a in rest_of_agents:
                a.set_istay()
            return False, None, None, None

        # there are agents on the path
        closed_node_names = [self.u_leader.curr_node.xy_name]
        nodes_to_open = list(set(self.u_leader.plan[:]))
        closed_node_names.extend(node_names_of_plan)
        free_nodes_to_fill = []
        infected_agents = [node_name_to_agent_dict[c] for c in conflicts]
        directed_graph = []
        while len(nodes_to_open) > 0:
            next_node = nodes_to_open.pop(0)
            open_names = [n.xy_name for n in nodes_to_open]
            if next_node.xy_name not in closed_node_names:
                closed_node_names.append(next_node.xy_name)

            for nei_name in next_node.neighbours:
                if nei_name in open_names:
                    continue
                if nei_name in closed_node_names:
                    continue
                curr_node = self.nodes_dict[nei_name]
                directed_graph.append((next_node.xy_name, curr_node.xy_name))
                if nei_name in node_name_to_agent_dict:
                    # infect this agent
                    nei_agent = node_name_to_agent_dict[nei_name]
                    if nei_agent not in infected_agents:
                        infected_agents.append(nei_agent)
                    # n_of_conflicts += 1
                else:
                    # we found a free spot
                    free_nodes_to_fill.append(curr_node)
                    if len(free_nodes_to_fill) == n_of_conflicts:
                        for a in rest_of_agents:
                            if a not in infected_agents:
                                a.set_istay()
                        return True, infected_agents, free_nodes_to_fill, directed_graph
                nodes_to_open.append(curr_node)
                open_names = [n.xy_name for n in nodes_to_open]

        # all stay on place, because we found not enough free spots
        for a in self.agents:
            a.set_istay()
        return False, None, None, None

    def _plan_ways_aside(self, infected_agents, free_nodes_to_fill, directed_graph):
        # how to use h_func: h_func(from_node, to_node)

        # prep
        node_names_of_plan = list(set([n.xy_name for n in self.u_leader.plan]))
        for a in infected_agents:
            a.plan = [a.curr_node]
            if a.curr_node.xy_name not in node_names_of_plan:
                free_nodes_to_fill.append(a.curr_node)
        all_graph_names = [x[0] for x in directed_graph]
        all_graph_names.extend([x[1] for x in directed_graph])
        all_graph_names.extend(node_names_of_plan)
        all_graph_names = list(set(all_graph_names))

        step_count = 0
        while step_count < self.h - 1:
            prev_config = [a.plan[-1].xy_name for a in infected_agents]
            next_config = []
            assert len(set(prev_config)) == len(prev_config)
            for i_a in infected_agents:
                last_node = i_a.plan[-1]
                last_node_name = i_a.plan[-1].xy_name
                # on the path
                if last_node_name in node_names_of_plan:

                    next_to_move = [n for n in last_node.neighbours if n not in prev_config]
                    next_to_move = [n for n in next_to_move if n in all_graph_names]
                    # surrounded by agents
                    if len(next_to_move) == 0:
                        i_a.plan.append(last_node)
                        next_config.append(last_node_name)
                        continue

                    to_nodes_tuples = list(filter(lambda x: x[0] in node_names_of_plan, directed_graph))
                    to_nodes_names = [x[1] for x in to_nodes_tuples]
                    to_nodes_names_free = [x for x in to_nodes_names if x not in prev_config]
                    # no free nodes near the path available
                    if len(to_nodes_names_free) == 0:
                        i_a.plan.append(last_node)
                        next_config.append(last_node_name)
                        continue

                    to_nodes_names_free.sort(key=lambda x: self.h_func(self.nodes_dict[x], last_node))
                    nearest_out_free = to_nodes_names_free[0]

                    # next_to_move = [n for n in last_node.neighbours if n in all_graph_names]
                    # next_to_move.remove(last_node_name)
                    next_to_move.sort(key=lambda x: self.h_func(self.nodes_dict[x], self.nodes_dict[nearest_out_free]))
                    next_node_name = next_to_move[0]
                    if next_node_name not in next_config:
                        i_a.plan.append(self.nodes_dict[next_node_name])
                        next_config.append(next_node_name)
                        continue
                    else:
                        i_a.plan.append(last_node)
                        next_config.append(last_node_name)
                        continue
                # out of the path
                else:
                    to_nodes_tuples = list(filter(lambda x: x[0] == last_node_name, directed_graph))
                    to_nodes_names = [x[1] for x in to_nodes_tuples]
                    next_node_name = last_node_name
                    for to_node_name in to_nodes_names:
                        if to_node_name in prev_config:
                            continue
                        next_node_name = to_node_name
                        break
                    if next_node_name not in next_config:
                        i_a.plan.append(self.nodes_dict[next_node_name])
                        next_config.append(next_node_name)
                        continue
                    else:
                        i_a.plan.append(last_node)
                        next_config.append(last_node_name)
                        continue

            # all arrived out of the path
            if all([a.plan[-1].xy_name not in node_names_of_plan for a in infected_agents]):
                break
            step_count += 1

        # plan for leader
        for a in infected_agents:
            a.plan = a.plan[1:]
            a.fulfill_the_plan()
        rest_of_agents = self.agents[1:]
        self.u_leader.dppibt_build_plan(h_agents=rest_of_agents)

    def _all_find_conf_agents(self):
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
        return there_are_collisions


@use_profiler(save_dir='../stats/alg_dppibt_v2.pstat')
def main():
    # Alg params
    # mem_weight = 1
    mem_weight = 2
    h = 10
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
        n_agents=600,
        n_problems=1,
        classical_rhcr_mapf=True,
        # classical_rhcr_mapf=False,
        time_to_think_limit=100000,  # seconds
        rhcr_mapf_limit=10000,
        global_time_limit=6000,  # seconds

        # Map
        # img_dir='empty-32-32.map',  # 32-32
        img_dir='random-32-32-10.map',  # 32-32          | LNS | Up to 400 agents with w=5, h=2, lim=1min.
        # img_dir='random-32-32-20.map',  # 32-32
        # img_dir='room-32-32-4.map',  # 32-32
        # img_dir='maze-32-32-2.map',  # 32-32

        # img_dir='tree.map', predefined_nodes=True, scen_name='tree',  # yes
        # img_dir='corners.map', predefined_nodes=True, scen_name='corners',  # yes
        # img_dir='tunnel.map', predefined_nodes=True, scen_name='tunnel',  # yes (slow)
        # img_dir='string.map', predefined_nodes=True, scen_name='string',  # yes
        # img_dir='loop_chain.map', predefined_nodes=True, scen_name='loop_chain',  # no
        # img_dir='connector.map', predefined_nodes=True, scen_name='connector',  # yes
        # img_dir='10_10_my_rand.map',  # 32-32
        # img_dir='random-64-64-20.map',  # 64-64
        # img_dir='warehouse-10-20-10-2-1.map',  # 32-32
    )


if __name__ == '__main__':
    main()