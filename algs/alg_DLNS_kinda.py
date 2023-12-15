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
        self.was_already_idle = []
        self.done_for_now = False

    def secondary_sds_init_plan(self, nums_order_list):
        self.was_already_idle = []
        self.done_for_now = False
        if self.pf_weight == 0:
            return self.plan

        if check_stay_at_same_node(self.plan, self.next_goal_node):
            return self.plan

        v_constr_dict, e_constr_dict, perm_constr_dict, xyt_problem = build_constraints(self.nodes, {})
        h_agents = []
        for nei in self.nei_list:
            # if check_stay_at_goal(self.plan, self.next_goal_node):
            #     h_agents.append(nei)
            #     continue
            # h_agents.append(nei)
            if nums_order_list.index(nei.num) < nums_order_list.index(self.num):
                h_agents.append(nei)
        nei_pfs, max_plan_len = self._build_nei_pfs(h_agents)

        self.execute_a_star(v_constr_dict, e_constr_dict, perm_constr_dict, xyt_problem, nei_pfs)
        return self.plan

    def replan(self, nums_order_list, inner_iter):
        h_agents, to_continue = self._replan_get_h_agents(nums_order_list=nums_order_list, inner_iter=inner_iter)
        if to_continue:
            # old_plan = self.plan
            self.plan = None
            self.build_plan(h_agents)
            self.done_for_now = True

    def _replan_get_h_agents(self, nums_order_list, inner_iter):
        # h_agents = [self.nei_dict[conf_name] for conf_name in self.a_names_in_conf_list]
        if len(self.a_names_in_conf_list) == 0:
            return [], False

        self.done_for_now = False
        for conf_nei_name in self.a_names_in_conf_list:
            conf_nei = self.nei_dict[conf_nei_name]
            if nums_order_list.index(conf_nei.num) < nums_order_list.index(self.num):
                if not conf_nei.done_for_now:
                    return [], False

        if not self.plan_succeeded:
            self.done_for_now = True
            return [], False

        h_agents = []
        for nei in self.nei_list:

            if not self.nei_succ_dict[nei.name]:
                h_agents.append(nei)
                continue
            # if check_stay_at_goal(self.nei_plans_dict[nei.name], nei.next_goal_node):
            #     # self.was_already_idle.append(nei.name)
            #     h_agents.append(nei)
            #     continue
            if nums_order_list.index(nei.num) < nums_order_list.index(self.num):
                h_agents.append(nei)
                continue
            # if nei.name not in self.a_names_in_conf_list:
            #     h_agents.append(nei)
            #     continue

        return h_agents, True

    # # POTENTIAL FIELDS ****************************** pf_weight ******************************
    def _build_nei_pfs(self, h_agents):
        if self.pf_weight == 0:
            self.nei_pfs = None
            return None, None
        if len(h_agents) == 0:
            return None, None
        max_plan_len = max([len(self.nei_plans_dict[agent.name]) for agent in h_agents])
        nei_pfs = np.zeros((self.map_dim[0], self.map_dim[1], max_plan_len))  # x, y, t
        for nei_agent in h_agents:

            nei_plan = self.nei_plans_dict[nei_agent.name]
            nei_heuristic_value = self.nei_h_dict[nei_agent.name]
            nei_pf = self.nei_pf_dict[nei_agent.name]

            up_until_t = len(nei_plan)
            weight = self._get_weight(nei_heuristic_value=nei_heuristic_value)
            nei_pfs[:, :, :up_until_t] += weight * nei_pf

        # # ---------- memory part ---------- # #
        self.memory *= 0.9
        norm_memory = np.repeat(self.memory[:, :, np.newaxis], max_plan_len, axis=2)
        memory_weight = 2
        norm_memory *= memory_weight
        # print(norm_memory)

        self.nei_pfs = nei_pfs
        nei_pfs += norm_memory
        return nei_pfs, max_plan_len


class AlgSDS(AlgParObsPFPrPSeq):
    """
    Public methods:
    .first_init(env)
    .reset()
    .get_actions(observations)
    """
    def __init__(self, params, alg_name):
        super().__init__(params, alg_name)
        self.nums_order_list = None
        self.rand_nums_order_list = None

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

    def _sds_shuffle(self):
        # random.shuffle(self.agents)
        if random.random() < 0.1:
            others_list = [a for a in self.agents if a.time_passed_from_last_goal > self.h]
            random.shuffle(others_list)
            reached_list = [a for a in self.agents if a.time_passed_from_last_goal <= self.h]
            random.shuffle(reached_list)
            others_list.extend(reached_list)
            self.agents = others_list
            return

        others_list = [a for a in self.agents if a.time_passed_from_last_goal > self.h-1]
        if len(others_list) > 0:
            others_list.sort(key=lambda a: a.heuristic_value, reverse=True)
        else:
            random.shuffle(others_list)
        # choose i_agent
        if len(others_list) > 0 and self.i_agent not in others_list:
            self.i_agent = others_list[0]
        # self.i_agent = self.agents_dict['agent_0']
        reached_list = [a for a in self.agents if a.time_passed_from_last_goal <= self.h-1]
        # random.shuffle(reached_list)
        if len(others_list) > 0 and len(reached_list) > 0:
            reached_distance_dict = {}
            for r_a in reached_list:
                r_a_distances = []
                for o_a in others_list:
                    distance = manhattan_distance_nodes(r_a.curr_node, o_a.curr_node)
                    r_a_distances.append(distance)
                reached_distance_dict[r_a.name] = min(r_a_distances)
            reached_list.sort(key=lambda a: reached_distance_dict[a.name])
        else:
            random.shuffle(reached_list)
        others_list.extend(reached_list)
        self.agents = others_list

        # i_agent = self.agents_dict['agent_0']
        # self.agents.sort(key=lambda a: distance_nodes(a.curr_node, i_agent.curr_node), reverse=False)

        # self.agents.sort(key=lambda a: a.time_passed_from_last_goal, reverse=True)
        # self.i_agent = self.agents[0]
        # self.agents.sort(key=lambda a: distance_nodes(a.curr_node, self.i_agent.curr_node), reverse=False)
        # print(f'The i_agent: ------------------------------ {self.i_agent.name}, val: {self.i_agent.time_passed_from_last_goal}')

        # self.agents.sort(key=lambda a: a.time_passed_from_last_goal + self.h_dict[a.prev_goal_node.xy_name][a.next_goal_node.x, a.next_goal_node.y], reverse=True)
        # self.agents.sort(key=lambda a: a.time_passed_from_last_goal, reverse=True)
        # self.agents.sort(key=lambda a: a.heuristic_value, reverse=True)

    def _build_plans(self):
        if self.h and self.curr_iteration % self.h != 0 and self.curr_iteration != 0:
            return {}
        distr_time = 0

        self._sds_shuffle()
        self.nums_order_list = [a.num for a in self.agents]
        self.rand_nums_order_list = self.nums_order_list[:]
        random.shuffle(self.rand_nums_order_list)

        # init path
        time_limit_crossed, distr_time = self._all_initial_sds_assignment(distr_time)
        if time_limit_crossed:
            return {}

        self._all_exchange_plans()

        time_limit_crossed, distr_time = self._all_secondary_sds_assignment(distr_time)
        if time_limit_crossed:
            return {}

        # while there are conflicts
        there_are_collisions, inner_iter = True, 0
        while there_are_collisions:
            inner_iter += 1

            # exchange with neighbours
            self._all_exchange_plans()

            # find collisions
            there_are_collisions, distr_time = self._all_find_conf_agents(distr_time, inner_iter)
            if not there_are_collisions:
                break

            # replan
            time_limit_crossed, distr_time = self._all_replan(distr_time, inner_iter)
            if time_limit_crossed:
                return {}

    def _all_initial_sds_assignment(self, distr_time):
        parallel_times = [0]
        for i, agent in enumerate(self.agents):

            local_start_time = time.time()

            agent.build_plan([])

            # limit check
            passed_time = time.time() - local_start_time
            parallel_times.append(passed_time)
            if distr_time + passed_time > self.time_to_think_limit:
                self._cut_up_to_the_limit(i)
                return True, distr_time + max(parallel_times)

        return False, distr_time + max(parallel_times)

    def _all_secondary_sds_assignment(self, distr_time):
        parallel_times = [0]
        for i, agent in enumerate(self.agents):

            local_start_time = time.time()

            agent.secondary_sds_init_plan(self.nums_order_list)

            # limit check
            passed_time = time.time() - local_start_time
            parallel_times.append(passed_time)
            if distr_time + passed_time > self.time_to_think_limit:
                self._cut_up_to_the_limit(i)
                return True, distr_time + max(parallel_times)

        return False, distr_time + max(parallel_times)

    def _all_find_conf_agents(self, distr_time, inner_iter):
        print()
        there_are_collisions, n_collisions = False, 0
        col_str = ''
        for agent1, agent2 in combinations(self.agents, 2):
            if agent1.name not in agent2.nei_dict:
                continue
            plan1 = agent1.get_full_plan()
            plan2 = agent2.get_full_plan()
            if not two_plans_have_no_confs(plan1, plan2):
                there_are_collisions = True
                agent1.a_names_in_conf_list.append(agent2.name)
                agent2.a_names_in_conf_list.append(agent1.name)
                n_collisions += 1
                col_str = f'{agent1.name} <-> {agent2.name}'
                # print(f'>>>> {agent1.name}-{agent2.name}')
                # return there_are_collisions, distr_time
        print(f'\n>>>> {inner_iter=}, {n_collisions} collisions, last one: {col_str}')
        return there_are_collisions, distr_time

    def _all_exchange_plans(self):
        for agent in self.agents:
            agent.a_names_in_conf_list = []
            agent.nei_plans_dict = {nei.name: nei.plan for nei in agent.nei_list}
            agent.nei_h_dict = {nei.name: nei.heuristic_value for nei in agent.nei_list}
            agent.nei_pf_dict = {nei.name: nei.pf_field for nei in agent.nei_list}
            agent.nei_succ_dict = {nei.name: nei.plan_succeeded for nei in agent.nei_list}

    def _all_replan(self, distr_time, inner_iter):
        parallel_times = [0]
        for i, agent in enumerate(self.agents):

            local_start_time = time.time()

            # EACH AGENT:
            agent.replan(nums_order_list=self.nums_order_list, inner_iter=inner_iter)

            # limit check
            passed_time = time.time() - local_start_time
            parallel_times.append(passed_time)
            if distr_time + passed_time > self.time_to_think_limit:
                self._cut_up_to_the_limit(i)
                return True, distr_time + max(parallel_times)

        return False, distr_time + max(parallel_times)


@use_profiler(save_dir='../stats/alg_sds.pstat')
def main():
    # Alg params
    pf_weight = 1
    pf_size = 4
    h = 5
    w = h
    # alg_name = 'SDS'
    # alg_name = 'PF-SDS'
    # alg_name = 'ParObs-SDS'
    alg_name = 'ParObs-PF-SDS'

    params_dict = {
        'SDS': {},
        'PF-SDS': {'pf_weight': pf_weight, 'pf_size': pf_size},
        'ParObs-SDS': {'h': h, 'w': w},
        'ParObs-PF-SDS': {'h': h, 'w': w, 'pf_weight': pf_weight, 'pf_size': pf_size},
    }

    alg = AlgSDS(params=params_dict[alg_name], alg_name=alg_name)
    test_single_alg(
        alg,

        # GENERAL
        # random_seed=True,
        random_seed=False,
        seed=222,
        PLOT_PER=1,
        # PLOT_PER=20,
        PLOT_RATE=0.001,
        PLOT_FROM=1,
        middle_plot=True,
        # middle_plot=False,
        final_plot=True,
        # final_plot=False,

        # FOR ENV
        # iterations=200,  # !!!
        iterations=100,
        # iterations=50,
        n_agents=400,
        n_problems=1,
        classical_rhcr_mapf=True,
        # classical_rhcr_mapf=False,
        time_to_think_limit=30,  # seconds
        rhcr_mapf_limit=10000,
        global_time_limit=6000,  # seconds

        # Map
        # img_dir='empty-32-32.map',  # 32-32
        # img_dir='random-32-32-10.map',  # 32-32          | LNS | Up to 400 agents with w=5, h=2, lim=1min.
        img_dir='random-32-32-20.map',  # 32-32
        # img_dir='room32-32-4.map',  # 32-32
        # img_dir='maze-32-32-2.map',  # 32-32

        # img_dir='10_10_my_rand.map',  # 32-32
        # img_dir='warehouse-10-20-10-2-1.map',  # 32-32
    )


if __name__ == '__main__':
    main()

# my_plan_len = len(self.plan) + self.heuristic_value
# # path length + index
# for nei_name, nei_plan in self.nei_plans_dict.items():
#     nei_agent = self.nei_dict[nei_name]
#     nei_heuristic_value = self.nei_h_dict[nei_name]
#     nei_plan_len = len(nei_plan) + nei_heuristic_value
#     if nei_name in self.a_names_in_conf_list:
#         h_agents.append(nei_agent)
#     elif my_plan_len > nei_plan_len:
#         if random.random() < p_l:
#             h_agents.append(nei_agent)
#     elif my_plan_len < nei_plan_len:
#         if random.random() < p_h:
#             h_agents.append(nei_agent)
#     elif self.num > nei_agent.num:
#         if random.random() < p_l:
#             h_agents.append(nei_agent)
#     else:
#         if random.random() < p_h:
#             h_agents.append(nei_agent)
# return h_agents


# my_plan_len = int(len(self.plan) + self.heuristic_value)
# if len(conf_plans_len_list) == 1 and conf_plans_len_list[0] == my_plan_len:
#     return 0.5
# conf_plans_len_list.append(my_plan_len)
# conf_plans_len_list.sort()
# my_order = conf_plans_len_list.index(my_plan_len)
# prob_change = 0.9 - 0.8 * (my_order / (len(conf_plans_len_list) - 1))
# return prob_change


# self.agents.sort(key=lambda a: self.h_dict[a.prev_goal_node.xy_name][a.next_goal_node.x, a.next_goal_node.y], reverse=True)
# print(f'\n{[a.time_passed_from_last_goal + self.h_dict[a.prev_goal_node.xy_name][a.next_goal_node.x, a.next_goal_node.y] for a in self.agents]}')

# if random.random() < 0.9:
#     self.agents.sort(key=lambda a: a.heuristic_value, reverse=True)
#     # self.agents.sort(key=lambda a: a.time_passed_from_last_goal, reverse=True)
#     # self.agents.sort(key=lambda a: a.time_passed_from_last_goal, reverse=False)
# else:
#     stuck_agents, other_agents = [], []
#     # going_agents = []
#     for agent in self.agents:
#         if not agent.plan_succeeded:
#             stuck_agents.append(agent)
#             continue
#         # if agent.heuristic_value > 0:
#         #     going_agents.append(agent)
#         #     continue
#         other_agents.append(agent)
#     random.shuffle(stuck_agents)
#     # random.shuffle(going_agents)
#     random.shuffle(other_agents)
#     # stuck_agents.extend(going_agents)
#     stuck_agents.extend(other_agents)
#     self.agents = stuck_agents

# def collect_all_nei_pfs(self):
#     if len(self.nei_list) == 0:
#         return None, None
#     max_plan_len = max([len(plan) for plan in self.nei_plans_dict.values()])
#     nei_pfs = np.zeros((self.map_dim[0], self.map_dim[1], max_plan_len))  # x, y, t
#     for nei_agent in self.nei_list:
#
#         nei_plan = self.nei_plans_dict[nei_agent.name]
#         nei_heuristic_value = self.nei_h_dict[nei_agent.name]
#         nei_pf = self.nei_pf_dict[nei_agent.name]
#
#         up_until_t = len(nei_plan)
#         weight = self._get_weight(nei_heuristic_value=nei_heuristic_value)
#         nei_pfs[:, :, :up_until_t] += weight * nei_pf
#
#     return nei_pfs, max_plan_len





# if self.done_for_now:
#     for conf_nei_name in self.a_names_in_conf_list:
#         conf_nei = self.nei_dict[conf_nei_name]
#         if nums_order_list.index(conf_nei.num) < nums_order_list.index(self.num):
#             break
#         if nums_order_list.index(conf_nei.num) > nums_order_list.index(self.num):
#             if self.nei_succ_dict[conf_nei.name]:
#                 break
#         return [], False


# if check_stay_at_goal(self.plan, self.next_goal_node):
#     return
# SDS variant
# h_agents = []
# for nei in self.nei_list:
#     if random.random() < 0.9:
#         h_agents.append(nei)
# return h_agents, True

# LNS variant
# in_process = 0
# for nei in self.nei_list:
#     if nums_order_list.index(nei.num) < nums_order_list.index(self.num) and nei.name in self.a_names_in_conf_list:
#         break
#     if nums_order_list.index(nei.num) > nums_order_list.index(self.num) and self.nei_succ_dict[nei.name]:
#         in_process += 1
# if in_process == len(self.nei_list):
#     return [], False

# if all([check_stay_at_goal(self.nei_plans_dict[nei.name], nei.next_goal_node) for nei in self.nei_list]):
#     print('all')

# if I reached the goal already and have a collision
# if check_stay_at_goal(self.plan, self.next_goal_node):
#     h_agents.append(nei)
#     continue

# if nei.name in self.was_already_idle:
#     if inner_iter < self.h:
#         continue
#     h_agents.append(nei)
#     continue

