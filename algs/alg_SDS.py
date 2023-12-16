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
        self.lower_agents_processed = []
        self.higher_agents_processed_prev = []
        self.order = None
        self.order_init = None
        self.team_leader = None
        self.team_queue = []
        self.has_conf = True

    def set_order(self, order):
        self.order = order
        self.order_init = order

    def secondary_sds_init_plan(self):
        self.lower_agents_processed = []
        self.has_conf = True

        # Decide who is your team leader, or be one
        self._set_team_leader()

        # if I am standing on the goal
        if check_stay_at_same_node(self.plan, self.next_goal_node):
            return self.plan

        self._avoid_standing_agents()
        # if self.pf_weight == 0:
        #     # if it is possible to avoid the standing agents - better
        #     self._avoid_standing_agents()
        # else:
        #     self._avoid_standing_agents()

        return self.plan

    def replan(self, inner_iter):
        if len(self.a_names_in_conf_list) == 0:
            self.has_conf = False
            return
        i_am_the_highest, lower_a = self._set_i_am_the_highest()
        if i_am_the_highest:
            self._H_policy(lower_a)
        return
        # self.set_istay()
        # return

    def _set_team_leader(self):
        orders_around = {nei.order_init: nei for nei in self.nei_list}
        min_order = min(orders_around.keys())
        if self.order_init < min_order:
            # I am the leader
            self.team_leader = self
        else:
            # The leader is somebody else
            self.team_leader = orders_around[min_order]
            self.order = min_order
        self.team_queue = [self.name]

    def _avoid_standing_agents(self):
        # if it is possible to avoid the standing agents - better
        last_target = self.plan[-1]
        init_plan = self.plan
        l_agents = []
        for nei in self.nei_list:
            if check_stay_at_same_node(nei.plan, nei.curr_node):
                l_agents.append(nei)
                continue
        self.build_plan(h_agents=l_agents, goal=last_target)
        if self.plan[-1].xy_name != last_target.xy_name:
            self.plan = init_plan
            self.plan_succeeded = True
            self._fulfill_the_plan()
            self._create_pf_field()

    def _avoid_nei_pfs(self):
        v_constr_dict, e_constr_dict, perm_constr_dict, xyt_problem = build_constraints(self.nodes, {})
        h_agents = []
        for nei in self.nei_list:
            if nei.order < self.order:
                h_agents.append(nei)
        nei_pfs, max_plan_len = self._build_nei_pfs(h_agents)
        self.execute_a_star(v_constr_dict, e_constr_dict, perm_constr_dict, xyt_problem, nei_pfs)

    def _set_i_am_the_highest(self):
        # There is conf already
        # I am stuck
        if not self.plan_succeeded:
            return False, None

        self.a_names_in_conf_list.sort(key=lambda x: x[1])
        first_conf_name, first_conf_dist = self.a_names_in_conf_list[0]
        first_conf_agent = self.nei_dict[first_conf_name]
        # still in conf (but maybe the changed plan did the job already)
        # if two_plans_have_no_confs(self.get_full_plan(), first_conf_agent.get_full_plan()):
        #     return False, None

        if not first_conf_agent.plan_succeeded:
            return True, first_conf_agent

        # another agent from the higher team
        if first_conf_agent.order < self.order:
            return False, None

        # another agent from the lower team
        if first_conf_agent.order > self.order:
            # no need to fill some team queues
            return True, first_conf_agent

        if not first_conf_agent.plan_succeeded:
            return True, first_conf_agent

        # check if all the agents that before me in the queue finished
        team_queue = self.team_leader.team_queue
        if self.name in self.team_leader.team_queue:  # my name is in queue
            # check if all above me did not finish
            my_q_num = self.team_leader.team_queue.index(self.name)
            for q_name in self.team_leader.team_queue[:my_q_num]:
                if q_name in self.nei_dict:
                    nei = self.nei_dict[q_name]
                    if nei.plan_succeeded and nei.has_conf:
                        return False, None
        else:  # my name is not in queue
            # check if all in the queue did not finish
            for q_name in self.team_leader.team_queue:
                if q_name in self.nei_dict:
                    nei = self.nei_dict[q_name]
                    if nei.plan_succeeded and nei.has_conf:
                        return False, None

        if self.name not in self.team_leader.team_queue:
            # from the same team and different order_init
            if self.order_init > first_conf_agent.order_init:
                return False, None
            self.team_leader.team_queue.append(self.name)
        self.team_leader.team_queue.append(first_conf_agent.name)
        return True, first_conf_agent

    def _H_policy(self, lower_a):

        # if L is not idle send request
        if lower_a.plan_succeeded:
            to_continue = lower_a.L_policy(self)
            if self.order < lower_a.order:
                self.lower_agents_processed.append(lower_a.name)
            if not to_continue:
                return

        # after receiving alt plan
        h_agents = []
        for nei in self.nei_list:

            # idle agents
            if not nei.plan_succeeded:
                h_agents.append(nei)
                continue

            # H + L agents
            if nei.name in self.team_leader.team_queue:
                h_agents.append(nei)
                continue

            # consider all the higher teams
            if nei.team_leader.order_init < self.team_leader.order_init:
                h_agents.append(nei)
                continue

            # this one for the agents from the lower teams
            if nei.name in self.lower_agents_processed:
                h_agents.append(nei)
                continue

        self.plan = None
        self.build_plan(h_agents)

        return

    def L_policy(self, req_agent):
        assert req_agent.name != self.name

        # try to consider H agent that sent the request
        prev_last_node = self.plan[-1]
        self._L_policy_build_plans(req_agent, prev_last_node=prev_last_node, take_all=True)
        if self.plan_succeeded and self.plan[-1].xy_name == prev_last_node.xy_name:
            return False

        # ignore the H agent that sent the request
        self._L_policy_build_plans(req_agent, take_all=False)
        return True

    def _L_policy_build_plans(self, req_agent, prev_last_node=None, take_all=False):
        h_agents = []
        for nei in self.nei_list:

            if take_all:
                h_agents.append(nei)
                continue

            # idle agents
            if not nei.plan_succeeded:
                h_agents.append(nei)
                continue

            # H + L agents
            if nei.name != req_agent.name and nei.name in self.team_leader.team_queue:
                h_agents.append(nei)
                continue

            # consider all the higher teams
            if nei.name != req_agent.name and nei.team_leader.order_init < self.team_leader.order_init:
                h_agents.append(nei)
                continue

            # this one for the req_agent's lower teams
            if nei.name != req_agent.name and nei.name in req_agent.lower_agents_processed:
                h_agents.append(nei)
                continue

        if not take_all:
            assert req_agent not in h_agents

        nei_nodes, nei_nodes_dict = get_nei_nodes(self.curr_node, self.h, self.nodes_dict)
        if prev_last_node:
            rand_goal_node = prev_last_node
        elif self.curr_node.xy_name != self.next_goal_node.xy_name and self.next_goal_node.xy_name in nei_nodes_dict:
            # rand_goal_node = self.next_goal_node
            nei_t_nodes, nei_t_nodes_dict = get_nei_nodes(self.next_goal_node, self.h, self.nodes_dict)
            nei_nodes = list(filter(lambda n: n.xy_name in nei_t_nodes_dict, nei_nodes))
            rand_goal_node = random.choice(nei_nodes) if len(nei_nodes) != 0 else self.curr_node
        else:
            h_agents_node_names = [nei.curr_node.xy_name for nei in h_agents]
            # h_agents_node_names.extend([nei.plan[-1].xy_name for nei in h_agents])
            h_agents_node_names.extend([node.xy_name for node in req_agent.plan])
            h_agents_node_names = list(set(h_agents_node_names))
            nei_nodes = list(filter(lambda n: n.xy_name not in h_agents_node_names, nei_nodes))
            rand_goal_node = random.choice(nei_nodes) if len(nei_nodes) != 0 else self.curr_node

        self.plan = None
        self.build_plan(h_agents, goal=rand_goal_node)

    # # POTENTIAL FIELDS ****************************** pf_weight ******************************
    def _build_nei_pfs(self, h_agents):
        if self.pf_weight == 0:
            self.nei_pfs = None
            return None, None
        # if len(h_agents) == 0:
        #     return None, None
        # max_plan_len = max([len(self.nei_plans_dict[agent.name]) for agent in h_agents])
        # nei_pfs = np.zeros((self.map_dim[0], self.map_dim[1], max_plan_len))  # x, y, t
        # for nei_agent in h_agents:
        #
        #     nei_plan = self.nei_plans_dict[nei_agent.name]
        #     nei_heuristic_value = self.nei_h_dict[nei_agent.name]
        #     nei_pf = self.nei_pf_dict[nei_agent.name]
        #
        #     up_until_t = len(nei_plan)
        #     weight = self._get_weight(nei_heuristic_value=nei_heuristic_value)
        #     nei_pfs[:, :, :up_until_t] += weight * nei_pf
        # self.nei_pfs = nei_pfs
        # return nei_pfs, max_plan_len

        # # ---------- memory part ---------- # #
        if len(h_agents) == 0:
            return None, None
        max_plan_len = max([len(self.nei_plans_dict[agent.name]) for agent in h_agents])
        self.memory *= 0.9
        norm_memory = np.repeat(self.memory[:, :, np.newaxis], max_plan_len, axis=2)
        norm_memory *= self.pf_weight
        # print(norm_memory)
        self.nei_pfs = norm_memory
        return norm_memory, max_plan_len


class AlgSDS(AlgParObsPFPrPSeq):
    """
    Public methods:
    .first_init(env)
    .reset()
    .get_actions(observations)
    """
    def __init__(self, params, alg_name):
        super().__init__(params, alg_name)

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
        failed_list = [a for a in self.agents if not a.plan_succeeded]
        succ_list = [a for a in self.agents if a.plan_succeeded]
        others_list = [a for a in succ_list if a.time_passed_from_last_goal > self.h+1]
        random.shuffle(others_list)
        # choose i_agent
        reached_list = [a for a in succ_list if a.time_passed_from_last_goal <= self.h+1]
        random.shuffle(reached_list)
        failed_list.extend(others_list)
        if len(others_list) > 0 and self.i_agent not in failed_list:
            self.i_agent = others_list[0]
        failed_list.extend(reached_list)
        self.agents = failed_list
        # others_list.extend(reached_list)
        # self.agents = others_list
        for i, agent in enumerate(self.agents):
            agent.set_order(i)

    def _get_info_to_send(self):
        orders_dict = {}
        if self.curr_iteration != 0:
            orders_dict = {agent.name: agent.team_leader.name for agent in self.agents}
            leaders_name = list(set(orders_dict.values()))
            orders_dict = {k: leaders_name.index(v) for k, v in orders_dict.items()}
        info_to_send = {'orders_dict': orders_dict, 'one_master': self.i_agent}
        return info_to_send

    def _build_plans(self):
        # one_master = [agent for agent in self.agents if agent.master == agent.master_init][0]
        info_to_send = self._get_info_to_send()
        if self.h and self.curr_iteration % self.h != 0 and self.curr_iteration != 0:
            return info_to_send
        distr_time = 0

        self._sds_shuffle()  # set the masters

        # init path
        time_limit_crossed, distr_time = self._all_initial_sds_assignment(distr_time)
        if time_limit_crossed:
            return info_to_send

        self._all_exchange_plans()

        time_limit_crossed, distr_time = self._all_secondary_sds_assignment(distr_time)
        if time_limit_crossed:
            return info_to_send

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
                return info_to_send
        return info_to_send

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

            agent.secondary_sds_init_plan()

            # limit check
            passed_time = time.time() - local_start_time
            parallel_times.append(passed_time)
            if distr_time + passed_time > self.time_to_think_limit:
                self._cut_up_to_the_limit(i)
                return True, distr_time + max(parallel_times)

        return False, distr_time + max(parallel_times)

    def _all_find_conf_agents(self, distr_time, inner_iter):
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
        print(f'\r>>>> {inner_iter=}, {n_collisions} collisions, last one: {col_str}', end='')
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
            agent.replan(inner_iter=inner_iter)

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
    pf_size = 2
    h = 3
    w = h
    # alg_name = 'SDS'
    # alg_name = 'PF-SDS'
    alg_name = 'ParObs-SDS'
    # alg_name = 'ParObs-PF-SDS'

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
        seed=321,
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
        n_agents=500,
        n_problems=1,
        classical_rhcr_mapf=True,
        # classical_rhcr_mapf=False,
        time_to_think_limit=100000,  # seconds
        rhcr_mapf_limit=10000,
        global_time_limit=6000,  # seconds

        # Map
        # img_dir='empty-32-32.map',  # 32-32
        # img_dir='random-32-32-10.map',  # 32-32          | LNS | Up to 400 agents with w=5, h=2, lim=1min.
        img_dir='random-32-32-20.map',  # 32-32
        # img_dir='room-32-32-4.map',  # 32-32
        # img_dir='maze-32-32-2.map',  # 32-32

        # img_dir='10_10_my_rand.map',  # 32-32
        # img_dir='random-64-64-20.map',  # 64-64
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

# if not self.plan_succeeded:
#     for lower_a_name in self.lower_agents_processed:
#         lower_a = self.nei_dict[lower_a_name]
#         lower_a.set_istay()

# conf_orders_dict = {}
# for conf_nei_name, conf_index in self.a_names_in_conf_list:
#     # conf_orders_dict[conf_nei_name] = nums_order_list.index(self.nei_dict[conf_nei_name].num)
#     nei_order = nums_order_list.index(self.nei_dict[conf_nei_name].num)
#     conf_orders_dict[nei_order] = conf_nei_name
# conf_orders = list(conf_orders_dict.keys())
# conf_orders.sort()
# first_order, first_order_agent = conf_orders[0], self.nei_dict[conf_orders_dict[conf_orders[0]]]
# assert self_order != first_order
# if self_order < first_order or not first_order_agent.plan_succeeded:
#     lower_a_name = conf_orders_dict[first_order]
#     lower_a = self.nei_dict[lower_a_name]
#     return True, lower_a
# return False, None