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
        self.order_init = None
        self.team_leader = None
        self.team_queue = []
        self.has_conf = True
        self.mem_weight = self.params['mem_weight'] if 'mem_weight' in self.params else 0
        self.changed_path = True

    def set_order(self, order):
        self.order = order
        self.order_init = order

    def secondary_sds_init(self):
        self.a_names_in_conf_list = []
        self.lower_agents_processed = []
        self.plan = None
        self.plan_succeeded = True
        self.has_conf = True
        self.changed_path = True
        # Decide who is your team leader, or be one
        self._set_team_leader()

    def replan(self, inner_iter):
        if self.plan and len(self.a_names_in_conf_list) == 0:
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
        min_order = min(orders_around.keys()) if len(orders_around) > 0 else self.order
        if self.order_init <= min_order:
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
        if self.name in team_queue:  # my name is in queue
            # check if all above me did not finish
            my_q_num = team_queue.index(self.name)
            for q_name in team_queue[:my_q_num]:
                if q_name in self.nei_dict:
                    nei = self.nei_dict[q_name]
                    if nei.plan_succeeded and nei.has_conf:
                        return False, None
        else:  # my name is not in queue
            # check if all in the queue did not finish
            for q_name in team_queue:
                if q_name in self.nei_dict:
                    nei = self.nei_dict[q_name]
                    if nei.plan_succeeded and nei.has_conf:
                        return False, None
            # from the same team and different order_init
            for nei in self.nei_list:
                if nei.order_init < self.order_init:
                    if nei.plan_succeeded and nei.has_conf:
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
        self.changed_path = True
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
            # if nei.name != req_agent.name and nei.name in req_agent.lower_agents_processed:
            #     h_agents.append(nei)
            #     continue

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
            # h_agents_node_names = []
            h_agents_node_names = [nei.curr_node.xy_name for nei in h_agents]
            # h_agents_node_names.extend([nei.plan[-1].xy_name for nei in h_agents])
            h_agents_node_names.extend([node.xy_name for node in req_agent.plan])
            # h_agents_node_names.extend(self.curr_node.neighbours)
            h_agents_node_names = list(set(h_agents_node_names))
            nei_nodes = list(filter(lambda n: n.xy_name not in h_agents_node_names, nei_nodes))
            rand_goal_node = random.choice(nei_nodes) if len(nei_nodes) != 0 else self.curr_node

        self.plan = None
        self.build_plan(h_agents, goal=rand_goal_node)
        self.changed_path = True

    # # POTENTIAL FIELDS ****************************** pf_weight ******************************
    def _build_nei_pfs(self, h_agents):
        if self.mem_weight == 0:
            self.nei_pfs = None
            return None, None

        # # ---------- memory part ---------- # #
        if len(h_agents) == 0:
            return None, None
        max_plan_len = max([len(agent.plan) for agent in h_agents])
        self.memory *= 0.9
        norm_memory = np.repeat(self.memory[:, :, np.newaxis], max_plan_len, axis=2)
        norm_memory *= self.mem_weight
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

    def _build_plans(self):
        # one_master = [agent for agent in self.agents if agent.master == agent.master_init][0]
        if self.h and self.curr_iteration % self.h != 0 and self.curr_iteration != 0:
            return self._get_info_to_send()
        distr_time = 0

        self._all_set_order(to_shuffle=False)  # set the masters

        time_limit_crossed, distr_time = self._all_secondary_sds_init(distr_time)
        if time_limit_crossed:
            return self._get_info_to_send()

        # build (numpy) roadmaps
        pass

        # build an order
        pass

        # build last resort locations in the roadmaps
        pass

        # build paths while adjusting to the team leader
        pass

        # find collisions
        there_are_collisions, distr_time = self._all_find_conf_agents(distr_time, -1)
        assert there_are_collisions

        return self._get_info_to_send()

    def _all_set_order(self, to_shuffle=True):
        # random.shuffle(self.agents)
        final_list = [a for a in self.agents if not a.plan_succeeded]
        succ_list = [a for a in self.agents if a.plan_succeeded]
        others_list = [a for a in succ_list if a.time_passed_from_last_goal > self.h+1]
        if to_shuffle:
            random.shuffle(others_list)
        # choose i_agent
        reached_list = [a for a in succ_list if a.time_passed_from_last_goal <= self.h+1]
        if to_shuffle:
            random.shuffle(reached_list)
        final_list.extend(others_list)
        if len(others_list) > 0 and self.i_agent not in final_list:
            self.i_agent = others_list[0]
        final_list.extend(reached_list)

        self.agents = final_list
        for i, agent in enumerate(self.agents):
            agent.set_order(i)

    def _get_info_to_send(self):
        orders_dict = {}
        if self.curr_iteration != 0:
            orders_dict = {agent.name: agent.order for agent in self.agents}
            leaders_orders = list(set(orders_dict.values()))
            leaders_orders.sort()
            orders_dict = {k: leaders_orders.index(v) for k, v in orders_dict.items()}
        info_to_send = {'orders_dict': orders_dict, 'one_master': self.i_agent}
        return info_to_send

    def _all_secondary_sds_init(self, distr_time):
        parallel_times = [0]
        for i, agent in enumerate(self.agents):

            local_start_time = time.time()

            agent.secondary_sds_init()

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
            if not agent1.changed_path and not agent2.changed_path:
                the_conf = list(filter(lambda x: x[0] == agent2.name, agent1.a_names_in_conf_list_prev))
                if len(the_conf) == 0:
                    have_confs = False
                else:
                    have_confs = True
                    conf_index = the_conf[0][1]
            else:
                plan1 = agent1.get_full_plan()
                plan2 = agent2.get_full_plan()
                have_confs, conf_index = two_plans_have_confs_at(plan1, plan2)
            if have_confs:
                there_are_collisions = True
                agent1.a_names_in_conf_list.append((agent2.name, conf_index))
                agent2.a_names_in_conf_list.append((agent1.name, conf_index))
                n_collisions += 1
                col_str = f'{agent1.name} <-> {agent2.name}'
        for agent1, agent2 in combinations(self.agents, 2):
            agent1.changed_path = False
            agent2.changed_path = False
        print(f'\r>>>> {inner_iter=}, {n_collisions} collisions, last one: {col_str}', end='')
        return there_are_collisions, distr_time

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