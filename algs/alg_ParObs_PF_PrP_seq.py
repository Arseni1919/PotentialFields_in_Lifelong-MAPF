from typing import List, Dict
from tools_for_graph import *
from tools_for_heuristics import *
from tools_for_plotting import *
from algs.test_single_alg import test_single_alg
from algs.alg_a_star_space_time import a_star_xyt


class ParObsPFPrPAgent:
    """
    Public methods:
    .update_obs(obs, **kwargs)
    .clean_nei()
    .add_nei(agent1)
    .build_plan(h_agents)
    .choose_action()
    """

    def __init__(self, num: int, start_node, next_goal_node, **kwargs):
        # h_value = h_dict[to_node.xy_name][from_node.x, from_node.y]
        self.num = num
        self.name = f'agent_{num}'
        self.start_node: Node = start_node
        self.name_start_node = start_node.xy_name
        self.prev_node: Node = start_node
        self.curr_node: Node = start_node
        self.prev_goal_node: Node = start_node
        self.next_goal_node: Node = next_goal_node
        self.name_next_goal_node = next_goal_node.xy_name
        self.first_goal_node: Node = next_goal_node
        self.closed_goal_nodes: List[Node] = []
        self.plan = None
        self.plan_succeeded = False
        self.nodes = kwargs['nodes']
        self.nodes_dict = kwargs['nodes_dict']
        self.h_func = kwargs['h_func']
        self.h_dict = kwargs['h_dict']
        self.params = kwargs['params']
        self.map_dim = kwargs['map_dim']
        self.heuristic_value = None
        self.heuristic_value_init = self.h_dict[self.next_goal_node.xy_name][self.curr_node.x, self.curr_node.y]
        self.pf_field = None
        self.memory = np.zeros((self.map_dim[0], self.map_dim[1]))
        self.nei_list, self.nei_dict, self.nei_plans_dict, self.nei_h_dict, self.nei_pf_dict, self.nei_succ_dict = [], {}, {}, {}, {}, {}
        self.nei_num_dict = {}
        self.nei_pfs = None
        self.h, self.w = set_h_and_w(self)
        self.pf_weight = set_pf_weight(self)
        self.latest_arrival = None
        self.time_passed_from_last_goal = None

    def update_obs(self, obs, **kwargs):
        self.curr_node = obs['curr_node']
        self.prev_node = obs['prev_node']
        self.prev_goal_node = obs['prev_goal_node']
        self.next_goal_node = obs['next_goal_node']
        self.closed_goal_nodes = obs['closed_goal_nodes']
        self.latest_arrival = obs['latest_arrival']
        self.time_passed_from_last_goal = obs['time_passed_from_last_goal']
        self.heuristic_value = self.h_dict[self.next_goal_node.xy_name][self.curr_node.x, self.curr_node.y]
        if self.curr_node.xy_name != self.next_goal_node.xy_name:
            self.memory[self.curr_node.x, self.curr_node.y] += 1

    def clean_nei(self):
        self.nei_list, self.nei_dict, self.nei_plans_dict, self.nei_h_dict, self.nei_pf_dict, self.nei_succ_dict = [], {}, {}, {}, {}, {}
        self.nei_num_dict = {}

    def add_nei(self, nei_agent):
        self.nei_list.append(nei_agent)
        self.nei_dict[nei_agent.name] = nei_agent
        self.nei_plans_dict[nei_agent.name] = nei_agent.plan
        self.nei_h_dict[nei_agent.name] = nei_agent.heuristic_value
        self.nei_num_dict[nei_agent.name] = nei_agent.num
        self.nei_pf_dict[nei_agent.name] = None  # in the _all_exchange_plans method
        self.nei_succ_dict[nei_agent.name] = None  # in the _all_exchange_plans method

    def build_plan(self, h_agents, goal=None, nodes=None, nodes_dict=None):
        # self._execute_a_star(h_agents)
        if h_agents is None:
            h_agents = []
        if self.plan is None or len(self.plan) == 0:
            nei_h_agents = [agent for agent in h_agents if agent.name in self.nei_dict]
            sub_results = create_sub_results(nei_h_agents)
            v_constr_dict, e_constr_dict, perm_constr_dict, xyt_problem = build_constraints(self.nodes, sub_results)
            nei_pfs, max_plan_len = self._build_nei_pfs(nei_h_agents)
            self.execute_a_star(v_constr_dict, e_constr_dict, perm_constr_dict, xyt_problem, nei_pfs,
                                goal=goal, nodes=nodes, nodes_dict=nodes_dict)
        return self.plan

    def correct_nei_pfs(self):
        if self.nei_pfs is not None:
            self.nei_pfs = self.nei_pfs[:, :, 1:]
            if self.nei_pfs.shape[2] == 0:
                self.nei_pfs = None

    def choose_action(self):
        next_node: Node = self.plan.pop(0)
        if self.pf_weight != 0:
            self.pf_field = self.pf_field[:, :, 1:]
            self.correct_nei_pfs()
        return next_node.xy_name

    def get_full_plan(self):
        full_plan = [self.curr_node]
        full_plan.extend(self.plan)
        return full_plan

    def _get_weight(self, nei_heuristic_value):

        # pf_weight_pref = self.params['pf_weight_pref'] if 'pf_weight_pref' in self.params else 'short_paths'
        # pf_weight_pref = self.params['pf_weight_pref'] if 'pf_weight_pref' in self.params else 'long_paths'
        # pf_weight_pref = self.params['pf_weight_pref'] if 'pf_weight_pref' in self.params else 'my_h_long'
        # pf_weight_pref = self.params['pf_weight_pref'] if 'pf_weight_pref' in self.params else 'my_h_short'
        pf_weight_pref = self.params['pf_weight_pref'] if 'pf_weight_pref' in self.params else 'uniform'

        # Prefer Longer Paths
        if pf_weight_pref == 'long_paths':
            relative_length = (nei_heuristic_value / self.heuristic_value) / 2
            if 0 <= relative_length < 0.2:
                return 0
            if 0.2 <= relative_length < 0.4:
                return 0.25 * self.pf_weight
            if 0.4 <= relative_length < 0.6:
                return 0.5 * self.pf_weight
            if 0.6 <= relative_length < 0.8:
                return 0.75 * self.pf_weight
            if 0.8 <= relative_length:
                return self.pf_weight

        # Prefer Shorter Paths
        if pf_weight_pref == 'short_paths':
            relative_length = (nei_heuristic_value / self.heuristic_value) / 2
            if 0 <= relative_length < 0.2:
                return self.pf_weight
            if 0.2 <= relative_length < 0.4:
                return 0.75 * self.pf_weight
            if 0.4 <= relative_length < 0.6:
                return 0.5 * self.pf_weight
            if 0.6 <= relative_length < 0.8:
                return 0.25 * self.pf_weight
            if 0.8 <= relative_length:
                return 0

        if pf_weight_pref == 'my_h_long':
            big_size = max(self.map_dim) / 2
            if self.heuristic_value < big_size:
                partial = 1 - (self.heuristic_value / big_size)
                return partial * self.pf_weight
            return 0

        if pf_weight_pref == 'my_h_short':
            big_size = max(self.map_dim) / 2
            if self.heuristic_value < big_size:
                partial = self.heuristic_value / big_size
                return partial * self.pf_weight
            return self.pf_weight

        if pf_weight_pref == 'uniform':
            return self.pf_weight

        return self.pf_weight

    # POTENTIAL FIELDS ****************************** pf_weight ******************************
    def _build_nei_pfs(self, h_agents):
        if self.pf_weight == 0:
            self.nei_pfs = None
            return None, None
        if len(h_agents) == 0:
            return None, None
        max_plan_len = max([len(agent.plan) for agent in h_agents])
        nei_pfs = np.zeros((self.map_dim[0], self.map_dim[1], max_plan_len))  # x, y, t
        for nei_agent in h_agents:
            up_until_t = len(nei_agent.plan)
            weight = self._get_weight(nei_heuristic_value=nei_agent.heuristic_value)
            nei_pfs[:, :, :up_until_t] += weight * nei_agent.pf_field

        self.nei_pfs = nei_pfs
        return nei_pfs, max_plan_len

    # POTENTIAL FIELDS ****************************** pf_size  ******************************
    # POTENTIAL FIELDS ****************************** pf_shape ******************************
    def _get_gradient_list(self):
        pf_size = self.params['pf_size']
        pf_shape = self.params['pf_shape'] if 'pf_shape' in self.params else 2
        if pf_size == 'h':
            h_value = self.h_func(self.start_node, self.next_goal_node)
        else:
            h_value = pf_shape ** pf_size
        biggest_value = h_value
        gradient_list = [h_value]
        while h_value > 1:
            h_value /= pf_shape
            gradient_list.append(h_value)
        gradient_list = [i / biggest_value for i in gradient_list]
        return gradient_list

    # POTENTIAL FIELDS ************************* pf_shape of circle ************************
    def _create_pf_field(self):
        # if self.curr_node.xy_name == self.next_goal_node.xy_name: return
        if self.pf_weight == 0: return
        gradient_list = self._get_gradient_list()
        if len(gradient_list) == 0: return

        self.pf_field = np.zeros((self.map_dim[0], self.map_dim[1], len(self.plan)))
        if check_stay_at_same_node(self.plan, self.next_goal_node):
            return
        for i_time, next_node in enumerate(self.plan):
            nei_nodes, nei_nodes_dict = get_nei_nodes(next_node, len(gradient_list), self.nodes_dict)
            for i_node in nei_nodes:
                distance_index = min(len(gradient_list) - 1, math.floor(euclidean_distance_nodes(next_node, i_node)))
                self.pf_field[i_node.x, i_node.y, i_time] += float(gradient_list[distance_index])
        # plot_magnet_field(self.magnet_field)

    def execute_a_star(self, v_constr_dict, e_constr_dict, perm_constr_dict, xyt_problem, nei_pfs,
                       goal=None, nodes=None, nodes_dict=None):
        if goal is None:
            goal = self.next_goal_node
        if nodes is None or nodes_dict is None:
            nodes, nodes_dict = self.nodes, self.nodes_dict
        # v_constr_dict, e_constr_dict, perm_constr_dict, xyt_problem = self._create_constraints(h_agents)
        new_plan, a_s_info = a_star_xyt(start=self.curr_node, goal=goal,
                                        nodes=nodes, nodes_dict=nodes_dict, h_func=self.h_func,
                                        v_constr_dict=v_constr_dict, e_constr_dict=e_constr_dict,
                                        perm_constr_dict=perm_constr_dict,
                                        agent_name=self.name,
                                        # nei_pfs=nei_pfs, k_time=self.w + 1, xyt_problem=xyt_problem)
                                        nei_pfs=nei_pfs, xyt_problem=xyt_problem)
        if new_plan is not None:
            # pop out the current location, because you will order to move to the next location
            self.plan_succeeded = True
            new_plan.pop(0)
            self.plan = new_plan
            self.fulfill_the_plan()
            self._create_pf_field()
        else:
            # self.plan = None
            # IStay
            self.set_istay()

    def fulfill_the_plan(self):
        if len(self.plan) == 0:
            self.plan = [self.curr_node]
        if self.h and self.h < 1000:
            while len(self.plan) < self.h:
                self.plan.append(self.plan[-1])

    def set_istay(self):
        self.plan = [self.curr_node]
        self.fulfill_the_plan()
        self._create_pf_field()
        self.plan_succeeded = False
        # print(f' \r\t --- [{self.name}]: I stay!', end='')


class AlgParObsPFPrPSeq:
    """
    Public methods:
    .first_init(env)
    .reset()
    .get_actions(observations)
    """

    def __init__(self, params, alg_name):
        self.params = params
        self.alg_name = alg_name
        self.env = None
        self.agents = None
        self.agents_dict = {}
        self.n_agents = None
        self.map_dim = None
        self.nodes, self.nodes_dict = None, None
        self.h_dict = None
        self.h_func = None
        self.curr_iteration = None
        self.agents_names_with_new_goals = []
        self.i_agent = None

        # RHCR part
        self.h, self.w = set_h_and_w(self)

        # limits
        self.time_to_think_limit = None

    def first_init(self, env, **kwargs):
        self.env = env
        self.n_agents = env.n_agents
        self.map_dim = env.map_dim
        self.nodes, self.nodes_dict = env.nodes, env.nodes_dict
        self.h_dict = env.h_dict
        self.h_func = env.h_func

        self.agents = None
        self.agents_dict = {}
        self.curr_iteration = None
        self.agents_names_with_new_goals = []

        # limits
        self.time_to_think_limit = 1e6
        if 'time_to_think_limit' in kwargs:
            self.time_to_think_limit = kwargs['time_to_think_limit']

    def reset(self):
        self.agents: List[ParObsPFPrPAgent] = []
        for env_agent in self.env.agents:
            new_agent = ParObsPFPrPAgent(
                num=env_agent.num, start_node=env_agent.start_node, next_goal_node=env_agent.next_goal_node,
                nodes=self.nodes, nodes_dict=self.nodes_dict, h_func=self.h_func, h_dict=self.h_dict,
                map_dim=self.map_dim, params=self.params
            )
            self.agents.append(new_agent)
            self.agents_dict[new_agent.name] = new_agent
            self.curr_iteration = 0
        self.i_agent = self.agents_dict['agent_0']

    # @check_time_limit()
    def get_actions(self, observations, **kwargs):
        """
        observations[agent.name] = {
                'num': agent.num,
                'curr_node': agent.curr_node,
                'next_goal_node': agent.next_goal_node,
            }
        actions: {agent_name: node_name, ...}
        """
        self.curr_iteration = kwargs['iteration']

        # update the current state
        self.agents_names_with_new_goals = observations['agents_names_with_new_goals']
        for agent in self.agents:
            agent.update_obs(observations[agent.name], agents_dict=self.agents_dict)

        # update neighbours - RHCR part
        self._update_neighbours()

        # build the plans - PF part
        build_plans_info = self._build_plans()

        # choose the actions
        actions = {agent.name: agent.choose_action() for agent in self.agents}
        self._add_one_shot_to_actions(actions)

        alg_info = {
            'i_agent': self.i_agent,
            'i_nodes': self.nodes,
            'alg_name': self.alg_name,
            'time_to_think_limit': self.time_to_think_limit,
        }
        alg_info.update(build_plans_info)

        # checks
        check_actions_if_vc(self.agents, actions)
        check_actions_if_ec(self.agents, actions)

        return actions, alg_info

    def _update_neighbours(self):
        _ = [agent.clean_nei() for agent in self.agents]
        if self.h is None or self.h >= 1e6:
            for agent1, agent2 in combinations(self.agents, 2):
                agent1.add_nei(agent2)
                agent2.add_nei(agent1)
        else:
            for agent1, agent2 in combinations(self.agents, 2):
                distance = manhattan_distance_nodes(agent1.curr_node, agent2.curr_node)
                if distance <= 2 * self.h + 1:
                    agent1.add_nei(agent2)
                    agent2.add_nei(agent1)

    def _update_order(self):
        finished_list = []
        unfinished_list = []
        for agent in self.agents:
            if agent.plan is not None and len(agent.plan) == 0:
                finished_list.append(agent)
            else:
                unfinished_list.append(agent)
        self.agents = unfinished_list
        self.agents.extend(finished_list)

    def _reshuffle_agents(self):
        # print(f'\n**************** random reshuffle ****************\n')

        stuck_agents = [agent for agent in self.agents if not agent.plan_succeeded]
        good_agents = [agent for agent in self.agents if agent.plan_succeeded]
        random.shuffle(stuck_agents)
        random.shuffle(good_agents)
        stuck_agents.extend(good_agents)
        self.agents = stuck_agents

        # random.shuffle(self.agents)

        for agent in self.agents:
            agent.plan = None

    def _implement_istay(self):
        # IStay
        there_is_conf = True
        # pairs_list = list(combinations(self.agents, 2))
        standing_agents = set()
        while there_is_conf:
            there_is_conf = False
            for agent1, agent2 in combinations(self.agents, 2):
                if agent1.name not in agent2.nei_dict:
                    continue
                if agent1.name in standing_agents:
                    if agent2.name in standing_agents:
                        continue
                    if not plan_has_no_conf_with_vertex(agent2.plan, agent1.curr_node):
                        there_is_conf = True
                        agent2.set_istay()
                        standing_agents.add(agent2.name)
                        break
                    else:
                        continue
                if agent2.name in standing_agents:
                    if not plan_has_no_conf_with_vertex(agent1.plan, agent2.curr_node):
                        there_is_conf = True
                        agent1.set_istay()
                        standing_agents.add(agent1.name)
                        break
                    else:
                        continue
                if not two_plans_have_no_confs(agent1.plan, agent2.plan):
                    there_is_conf = True
                    agent1.set_istay()
                    agent2.set_istay()
                    standing_agents.add(agent1.name)
                    standing_agents.add(agent2.name)
                    break

    def _cut_up_to_the_limit(self, i):
        if len(self.agents) >= i:
            for failed_agent in self.agents[i + 1:]:
                failed_agent.set_istay()
        self._implement_istay()

    def _build_plans_restart(self):
        if self.h and self.curr_iteration % self.h != 0 and self.curr_iteration != 0:
            return
        start_time = time.time()
        self._update_order()

        h_agents = []
        need_to_shuffle = False
        for i, agent in enumerate(self.agents):
            agent.build_plan(h_agents)
            h_agents.append(agent)
            if not agent.plan_succeeded:
                need_to_shuffle = True

            # limit check
            end_time = time.time() - start_time
            if end_time > self.time_to_think_limit:
                self._cut_up_to_the_limit(i)
                return

        # IStay
        self._implement_istay()

        if need_to_shuffle:
            random.shuffle(self.agents)

    def initial_prp_assignment(self, start_time):
        # Persist
        h_agents = []
        for i, agent in enumerate(self.agents):
            agent.build_plan(h_agents)
            h_agents.append(agent)

            # limit check
            end_time = time.time() - start_time
            if end_time > self.time_to_think_limit:
                self._cut_up_to_the_limit(i)
                return True
        return False

    def _build_plans_persist(self):
        if self.h and self.curr_iteration % self.h != 0 and self.curr_iteration != 0:
            return
        start_time = time.time()
        # self._update_order()
        self._reshuffle_agents()

        # Persist
        time_limit_crossed = self.initial_prp_assignment(start_time)
        if time_limit_crossed:
            return

        # IStay
        self._implement_istay()

    def _add_one_shot_to_actions(self, actions):
        if 'one_shot' in self.params:
            actions['one_shot'] = self.params['one_shot']
            actions['latest_arrivals'] = {}
            actions["succeeded"] = True
            for agent in self.agents:
                if not agent.plan_succeeded:
                    actions["succeeded"] = False
                    break
                else:
                    actions['latest_arrivals'][agent.name] = len(agent.plan)

    def _build_plans(self):
        if self.h is None:
            self._build_plans_restart()
        else:
            self._build_plans_persist()
        return {}


@use_profiler(save_dir='../stats/alg_par_obs_pf_prp_seq.pstat')
def main():
    # Alg params
    alg_name = 'PrP'
    # alg_name = 'PF-PrP'
    # alg_name = 'ParObs-PrP'
    # alg_name = 'ParObs-PF-PrP'

    params_dict = {
        'PrP': {},
        'PF-PrP': {
            # 'pf_weight': 0.1,
            'pf_weight': 1,
            # 'pf_weight': 2,
            # 'pf_size': 'h',
            'pf_size': 3,
            # 'pf_size': 5,
            # 'pf_size': 2,
            'pf_shape': 2,
        },
        'ParObs-PrP': {
            # For RHCR
            'h': 5,  # my step
            'w': 5,  # my planning
        },
        'ParObs-PF-PrP': {
            # For PF
            # 'pf_weight': 0.5,
            'pf_weight': 1,
            # 'pf_weight': 2,
            # 'pf_weight': 3,
            # 'pf_weight': 5,
            # 'pf_weight': 10,
            # 'pf_size': 'h',
            # 'pf_size': 5,
            'pf_size': 4,
            # 'pf_size': 2,
            # For RHCR
            'h': 5,  # my step
            'w': 5,  # my planning
        },
    }

    alg = AlgParObsPFPrPSeq(params=params_dict[alg_name], alg_name=alg_name)
    test_single_alg(
        alg,

        # GENERAL
        # random_seed=True,
        random_seed=False,
        seed=321,
        PLOT_PER=1,
        PLOT_RATE=0.001,
        PLOT_FROM=1,
        middle_plot=True,
        # middle_plot=False,
        final_plot=True,
        # final_plot=False,

        # FOR ENV
        iterations=200,
        # iterations=100,
        # iterations=50,
        n_agents=150,
        n_problems=1,
        classical_rhcr_mapf=True,
        # classical_rhcr_mapf=False,
        global_time_limit=100000,
        time_to_think_limit=100000,  # seconds
        rhcr_mapf_limit=10000,

        # Map
        # img_dir='empty-32-32.map',  # 32-32
        # img_dir='random-32-32-10.map',  # 32-32          | LNS | Up to 400 agents with w=5, h=2, lim=1min.
        img_dir='random-32-32-20.map',  # 32-32
        # img_dir='room-32-32-4.map',  # 32-32
        # img_dir='maze-32-32-2.map',  # 32-32
        # img_dir='empty-48-48.map',
    )


if __name__ == '__main__':
    main()





