from typing import List
from tools_for_graph import *
from tools_for_heuristics import *
from tools_for_plotting import *
from algs.alg_a_star_space_time import a_star_xyt


class SimAgent:
    def __init__(self, num, start_node, next_goal_node):
        self.num = num
        self.name = f'agent_{num}'
        self.start_node: Node = start_node
        self.prev_node: Node = start_node
        self.curr_node: Node = start_node
        self.next_goal_node: Node = next_goal_node
        self.closed_goal_nodes: List[Node] = []
        self.plan = []
        self.reached_the_goal = False
        self.latest_arrival = None
        # self.nei_list, self.nei_dict = [], {}

    def latest_arrival_at_the_goal(self, iteration):
        if self.curr_node.xy_name != self.next_goal_node.xy_name:
            self.reached_the_goal = False
            return
        if not self.reached_the_goal:
            self.reached_the_goal = True
            self.latest_arrival = iteration

    def build_plan(self, **kwargs):
        nodes = kwargs['nodes']
        nodes_dict = kwargs['nodes_dict']
        h_func = kwargs['h_func']
        v_constr_dict = {node.xy_name: [] for node in nodes}
        e_constr_dict = {node.xy_name: [] for node in nodes}
        perm_constr_dict = {node.xy_name: [] for node in nodes}
        new_plan, a_s_info = a_star_xyt(start=self.curr_node, goal=self.next_goal_node,
                                        nodes=nodes, nodes_dict=nodes_dict, h_func=h_func,
                                        v_constr_dict=v_constr_dict, e_constr_dict=e_constr_dict,
                                        perm_constr_dict=perm_constr_dict,
                                        # magnet_w=magnet_w, mag_cost_func=mag_cost_func,
                                        # plotter=self.plotter, middle_plot=self.middle_plot,
                                        # iter_limit=self.iter_limit, k_time=k_time,
                                        agent_name=self.name)
        new_plan.pop(0)
        self.plan = new_plan


class EnvLifelongMAPF:
    def __init__(self, n_agents, img_dir, **kwargs):
        self.n_agents = n_agents
        self.agents: List[SimAgent] = None
        self.img_dir = img_dir
        self.classical_rhcr_mapf = kwargs['classical_rhcr_mapf'] if 'classical_rhcr_mapf' in kwargs else False
        if self.classical_rhcr_mapf:
            self.rhcr_mapf_limit = kwargs['rhcr_mapf_limit']
        path_to_maps = kwargs['path_to_maps'] if 'path_to_maps' in kwargs else '../maps'
        self.map_dim = get_dims_from_pic(img_dir=img_dir, path=path_to_maps)
        self.nodes, self.nodes_dict, self.img_np = build_graph_nodes(img_dir=img_dir, path=path_to_maps, show_map=False)
        path_to_heuristics = kwargs['path_to_heuristics'] if 'path_to_heuristics' in kwargs else '../logs_for_heuristics'
        self.h_dict = parallel_build_heuristic_for_entire_map(self.nodes, self.map_dim,
                                                              img_dir=img_dir, path=path_to_heuristics)
        self.h_func = h_func_creator(self.h_dict)

        # for a single run
        self.start_nodes = None
        self.first_goal_nodes = None
        self.iteration = None

        # for plotting
        self.middle_plot = kwargs['middle_plot']
        if self.middle_plot:
            self.plot_per = kwargs['plot_per']
            self.plot_rate = kwargs['plot_rate']
            self.plot_from = kwargs['plot_from']
            # self.fig, self.ax = plt.subplots()
            # self.fig, self.ax = plt.subplots(figsize=(14, 7))
            # self.fig, self.ax = plt.subplots(figsize=(7, 7))
            self.fig, self.ax = plt.subplots(1, 2, figsize=(14, 7))

    def reset(self, same_start):
        self.iteration = 0
        first_run = same_start and self.start_nodes is None
        if first_run or not same_start:
            self.start_nodes = random.sample(self.nodes, self.n_agents)
            # available_nodes = [node for node in self.nodes if node not in self.start_nodes]
            # self.first_goal_nodes = random.sample(available_nodes, self.n_agents)
            self.first_goal_nodes = random.sample(self.nodes, self.n_agents)
        self._create_agents()
        observations = self._get_observations([a.name for a in self.agents])
        return observations

    def sample_actions(self, **kwargs):
        actions = {}
        for agent in self.agents:
            if len(agent.plan) == 0:
                agent.build_plan(nodes=self.nodes, nodes_dict=self.nodes_dict, h_func=self.h_func)
            next_node = agent.plan.pop(0)
            actions[agent.name] = next_node.xy_name
        return actions

    def _classical_rhcr_mapf_termination(self):
        if self.classical_rhcr_mapf:
            if self.iteration >= self.rhcr_mapf_limit:
                return True, 0
            for agent in self.agents:
                if not agent.reached_the_goal:
                    return False, 0
            return True, 1
        return False, 0

    def _process_single_shot(self, actions):
        to_continue = True
        observations, succeeded, termination, info = None, None, None, None
        if 'one_shot' in actions:
            to_continue = False
            observations = self._get_observations([])
            succeeded = True
            succeeded, termination, info = {}, True, {}
        return to_continue, (observations, succeeded, termination, info)

    def step(self, actions):
        """
        Events might be:
        (1) reaching the goal by any agent, and receiving next assignment
        (2) proceeding to the next moving horizon (h/w in RHCR)
        (3) a collision
        (4) no plan for any agent
        """
        # to_continue, return_values = self._process_single_shot(actions)
        # if not to_continue:
        #     observations, succeeded, termination, info = return_values
        #     return observations, succeeded, termination, info
        self.iteration += 1
        self._execute_actions(actions)
        agents_names_with_new_goals = self._execute_event_new_goal()
        observations = self._get_observations(agents_names_with_new_goals)
        termination, succeeded = self._classical_rhcr_mapf_termination()
        info = {}
        return observations, succeeded, termination, info

    def render(self, info):
        if self.middle_plot and info['i'] >= self.plot_from and info['i'] % self.plot_per == 0:
            plot_env_field(self.ax[0], info)
            plot_magnet_agent_view(self.ax[1], info)
            plt.pause(self.plot_rate)
        n_closed_goals = sum([len(agent.closed_goal_nodes) for agent in self.agents])
        print(f"\n\n[{len(self.agents)}][{info['alg_name']}] PROBLEM: {info['i_problem'] + 1}/{info['n_problems']}, ITERATION: {info['i'] + 1}\n"
              f"Total closed goals --------------------------------> {n_closed_goals}\n"
              f"Total time --------------------------------> {info['runtime']: .2f}s\n")

    def _create_agents(self):
        self.agents = []
        for i, (start_node, goal_node) in enumerate(zip(self.start_nodes, self.first_goal_nodes)):
            new_agent = SimAgent(num=i, start_node=start_node, next_goal_node=goal_node)
            self.agents.append(new_agent)

    def _get_observations(self, agents_names_with_new_goals):
        observations = {
            'agents_names': [agent.name for agent in self.agents],
            'agents_names_with_new_goals': agents_names_with_new_goals
        }
        for agent in self.agents:
            observations[agent.name] = {
                'num': agent.num,
                'curr_node': agent.curr_node,
                'prev_node': agent.prev_node,
                'next_goal_node': agent.next_goal_node,
                'closed_goal_nodes': agent.closed_goal_nodes,
                'latest_arrival': agent.latest_arrival,
                # 'nei_list': [nei.name for nei in agent.nei_list]
            }
        return observations

    def _execute_actions(self, actions):
        for agent in self.agents:
            next_node_name = actions[agent.name]
            agent.prev_node = agent.curr_node
            agent.curr_node = self.nodes_dict[next_node_name]
            if self.classical_rhcr_mapf:
                agent.latest_arrival_at_the_goal(self.iteration)
        # checks
        check_if_nei_pos(self.agents)
        check_if_vc(self.agents)
        check_if_ec(self.agents)

    def _execute_event_new_goal(self):
        if self.classical_rhcr_mapf:
            return []
        goals_names_list = [agent.next_goal_node.xy_name for agent in self.agents]
        available_nodes = [node for node in self.nodes if node.xy_name not in goals_names_list]
        random.shuffle(available_nodes)
        agents_names_with_new_goals = []
        for agent in self.agents:
            if agent.curr_node.xy_name == (closed_goal := agent.next_goal_node).xy_name:
                agent.closed_goal_nodes.append(closed_goal)
                new_goal_node = available_nodes.pop()
                agent.next_goal_node = new_goal_node
                agent.plan = []
                agents_names_with_new_goals.append(agent.name)
        return agents_names_with_new_goals

    def close(self):
        pass


@use_profiler(save_dir='../stats/EnvLifelongMAPF.pstat')
def main():
    # --------------------------------------------------- #
    # params
    # --------------------------------------------------- #

    # General
    # random_seed = True
    random_seed = False
    seed = 612
    PLOT_PER = 1
    PLOT_RATE = 0.001
    PLOT_FROM = 1
    middle_plot = True
    # middle_plot = False
    final_plot = True
    # final_plot = False

    # --------------------------------------------------- #

    # For env
    iterations = 200
    n_agents = 2
    n_problems = 3
    # Map
    # img_dir = 'empty-32-32.map'  # 32-32
    # img_dir = 'random-32-32-10.map'  # 32-32          | LNS | Up to 400 agents with w=5, h=2, lim=1min.
    img_dir = 'random-32-32-20.map'  # 32-32
    # img_dir = 'room-32-32-4.map'  # 32-32
    # img_dir = 'maze-32-32-2.map'  # 32-32
    # img_dir = 'den312d.map'  # 65-81

    # For alg
    pass

    # for save
    # to_save_results = True
    # to_save_results = False
    # file_dir = f'logs_for_plots/{datetime.now().strftime("%Y-%m-%d--%H-%M")}_MAP-{img_dir[:-4]}.json'

    # --------------------------------------------------- #
    # --------------------------------------------------- #

    # init
    set_seed(random_seed, seed)
    env = EnvLifelongMAPF(
        n_agents=n_agents, img_dir=img_dir,
        plot_per=PLOT_PER, plot_rate=PLOT_RATE, plot_from=PLOT_FROM,
        middle_plot=middle_plot, final_plot=final_plot,
    )
    start_time = time.time()
    info = {
        'iterations': iterations,
        'n_problems': n_problems,
        'n_agents': n_agents,
        'img_dir': img_dir,
        'map_dim': env.map_dim,
        'img_np': env.img_np,
        'PLOT_FROM': 0,
    }

    # loop for n_agents

    for i_problem in range(n_problems):

        observations = env.reset(same_start=False)

        # loop for algs
        # observations = env.reset(same_start=True)

        # main loop
        for i in range(iterations):

            # step
            actions = env.sample_actions()  # here is the agents' decision
            observations, rewards, termination, step_info = env.step(actions)

            # render
            info.update(observations)
            info['i_problem'] = i_problem
            info['i'] = i
            info['runtime'] = time.time() - start_time
            env.render(info)

            # unexpected termination
            if termination:
                env.close()


if __name__ == '__main__':
    main()
