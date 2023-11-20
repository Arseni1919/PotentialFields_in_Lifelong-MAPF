from typing import List, Dict

import matplotlib.pyplot as plt

from tools_for_graph import *
from tools_for_heuristics import *
from tools_for_plotting import *
from environments.env_LL_MAPF import EnvLifelongMAPF
from algs.alg_ParObs_PF_PrP_seq import AlgParObsPFPrPSeq
from algs.alg_LNS2 import AlgLNS2Seq


def save_results(**kwargs):
    algorithms = kwargs['algorithms']
    runs_per_n_agents = kwargs['runs_per_n_agents']
    img_dir = kwargs['img_dir']
    file_dir = f'logs_for_graphs/{datetime.now().strftime("%Y-%m-%d--%H-%M")}_ALGS-{len(algorithms)}_RUNS-{runs_per_n_agents}_MAP-{img_dir[:-4]}.json'


def show_results():
    pass


def run_big_experiments(**kwargs):
    # ------------------------- General
    random_seed = kwargs['random_seed']
    seed = kwargs['seed']
    # save & show
    middle_plot = kwargs['middle_plot']
    final_plot = kwargs['final_plot']
    to_save_results = kwargs['to_save_results']

    # ------------------------- For Simulation
    n_agents_list = kwargs['n_agents_list']
    runs_per_n_agents = kwargs['runs_per_n_agents']

    # ------------------------- For Env
    iterations = kwargs['iterations']

    # ------------------------- For algs
    algorithms = kwargs['algorithms']
    # limits
    time_to_think_limit = kwargs['time_to_think_limit']  # seconds

    # ------------------------- Map
    img_dir = kwargs['img_dir']

    # ------------------------- Plotting
    logs_dict = {
        alg.alg_name: {
            f'{n_agents}': {
                'n_closed_goals': []
            } for n_agents in n_agents_list
        } for alg in algorithms
    }
    if middle_plot:
        fig, ax = plt.subplots()

    # init
    set_seed(random_seed, seed)

    # n agents
    for n_agents in n_agents_list:
        env = EnvLifelongMAPF(n_agents=n_agents, img_dir=img_dir,
                              middle_plot=False, final_plot=False, plot_per=1, plot_rate=0.001, plot_from=50,
                              path_to_maps='maps', path_to_heuristics='logs_for_heuristics')
        # n runs
        for i_run in range(runs_per_n_agents):
            env.reset(same_start=False)
            # algorithms
            for alg in algorithms:
                alg.first_init(env)
                observations = env.reset(same_start=True)
                alg.reset()

                start_time = time.time()
                info = {
                    'iterations': iterations,
                    'n_problems': i_run,
                    'n_agents': n_agents,
                    'img_dir': img_dir,
                    'map_dim': env.map_dim,
                    'img_np': env.img_np,
                }

                # iterations
                for i in range(iterations):

                    # !!!!!!!!!!!!!!!!!
                    actions, alg_info = alg.get_actions(observations, iteration=i)  # here is the agents' decision

                    # step
                    observations, rewards, termination, step_info = env.step(actions)

                    # render
                    info.update(observations)
                    info.update(alg_info)
                    info['i_problem'] = i_run
                    info['i'] = i
                    info['runtime'] = time.time() - start_time
                    env.render(info)

                    # unexpected termination
                    if termination:
                        env.close()

                    # print
                    # n_closed_goals = sum([len(agent.closed_goal_nodes) for agent in alg.agents])
                    # print(f'\n ##########################################')
                    # print(f'\r [{n_agents} agents][{i_run + 1} run][{alg.alg_name}][{i} iteration] ---> {n_closed_goals}', end='')
                    # print(f'\n ##########################################')

                # logs
                n_closed_goals = sum([len(agent.closed_goal_nodes) for agent in alg.agents])
                logs_dict[alg.alg_name][f'{n_agents}']['n_closed_goals'].append(n_closed_goals)

                # for rendering
                if middle_plot:
                    ax.cla()
                    for i_alg in algorithms:
                        y_list = []
                        for n_a in n_agents_list:
                            y_list.append(np.mean(logs_dict[i_alg.alg_name][f'{n_a}']['n_closed_goals']))
                        ax.plot(n_agents_list, y_list, '-o', label=i_alg.alg_name)
                    ax.legend()
                    plt.pause(0.001)

    if to_save_results:
        pass

    if final_plot:
        pass

    plt.show()


@use_profiler(save_dir='../stats/run_big_experiments.pstat')
def main():

    run_big_experiments(
        # ------------------------- General
        random_seed=True,
        # random_seed=False,
        seed=958,
        # save & show
        middle_plot=True,
        # middle_plot=False,
        final_plot=True,
        # to_save_results = True,
        to_save_results=False,

        # ------------------------- For Simulation
        n_agents_list=[110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310],
        # n_agents_list=[10, 30, 50, 70, 90, 110],
        # n_agents_list=[80, 100, 120, 140, 160, 180, 200, 220, 240, 260],
        runs_per_n_agents=2,

        # ------------------------- For Env
        iterations=200,

        # ------------------------- For algs
        algorithms=[
            # AlgParObsPFPrPSeq(alg_name='PrP', params={}),
            # AlgParObsPFPrPSeq(alg_name='ParObs-PrP', params={'h': 5, 'w': 5}),
            # AlgParObsPFPrPSeq(alg_name='PF-PrP', params={'pf_weight': 5, 'pf_size': 3}),
            # AlgParObsPFPrPSeq(alg_name='ParObs-PF-PrP', params={'h': 5, 'w': 5, 'pf_weight': 5, 'pf_size': 3}),

            # AlgLNS2Seq(alg_name='LNS2', params={'big_N': 5}),
            AlgLNS2Seq(alg_name='ParObs-LNS2', params={'big_N': 5, 'h': 5, 'w': 5}),
            # AlgLNS2Seq(alg_name='PF-LNS2', params={'big_N': 5, 'pf_weight': 5, 'pf_size': 3}),
            AlgLNS2Seq(alg_name='(1/10)ParObs-PF-LNS2', params={'big_N': 5, 'h': 5, 'w': 5, 'pf_weight': 0.1, 'pf_size': 3}),
            AlgLNS2Seq(alg_name='(1/2)ParObs-PF-LNS2', params={'big_N': 5, 'h': 5, 'w': 5, 'pf_weight': 0.5, 'pf_size': 3}),
            AlgLNS2Seq(alg_name='(1)ParObs-PF-LNS2', params={'big_N': 5, 'h': 5, 'w': 5, 'pf_weight': 1, 'pf_size': 3}),
            AlgLNS2Seq(alg_name='(2)ParObs-PF-LNS2', params={'big_N': 5, 'h': 5, 'w': 5, 'pf_weight': 2, 'pf_size': 3}),
            # AlgLNS2Seq(alg_name='(10)ParObs-PF-LNS2', params={'big_N': 5, 'h': 5, 'w': 5, 'pf_weight': 10, 'pf_size': 3}),
        ],
        # limits
        time_to_think_limit=10,  # seconds

        # ------------------------- Map
        # img_dir = 'empty-32-32.map'  # 32-32
        img_dir='random-32-32-10.map'  # 32-32          | LNS | Up to 400 agents with w=5, h=2, lim=1min.
        # img_dir = 'random-32-32-20.map'  # 32-32
        # img_dir = 'room-32-32-4.map'  # 32-32
        # img_dir = 'maze-32-32-2.map'  # 32-32
    )


if __name__ == '__main__':
    main()

