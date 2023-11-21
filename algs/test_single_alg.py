from environments.env_LL_MAPF import EnvLifelongMAPF
from functions import *


def test_single_alg(alg, **kwargs):
    # --------------------------------------------------- #
    # params
    # --------------------------------------------------- #

    # General
    random_seed = kwargs['random_seed']
    seed = kwargs['seed']
    PLOT_PER = kwargs['PLOT_PER']
    PLOT_RATE = kwargs['PLOT_RATE']
    middle_plot = kwargs['middle_plot']
    final_plot = kwargs['final_plot']
    PLOT_FROM = kwargs['PLOT_FROM'] if 'PLOT_FROM' in kwargs else 0

    # --------------------------------------------------- #

    # For env
    iterations = kwargs['iterations']
    n_agents = kwargs['n_agents']
    n_problems = kwargs['n_problems']
    classical_mapf = kwargs['classical_mapf']
    time_to_think_limit = kwargs['time_to_think_limit']
    # Map
    img_dir = kwargs['img_dir']

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
        classical_mapf=classical_mapf,
        plot_per=PLOT_PER, plot_rate=PLOT_RATE, plot_from=PLOT_FROM,
        middle_plot=middle_plot, final_plot=final_plot,
    )

    # !!!!!!!!!!!!!!!!!
    alg.first_init(env, time_to_think_limit=time_to_think_limit)

    start_time = time.time()

    info = {
        'iterations': iterations,
        'n_problems': n_problems,
        'n_agents': n_agents,
        'img_dir': img_dir,
        'map_dim': env.map_dim,
        'img_np': env.img_np,
    }

    # loop for n_agents

    for i_problem in range(n_problems):

        observations = env.reset(same_start=False)

        # !!!!!!!!!!!!!!!!!
        alg.reset()

        # loop for algs
        # observations = env.reset(same_start=True)

        # main loop
        for i in range(iterations):

            # !!!!!!!!!!!!!!!!!
            actions, alg_info = alg.get_actions(observations, iteration=i)  # here is the agents' decision

            # step
            observations, rewards, termination, step_info = env.step(actions)

            # render
            info.update(observations)
            info.update(alg_info)
            info['i_problem'] = i_problem
            info['i'] = i
            info['runtime'] = time.time() - start_time
            env.render(info)

            # unexpected termination
            if termination:
                env.close()

    plt.show()


if __name__ == '__main__':
    # test_single_alg(alg)
    pass
