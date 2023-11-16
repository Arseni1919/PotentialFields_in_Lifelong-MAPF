import json
from datetime import datetime

import matplotlib.pyplot as plt

from tools_for_plotting import Plotter


def show_results(file_dir, plotter):
    """
    to_save_dict = {
        'statistics_dict': statistics_dict,
        'runs_per_n_agents': runs_per_n_agents,
        'n_agents_list': n_agents_list,
        'algs_to_test_names': heap_list(algs_to_test_dict.keys()),
        'img_dir': img_dir

    }
    """
    with open(f'{file_dir}', 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
    statistics_dict = json_object['stats_dict']
    runs_per_n_agents = json_object['runs_per_n_agents']
    n_agents_list = json_object['n_agents_list']
    img_png = json_object['img_dir']
    # algs_to_test_names = json_object['algs_to_test_names']
    algs_to_test_dict = json_object['algs_to_test_dict']
    plotter.plot_big_test(statistics_dict, runs_per_n_agents, algs_to_test_dict, n_agents_list, img_png, is_json=True)
    plt.show()


def main():

    """ 20 RUNS PER N AGENTS: """
    # file_dir = f'../logs_for_plots/2023-01-06--18-28_ALGS-3_RUNS-20_MAP-empty-48-48.json'
    # file_dir = f'../logs_for_plots/2023-01-06--20-44_ALGS-3_RUNS-20_MAP-random-64-64-10.json'
    # file_dir = f'../logs_for_plots/2023-01-09--08-23_ALGS-3_RUNS-20_MAP-warehouse-10-20-10-2-1.json'
    # file_dir = f'../logs_for_plots/2023-01-07--21-03_ALGS-3_RUNS-20_MAP-lt_gallowstemplar_n.json'

    # file_dir = f'../logs_for_plots/2023-01-17--19-06_ALGS-4_RUNS-10_MAP-empty-48-48.json'  # pbs
    # file_dir = f'../logs_for_plots/2023-01-17--21-44_ALGS-4_RUNS-10_MAP-random-64-64-10.json'  # pbs

    # file_dir = '../logs_for_plots/2023-01-17--13-51_ALGS-3_RUNS-5_MAP-random-64-64-10.json'  # mgm
    # file_dir = '../logs_for_plots/appendix_pp_dsa_sds.json'  # dsa

    # file_dir = f'../logs_for_plots/2023-08-14--08-24_ALGS-6_RUNS-3_MAP-warehouse-10-20-10-2-1.json'

    # main paper
    # file_dir = f'../logs_for_plots/empty_1.json'
    # file_dir = f'../logs_for_plots/rand_1.json'
    # file_dir = f'../logs_for_plots/ware_1.json'
    # file_dir = f'../logs_for_plots/game_1.json'
    file_dir = f'../logs_for_graphs/dsa_and_others_in_random_map.json'
    # appendix
    # file_dir = f'../logs_for_plots/appendix_pbs_pp_empty.json'
    # file_dir = f'../logs_for_plots/appendix_pbs_pp_ware.json'
    # file_dir = f'../logs_for_plots/appendix_e_prp_dprp_kdprp.json'
    # file_dir = f'../logs_for_plots/appendix_w_h_empty.json'
    # file_dir = f'../logs_for_plots/appendix_w_h_ware_part_1.json'
    # file_dir = f'../logs_for_plots/appendix_p_h_p_l_empty.json'
    # file_dir = f'../logs_for_plots/appendix_p_h_p_l_ware.json'

    plotter = Plotter()
    show_results(file_dir, plotter=plotter)


if __name__ == '__main__':
    main()
