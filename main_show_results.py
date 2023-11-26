from globals import *
from tools_for_plotting import *


def show_results(**kwargs):
    plt.close()
    file_dir = kwargs['file_dir']
    with open(f'{file_dir}', 'r') as openfile:
        # Reading from json file
        logs_dict = json.load(openfile)
        classical_rhcr_mapf = logs_dict['classical_rhcr_mapf']
        if classical_rhcr_mapf:
            fig, ax = plt.subplots(1, 3, figsize=(12, 7))
            plot_sr(ax[0], info=logs_dict)
            plot_soc(ax[1], info=logs_dict)
            plot_time(ax[2], info=logs_dict)
        else:
            fig, ax = plt.subplots()
            plot_throughput(ax, info=logs_dict)
        plt.show()


def main():
    # file_dir = '2023-11-24--16-29_ALGS-4_RUNS-15_MAP-empty-32-32.json'
    # show_results(file_dir=f'logs_for_plots/{file_dir}')

    # file_dir = '2023-11-25--01-08_ALGS-4_RUNS-15_MAP-empty-32-32.json'
    # file_dir = '2023-11-25--12-22_ALGS-4_RUNS-15_MAP-random-32-32-10.json'
    # file_dir = '2023-11-26--00-02_ALGS-4_RUNS-15_MAP-room-32-32-4.json'
    file_dir = '2023-11-26--12-37_ALGS-4_RUNS-15_MAP-maze-32-32-2.json'
    show_results(file_dir=f'final_logs/{file_dir}')


if __name__ == '__main__':
    main()


