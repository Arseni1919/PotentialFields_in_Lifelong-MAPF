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
    file_dir = '2023-11-22--21-20_ALGS-1_RUNS-3_MAP-random-32-32-10.json'
    show_results(file_dir=f'logs_for_plots/{file_dir}')


if __name__ == '__main__':
    main()


