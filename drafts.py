import numpy as np

a = ['a', 'b', 'c']
b = ['', 'baa', 'aaac']
c = np.intersect1d(a, b)
d = np.random.randn(10, 10)
print(len(c))


import matplotlib.pyplot as plt
import json


def plot_time_metric_cactus(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    time_to_think_limit = info['time_to_think_limit']

    # x_list = n_agents_list[:4]
    x_list = n_agents_list
    for i_alg in alg_names:
        rt_list = []
        # res_str = ''
        for n_a in x_list:
            rt_list.extend(info[i_alg][f'{n_a}']['time'])
            # res_str += f'\t{n_a} - {rt_list[-1]: .2f}, '
        rt_list.sort()
        ax.plot(rt_list, '<choose marker>', color='<choose color>',
                alpha=0.5, label=f'{i_alg}', linewidth=2, markersize=10)
        # print(f'{i_alg}\t\t\t: {res_str}')
    # ax.set_xlim([min(x_list) - 20, max(x_list) + 20])
    # ax.set_xticks(x_list)
    ax.set_xlabel('Solved Instances', fontsize=15)
    ax.set_ylabel('Runtime', fontsize=15)
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.', fontweight="bold", size=11)
    legend_properties = {'weight': 'bold', 'size': 17}
    ax.legend(prop=legend_properties)

file_dir = ...
with open(f'{file_dir}', 'r') as openfile:
    # Reading from json file
    logs_dict = json.load(openfile)
    classical_rhcr_mapf = logs_dict['classical_rhcr_mapf']

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_time_metric_cactus(ax, info=logs_dict)