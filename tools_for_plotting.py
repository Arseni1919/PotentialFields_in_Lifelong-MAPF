from globals import *
from functions import get_color


def plot_magnet_field(path, data):
    plt.rcParams["figure.figsize"] = [8.00, 8.00]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot field
    if data is not None:
        x_l, y_l, z_l = np.nonzero(data > 0)
        col = data[data > 0]
        alpha_col = col / max(col) if len(col) > 0 else 1
        # alpha_col = np.exp(col) / max(np.exp(col))
        cm = plt.colormaps['Reds']  # , cmap=cm
        ax.scatter(x_l, y_l, z_l, c=col, alpha=alpha_col, marker='s', cmap=cm)
    # plot line
    if path:
        path_x = [node.x for node in path]
        path_y = [node.y for node in path]
        path_z = list(range(len(path_x)))
        ax.plot(path_x, path_y, path_z)
    plt.show()
    # plt.pause(2)


def get_line_marker(index, kind):
    if kind == 'l':
        lines = ['--', '-', '-.', ':']
        index = index % len(lines)
        return lines[index]
    elif kind == 'm':
        markers = ['^', '1', '2', 'X', 'd', 'v', 'o']
        index = index % len(markers)
        return markers[index]
    else:
        raise RuntimeError('no such kind')


def set_plot_title(ax, title, size=9):
    ax.set_title(f'{title}', fontweight="bold", size=size)


def set_log(ax):
    # log = True
    log = False
    if log:
        ax.set_yscale('log')
    return log


def plot_text_in_cactus(ax, l_x, l_y):
    if len(l_x) > 0:
        ax.text(l_x[-1] - 2, l_y[-1], f'{l_x[-1] + 1}', bbox=dict(facecolor='yellow', alpha=0.75))


def set_legend(ax, framealpha=None, size=9):
    to_put_legend = True
    # to_put_legend = False
    if to_put_legend:
        if not framealpha:
            framealpha = 0
        legend_properties = {'weight': 'bold', 'size': size}
        # legend_properties = {}
        if framealpha is not None:
            ax.legend(prop=legend_properties, framealpha=framealpha)
        else:
            ax.legend(prop=legend_properties)


# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #


def plot_env_field(ax, info):
    ax.cla()
    # nodes = info['nodes']
    # a_name = info['i_agent'].name if 'i_agent' in info else 'agent_0'
    iterations = info["iterations"]
    n_agents = info['n_agents']
    img_dir = info['img_dir']
    map_dim = info['map_dim']
    img_np = info['img_np']
    curr_iteration = info["i"]
    i_problem = info['i_problem']
    n_problems = info['n_problems']
    agents_names = info['agents_names']
    orders_dict = info['orders_dict']
    one_master = info['one_master']

    field = img_np * -1
    others_y_list, others_x_list, others_cm_list = [], [], []
    a_y_list, a_x_list, a_cm_list = [], [], []
    g_y_list, g_x_list, g_cm_list = [], [], []
    for i, agent_name in enumerate(agents_names):
        curr_node = info[agent_name]['curr_node']
        if agent_name == one_master.name:
            a_x_list.append(curr_node.x)
            a_y_list.append(curr_node.y)
            a_cm_list.append(get_color(orders_dict[agent_name]))
            # a_cm_list.append('k')
            next_goal_node = info[agent_name]['next_goal_node']
            g_x_list.append(next_goal_node.x)
            g_y_list.append(next_goal_node.y)
        else:
            others_y_list.append(curr_node.y)
            others_x_list.append(curr_node.x)
            others_cm_list.append(get_color(orders_dict[agent_name]))
    ax.scatter(a_y_list, a_x_list, s=200, c='white')
    ax.scatter(a_y_list, a_x_list, s=100, c=np.array(a_cm_list))
    ax.scatter(g_y_list, g_x_list, s=200, c='white', marker='X')
    ax.scatter(g_y_list, g_x_list, s=100, c='red', marker='X')
    ax.scatter(others_y_list, others_x_list, s=100, c='k')
    ax.scatter(others_y_list, others_x_list, s=50, c=np.array(others_cm_list))
    # ax.scatter(others_y_list, others_x_list, s=50, c='yellow')

    ax.imshow(field, origin='lower')
    ax.set_title(f'Map: {img_dir[:-4]}\n '
                 f'{n_agents} agents, selected: {one_master.name} - {one_master.order}\n'
                 f'(run:{i_problem + 1}/{n_problems}, time: {curr_iteration + 1}/{iterations})')


def plot_magnet_agent_view(ax, info):
    ax.cla()
    # paths_dict = info['paths_dict']
    agent = info['i_agent']
    nodes = info['i_nodes']
    side_x, side_y = info['map_dim']
    t = info['i']

    field = np.zeros((side_x, side_y))

    # magnet field
    if agent.nei_pfs is not None:
        if nodes:
            for node in nodes:
                field[node.x, node.y] = agent.nei_pfs[node.x, node.y, 0]

    # an agent
    ax.scatter(agent.curr_node.y, agent.curr_node.x, s=200, c='white')
    ax.scatter(agent.curr_node.y, agent.curr_node.x, s=100, c='k')
    ax.scatter(agent.next_goal_node.y, agent.next_goal_node.x, s=200, c='white', marker='X')
    ax.scatter(agent.next_goal_node.y, agent.next_goal_node.x, s=100, c='red', marker='X')

    # agent's nei poses
    # x_path = [node.x for node in agent.nei_nodes]
    # y_path = [node.y for node in agent.nei_nodes]
    # ax.scatter(y_path, x_path, c='green', alpha=0.05)


    # its path
    if agent.plan is not None and len(agent.plan) > 0:
        x_path = [node.x for node in agent.plan]
        y_path = [node.y for node in agent.plan]
        # ax.plot(x_path, y_path, c='yellow')
        ax.plot(y_path, x_path, c='blue')

    ax.imshow(field, origin='lower', cmap='hot')
    ax.set_title(f"{agent.name}'s View (time: {t})")


def plot_step_in_mapf_paths(ax, info):
    ax.cla()
    paths_dict = info['paths_dict']
    nodes = info['nodes']
    side_x = info['side_x']
    side_y = info['side_y']
    t = info['t']
    img_dir = info['img_dir']
    a_name = info['agent'].name if 'agent' in info else 'agent_0'
    longest_path = info['longest_path']

    field = np.zeros((side_x, side_y))

    if nodes:
        for node in nodes:
            field[node.x, node.y] = -1

    n = len(list(paths_dict.keys()))
    color_map = plt.cm.get_cmap('hsv', n)
    i = 0
    for agent_name, path in paths_dict.items():
        t_path = path[:t + 1]
        # for node in t_path:
        #     field[node.x, node.y] = 3
        if agent_name == a_name:
            ax.scatter(t_path[-1].y, t_path[-1].x, s=200, c='white')
            ax.scatter(t_path[-1].y, t_path[-1].x, s=100, c='k')
        else:
            ax.scatter(t_path[-1].y, t_path[-1].x, s=100, c='k')
            ax.scatter(t_path[-1].y, t_path[-1].x, s=50, c=np.array([color_map(i)]))
        # ax.text(t_path[-1].y - 0.4, t_path[-1].x - 0.4, agent_name[6:])
        i += 1

    # for agent_name, path in paths_dict.items():
    #     # field[path[0].x, path[0].y] = 4
    #     field[path[-1].x, path[-1].y] = 5

    ax.imshow(field, origin='lower')
    ax.set_title(f'Map: {img_dir[:-4]}, N_agents: {n} (time: {t}/{longest_path})')


def plot_sr(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    time_to_think_limit = info['time_to_think_limit']

    for i_alg in alg_names:
        sr_list = []
        for n_a in n_agents_list:
            sr_list.append(np.sum(info[i_alg][f'{n_a}']['sr']) / len(info[i_alg][f'{n_a}']['sr']))
        ax.plot(n_agents_list, sr_list, markers_lines_dict[i_alg], color=colors_dict[i_alg], alpha=0.5, label=f'{i_alg}')
    ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    ax.set_xticks(n_agents_list)
    ax.set_xlabel('N agents')
    ax.set_ylabel('Success Rate')
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    set_plot_title(ax, f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.',
                   size=10)
    set_legend(ax, size=12)


def plot_soc(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    time_to_think_limit = info['time_to_think_limit']

    for i_alg in alg_names:
        soc_list = []
        for n_a in n_agents_list:
            soc_list.append(np.mean(info[i_alg][f'{n_a}']['soc']))
        ax.plot(n_agents_list, soc_list, markers_lines_dict[i_alg], color=colors_dict[i_alg], alpha=0.5, label=f'{i_alg}')
    ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    ax.set_xticks(n_agents_list)
    ax.set_xlabel('N agents')
    ax.set_ylabel('Average SoC')
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    set_plot_title(ax, f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.',
                   size=10)
    set_legend(ax, size=12)


def plot_makespan(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    time_to_think_limit = info['time_to_think_limit']

    for i_alg in alg_names:
        makespan_list = []
        for n_a in n_agents_list:
            makespan_list.append(np.mean(info[i_alg][f'{n_a}']['makespan']))
        ax.plot(n_agents_list, makespan_list, '-^', label=f'{i_alg}')
    ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    ax.set_xticks(n_agents_list)
    ax.set_xlabel('N agents')
    ax.set_ylabel('Average Makespan')
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    set_plot_title(ax, f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.',
                   size=10)
    set_legend(ax, size=12)


def plot_time(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    time_to_think_limit = info['time_to_think_limit']

    for i_alg in alg_names:
        makespan_list = []
        for n_a in n_agents_list:
            makespan_list.append(np.mean(info[i_alg][f'{n_a}']['time']))
        ax.plot(n_agents_list, makespan_list, '-^', label=f'{i_alg}')
    ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    ax.set_xticks(n_agents_list)
    ax.set_xlabel('N agents')
    ax.set_ylabel('Average Time To Solve')
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    set_plot_title(ax, f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.',
                   size=10)
    set_legend(ax, size=12)


def plot_throughput(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    time_to_think_limit = info['time_to_think_limit']
    iterations = info['iterations']

    for i_alg in alg_names:
        y_list = []
        for n_a in n_agents_list:
            y_list.append(np.mean(info[i_alg][f'{n_a}']['n_closed_goals']))
        ax.plot(n_agents_list, y_list, markers_lines_dict[i_alg], color=colors_dict[i_alg], label=i_alg)
    ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    ax.set_xticks(n_agents_list)
    ax.set_xlabel('N agents')
    ax.set_ylabel('Average Throughput')
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec. | {iterations} iters.')
    set_plot_title(ax, f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec. | {iterations} iters.',
                   size=10)
    set_legend(ax, size=12)
    # ax.set_xlabel('N agents', labelpad=-1)
    # ax.set_ylabel('Average Throughput', labelpad=-1)
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
class Plotter:
    def __init__(self, map_dim=None, subplot_rows=2, subplot_cols=4, online_plotting=True):
        if map_dim:
            self.side_x, self.side_y = map_dim
        self.subplot_rows = subplot_rows
        self.subplot_cols = subplot_cols
        if online_plotting:
            self.fig, self.ax = plt.subplots(subplot_rows, subplot_cols, figsize=(14, 7))

    def close(self):
        plt.close()

    # online
    def plot_magnets_run(self, **kwargs):
        info = {
            'agent': kwargs['agent'],
            'paths_dict': kwargs['paths_dict'], 'nodes': kwargs['nodes'],
            'side_x': self.side_x, 'side_y': self.side_y, 't': kwargs['t'],
            'img_dir': kwargs['img_dir'] if 'img_dir' in kwargs else '',
        }
        plot_step_in_mapf_paths(self.ax[0], info)
        plot_magnet_agent_view(self.ax[1], info)
        plt.pause(0.001)
        # plt.pause(1)

    def plot_lists(self, open_list, closed_list, start, goal=None, path=None, nodes=None, a_star_run=False, **kwargs):
        plt.close()
        self.fig, self.ax = plt.subplots(1, 3, figsize=(14, 7))
        field = np.zeros((self.side_x, self.side_y))

        if nodes:
            for node in nodes:
                field[node.x, node.y] = -1

        for node in open_list:
            field[node.x, node.y] = 1

        for node in closed_list:
            field[node.x, node.y] = 2

        if path:
            for node in path:
                field[node.x, node.y] = 3

        field[start.x, start.y] = 4
        if goal:
            field[goal.x, goal.y] = 5

        self.ax[0].imshow(field, origin='lower')
        self.ax[0].set_title('general')

        # if path:
        #     for node in path:
        #         field[node.x, node.y] = 3
        #         self.ax[0].text(node.x, node.y, f'{node.ID}', bbox={'facecolor': 'yellow', 'alpha': 1, 'pad': 10})

        # open_list
        field = np.zeros((self.side_x, self.side_y))
        for node in open_list:
            if a_star_run:
                field[node.x, node.y] = node.g
            else:
                field[node.x, node.y] = node.g_dict[start.xy_name]
        self.ax[1].imshow(field, origin='lower')
        self.ax[1].set_title('open_list')

        # closed_list
        field = np.zeros((self.side_x, self.side_y))
        for node in closed_list:
            if a_star_run:
                field[node.x, node.y] = node.g
            else:
                field[node.x, node.y] = node.g_dict[start.xy_name]
        self.ax[2].imshow(field, origin='lower')
        self.ax[2].set_title('closed_list')

        self.fig.tight_layout()
        # plt.pause(1)
        # plt.pause(0.01)
        self.fig.suptitle(f'{kwargs["agent_name"]}', fontsize=16)
        plt.show()

    def plot_mapf_paths(self, paths_dict, nodes=None, **kwargs):
        plt.close()
        plt.rcParams["figure.figsize"] = [7.00, 7.00]
        # plt.rcParams["figure.autolayout"] = True
        plot_per = kwargs['plot_per']
        plot_rate = kwargs['plot_rate']
        self.fig, self.ax = plt.subplots()
        longest_path = max([len(path) for path in paths_dict.values()])

        for t in range(longest_path):
            if t % plot_per == 0:
                info = {
                    'paths_dict': paths_dict, 'nodes': nodes,
                    'side_x': self.side_x, 'side_y': self.side_y, 't': t,
                    'img_dir': kwargs['img_dir'] if 'img_dir' in kwargs else '',
                    'longest_path': longest_path,
                }
                plot_step_in_mapf_paths(self.ax, info)
                # plt.pause(1)
                plt.pause(plot_rate)

    # online
    # def plot_big_test(self, statistics_dict, runs_per_n_agents, algs_to_test_dict, n_agents_list, img_png='',
    #                   is_json=False, **kwargs):
    #     print('big plot starts')
    #     info = {
    #         'statistics_dict': statistics_dict,
    #         'runs_per_n_agents': runs_per_n_agents,
    #         'algs_to_test_dict': algs_to_test_dict,
    #         'n_agents_list': n_agents_list,
    #         'is_json': is_json
    #     }
    #     plot_success_rate(self.ax[0, 0], info)
    #
    #     plot_sol_quality(self.ax[0, 1], info)
    #
    #     plot_runtime_cactus(self.ax[0, 2], info)
    #
    #     plot_a_star_calls_counters(self.ax[0, 3], info)
    #
    #     plot_avr_nearby_agents(self.ax[1, 0], info)
    #
    #     plot_avr_distance(self.ax[1, 1], info)
    #
    #     # plot_n_nei(self.ax[1, 2], info)
    #
    #     plot_n_expanded_cactus(self.ax[1, 3], info)
    #
    #     time_per_alg_limit = f'{kwargs["time_per_alg_limit"]}' if 'time_per_alg_limit' in kwargs else '-'
    #     # self.fig.tight_layout()
    #     if 'i_run' in kwargs:
    #         title = f'{img_png}, {kwargs["i_run"]+1}/{runs_per_n_agents}, time limit: {float(time_per_alg_limit) * 60:0.0f} sec.'
    #     else:
    #         title = f'{img_png}'
    #     self.fig.suptitle(title, fontsize=16)
    #     plt.pause(0.1)
    #     print('big plot ends')
        # plt.show()



# def sub_plot_cactus_big_lines(ax, index, l_x, l_y, alg_name, alg_info):
#     line_style = get_line_or_marker(index, 'l')
#     linewidth = 2
#     alpha = 0.75
#     if 'color' in alg_info:
#         ax.plot(l_x, l_y, line_style, label=f'{alg_name}', alpha=alpha, color=alg_info['color'], linewidth=linewidth)
#     else:
#         raise RuntimeError('no color! ufff')
#         # ax.plot(l_x, l_y, line_style, label=f'{alg_name}', alpha=alpha, linewidth=linewidth)
#     plot_text_in_cactus(ax, l_x, l_y)
#
#
# def sub_plot_cactus_dist_lines(ax, index, l_x, l_y, alg_name, alg_info):
#     line_style = f"-{get_line_or_marker(index, 'm')}"
#     # line_style = get_line_or_marker(index, 'l')
#     linewidth = 1
#     alpha = 0.43
#     if 'color' in alg_info:
#         ax.plot(l_x, l_y, line_style, label=f'{alg_name}', alpha=alpha, color=alg_info['color'], linewidth=linewidth)
#         # ax.plot(l_x, l_y, line_style, alpha=alpha, color=alg_info['color'], linewidth=linewidth)
#     else:
#         raise RuntimeError('no color! ufff')
#         # ax.plot(l_x, l_y, line_style, label=f'{alg_name} (dist)', alpha=alpha, linewidth=linewidth)
#     plot_text_in_cactus(ax, l_x, l_y)


# def plot_success_rate(ax, info):
#     ax.cla()
#     statistics_dict = info['statistics_dict']
#     runs_per_n_agents = info['runs_per_n_agents']
#     algs_to_test_dict = info['algs_to_test_dict']
#     n_agents_list = info['n_agents_list']
#     is_json = info['is_json']
#
#     index = -1
#     for alg_name, (alg_func, alg_info) in algs_to_test_dict.items():
#         index += 1
#         marker = f"-{get_line_or_marker(index, 'm')}"
#         # success_rate
#         sr_x = []
#         sr_y = []
#         for n_agents in n_agents_list:
#             sr_list = get_list_n_run(statistics_dict, alg_name, n_agents, 'success_rate', runs_per_n_agents,
#                                      is_json)
#             if len(sr_list) > 0:
#                 sr_x.append(n_agents)
#                 sr_y.append(sum(sr_list) / len(sr_list))
#         if 'color' in alg_info:
#             ax.plot(sr_x, sr_y, marker, label=f'{alg_name}', alpha=0.9, color=alg_info['color'])
#         else:
#             ax.plot(sr_x, sr_y, marker, label=f'{alg_name}', alpha=0.9)
#
#     set_plot_title(ax, 'success rate')
#     ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
#     # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
#     # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#     ax.set_ylim([0, 1.5])
#     ax.set_xticks(n_agents_list)
#     ax.set_xlabel('N agents', labelpad=-1)
#     set_legend(ax)


# def plot_sol_quality(ax, info):
#     ax.cla()
#     statistics_dict = info['statistics_dict']
#     runs_per_n_agents = info['runs_per_n_agents']
#     algs_to_test_dict = info['algs_to_test_dict']
#     n_agents_list = info['n_agents_list']
#     is_json = info['is_json']
#
#     index = -1
#     for alg_name, (alg_func, alg_info) in algs_to_test_dict.items():
#         index += 1
#         # line = get_line_or_marker(index, 'l')
#         marker = f"-{get_line_or_marker(index, 'm')}"
#         # sol_quality
#         sq_x = []
#         sq_y = []
#         for n_agents in n_agents_list:
#             sq_list = get_list_sol_q_style(statistics_dict, alg_name, n_agents, 'sol_quality', runs_per_n_agents,
#                                            list(algs_to_test_dict.keys()), is_json)
#             if len(sq_list) > 0:
#                 sq_x.append(n_agents)
#                 sq_y.append(np.mean(sq_list))
#         if 'color' in alg_info:
#             ax.plot(sq_x, sq_y, marker, label=f'{alg_name}', alpha=0.8, color=alg_info['color'])
#         else:
#             ax.plot(sq_x, sq_y, marker, label=f'{alg_name}', alpha=0.8)
#
#     set_plot_title(ax, 'solution quality')
#     ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
#     ax.set_xticks(n_agents_list)
#     ax.set_xlabel('N agents', labelpad=-1)
#     # ax.set_ylabel('sol_quality')
#     set_legend(ax)


# def plot_runtime_cactus(ax, info):
#     ax.cla()
#     statistics_dict = info['statistics_dict']
#     runs_per_n_agents = info['runs_per_n_agents']
#     algs_to_test_dict = info['algs_to_test_dict']
#     n_agents_list = info['n_agents_list']
#     is_json = info['is_json']
#
#     max_instances = 0
#     index = -1
#     for alg_name, (alg_func, alg_info) in algs_to_test_dict.items():
#         index += 1
#         rt_y = get_list_runtime(statistics_dict, alg_name, n_agents_list, 'runtime', runs_per_n_agents, is_json)
#         rt_y.sort()
#         rt_x = list(range(len(rt_y)))
#         max_instances = max(max_instances, len(rt_x))
#
#         # dist_runtime
#         if alg_info['dist']:
#             it_y = get_list_runtime(statistics_dict, alg_name, n_agents_list, 'dist_runtime', runs_per_n_agents,
#                                     is_json)
#             it_y.sort()
#             it_x = list(range(len(it_y)))
#             max_instances = max(max_instances, len(it_x))
#             sub_plot_cactus_dist_lines(ax, index, it_x, it_y, alg_name, alg_info)
#         else:
#             sub_plot_cactus_big_lines(ax, index, rt_x, rt_y, alg_name, alg_info)
#
#
#     ax.set_xlim([0, max_instances + 2])
#     # ax.set_xticks(rt_x)
#     ax.set_xlabel('solved instances', labelpad=-1)
#     ax.set_ylabel('seconds', labelpad=-1)
#     is_log = set_log(ax)
#     set_plot_title(ax, f'runtime (cactus{" - log scale" if is_log else ""})')
#     # ax.set_ylim([1, 3000])
#     # ax.set_ylim([0, 3000])
#     set_legend(ax)


# def plot_a_star_calls_counters(ax, info):
#     ax.cla()
#     statistics_dict = info['statistics_dict']
#     runs_per_n_agents = info['runs_per_n_agents']
#     algs_to_test_dict = info['algs_to_test_dict']
#     n_agents_list = info['n_agents_list']
#     is_json = info['is_json']
#
#     max_instances = 0
#     index = -1
#     for alg_name, (alg_func, alg_info) in algs_to_test_dict.items():
#         index += 1
#         # A* calls
#         ac_y = get_list_runtime(statistics_dict, alg_name, n_agents_list, 'a_star_calls_counter', runs_per_n_agents,
#                                 is_json)
#         ac_y.sort()
#         ac_x = list(range(len(ac_y)))
#         max_instances = max(max_instances, len(ac_x))
#
#         if alg_info['dist']:
#             # get_list_a_star(statistics_dict, alg_name, n_agents_list, list_type, is_json=False)
#             acd_y = merge_agents_lists(statistics_dict, alg_name, n_agents_list, 'a_star_calls_counter_dist', is_json)
#             acd_y.sort()
#             acd_x = list(range(len(acd_y)))
#             max_instances = max(max_instances, len(acd_x))
#             sub_plot_cactus_dist_lines(ax, index, acd_x, acd_y, alg_name, alg_info)
#         else:
#             sub_plot_cactus_big_lines(ax, index, ac_x, ac_y, alg_name, alg_info)
#
#
#     ax.set_xlim([0, max_instances + 2])
#     ax.set_xlabel('solved instances', labelpad=-1)
#     is_log = set_log(ax)
#     # ax.set_ylim([0, 3e7])
#     set_plot_title(ax, f'# of A* calls (cactus{" - log scale" if is_log else ""})')
#     set_legend(ax, framealpha=0)


# def plot_n_messages(ax, info):
#     # stats_dict[alg_name][n_agents]['n_messages_per_agent'].extend(alg_info['n_messages_per_agent'])
#     ax.cla()
#     statistics_dict = info['statistics_dict']
#     runs_per_n_agents = info['runs_per_n_agents']
#     algs_to_test_dict = info['algs_to_test_dict']
#     n_agents_list = info['n_agents_list']
#     is_json = info['is_json']
#
#     showfliers = False
#     # showfliers = True
#     big_table = []
#     counter = 0
#     for alg_name, (alg_func, alg_info) in algs_to_test_dict.items():
#         line_style = f"-{get_line_or_marker(counter, 'm')}"
#         counter += 1
#         if alg_info['dist']:
#             x_list = []
#             y_list = []
#             y2_list = []
#             for n_agents in n_agents_list:
#                 x_list.append(n_agents)
#                 n_agents_number = n_agents
#                 if is_json:
#                     n_agents = str(n_agents)
#                 y_list.append(np.mean(statistics_dict[alg_name][n_agents]['n_messages']))
#                 y2_list.append(np.mean(statistics_dict[alg_name][n_agents]['m_per_step'])/n_agents_number)
#
#             if len(y_list) > 0:
#                 if 'color' in alg_info:
#                     # ax.plot(x_list, y_list, '-o', label=f'{alg_name}', alpha=0.75, color=alg_info['color'])
#                     ax.plot(x_list, y2_list, line_style, label=f'{alg_name}', alpha=0.55, color=alg_info['color'])
#                 else:
#                     ax.plot(x_list, y_list, '-o', label=f'{alg_name}', alpha=0.75)
#
#     # ax.set_ylabel('n_messages_per_agent')
#     ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
#     ax.set_xticks(n_agents_list)
#     ax.set_xlabel('N agents', labelpad=-1)
#     set_plot_title(ax, f'# of messages per step')
#     set_legend(ax)


# def plot_n_steps_iters(ax, info):
#     # n_steps, n_small_iters
#     ax.cla()
#     statistics_dict = info['statistics_dict']
#     runs_per_n_agents = info['runs_per_n_agents']
#     algs_to_test_dict = info['algs_to_test_dict']
#     n_agents_list = info['n_agents_list']
#     is_json = info['is_json']
#
#     counter = 0
#     for alg_name, (alg_func, alg_info) in algs_to_test_dict.items():
#         line_style = f"-{get_line_or_marker(counter, 'm')}"
#         counter += 1
#         if alg_info['dist']:
#             x_list = []
#             steps_list = []
#             iters_list = []
#             for n_agents in n_agents_list:
#                 x_list.append(n_agents)
#                 if is_json:
#                     n_agents = str(n_agents)
#                 steps_list.append(np.mean(statistics_dict[alg_name][n_agents]['n_steps']))
#                 iters_list.append(np.mean(statistics_dict[alg_name][n_agents]['n_small_iters']))
#
#             if len(steps_list) > 0:
#                 if 'color' in alg_info:
#                     ax.plot(x_list, steps_list, line_style, label=f'{alg_name}', alpha=0.75, color=alg_info['color'])  # steps
#                     # ax.plot(x_list, iters_list, line_style, label=f'{alg_name}', alpha=0.55, color=alg_info['color'])  # iters
#     set_plot_title(ax, f'# of steps')
#     # set_plot_title(ax, f'# of iterations')
#     ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
#     ax.set_xticks(n_agents_list)
#     ax.set_xlabel('N agents', labelpad=-1)
#     set_legend(ax)


# def plot_n_expanded_cactus(ax, info):
#     ax.cla()
#     statistics_dict = info['statistics_dict']
#     runs_per_n_agents = info['runs_per_n_agents']
#     algs_to_test_dict = info['algs_to_test_dict']
#     n_agents_list = info['n_agents_list']
#     is_json = info['is_json']
#
#     max_instances = 0
#     index = -1
#     for alg_name, (alg_func, alg_info) in algs_to_test_dict.items():
#         index += 1
#         # runtime
#         rt_y = merge_agents_lists(statistics_dict, alg_name, n_agents_list, 'n_closed_per_run', is_json)
#         rt_y.sort()
#         rt_x = list(range(len(rt_y)))
#         max_instances = max(max_instances, len(rt_x))
#
#         if alg_info['dist']:
#             l_y = merge_agents_lists(statistics_dict, alg_name, n_agents_list, 'a_star_n_closed_dist', is_json)
#             l_y.sort()
#             l_x = list(range(len(l_y)))
#             max_instances = max(max_instances, len(l_x))
#             sub_plot_cactus_dist_lines(ax, index, l_x, l_y, alg_name, alg_info)
#         else:
#             sub_plot_cactus_big_lines(ax, index, rt_x, rt_y, alg_name, alg_info)
#
#     ax.set_xlim([0, max_instances + 2])
#     # ax.set_xticks(rt_x)
#     # ax.set_ylabel('n_closed')
#     ax.set_xlabel('solved instances', labelpad=-1)
#     # ax.set_xlabel('y: N expanded nodes (cactus - log scale)')
#     is_log = set_log(ax)
#     # ax.set_ylim([0, 3e7])
#     set_plot_title(ax, f'# of expanded nodes (cactus{" - log scale" if is_log else ""})')
#     set_legend(ax)


# def plot_n_agents_conf(ax, info):
#     ax.cla()
#     statistics_dict = info['statistics_dict']
#     runs_per_n_agents = info['runs_per_n_agents']
#     algs_to_test_dict = info['algs_to_test_dict']
#     n_agents_list = info['n_agents_list']
#     is_json = info['is_json']
#
#     for alg_name, (alg_func, alg_info) in algs_to_test_dict.items():
#         l_x = []
#         l_y = []
#         for n_agents in n_agents_list:
#             if is_json:
#                 n_agents = str(n_agents)
#             if 'n_agents_conf' in statistics_dict[alg_name][n_agents]:
#                 n_agents_conf = statistics_dict[alg_name][n_agents]['n_agents_conf']
#                 if len(n_agents_conf) > 0:
#                     l_y.append(np.mean(statistics_dict[alg_name][n_agents]['n_agents_conf']))
#                     l_x.append(n_agents)
#
#         if len(l_y) > 0:
#             if 'color' in alg_info:
#                 ax.plot(l_x, l_y, '-o', label=f'{alg_name}', alpha=0.75, color=alg_info['color'])
#             else:
#                 ax.plot(l_x, l_y, '-o', label=f'{alg_name}', alpha=0.75)
#
#     ax.set_ylabel('n_agents_conf')
#     ax.set_xlim([min(n_agents_list) - 1, max(n_agents_list) + 1])
#     ax.set_xticks(n_agents_list)
#     ax.set_xlabel('N agents')
#     set_legend(ax)


# def plot_conf_per_iter(ax, info, **kwargs):
#     ax.cla()
#     statistics_dict = info['statistics_dict']
#     runs_per_n_agents = info['runs_per_n_agents']
#     algs_to_test_dict = info['algs_to_test_dict']
#     n_agents_list = info['n_agents_list']
#     is_json = info['is_json']
#
#     index = -1
#     for alg_name, (alg_func, alg_info) in algs_to_test_dict.items():
#         index += 1
#         # line = get_line_or_marker(index, 'l')
#         marker = f"-{get_line_or_marker(index, 'm')}"
#         if alg_info['dist'] and 'n_agents' in kwargs:
#             l_y = statistics_dict[alg_name][kwargs['n_agents']]['confs_per_iter']
#             l_x = list(range(len(l_y)))
#
#             if len(l_y) > 0:
#                 if 'color' in alg_info:
#                     ax.plot(l_x, l_y, marker, label=f'{alg_name}', alpha=0.75, color=alg_info['color'])
#                 else:
#                     ax.plot(l_x, l_y, marker, label=f'{alg_name}', alpha=0.75)
#
#     # ax.set_ylabel('Conflicts per Iteration')
#     # ax.set_xlim([min_x - 1, max_x + 1])
#     # ax.set_xticks(n_agents_list)
#     ax.set_xlabel('Conflicts per Iteration')
#     set_legend(ax)


# def plot_n_nei(ax, info):
#     # n_steps, n_small_iters
#     ax.cla()
#     statistics_dict = info['statistics_dict']
#     runs_per_n_agents = info['runs_per_n_agents']
#     algs_to_test_dict = info['algs_to_test_dict']
#     n_agents_list = info['n_agents_list']
#     is_json = info['is_json']
#
#     counter = 0
#     for alg_name, (alg_func, alg_info) in algs_to_test_dict.items():
#         line_style = f"-{get_line_or_marker(counter, 'm')}"
#         counter += 1
#         if alg_info['dist']:
#             x_list = []
#             nei_list = []
#             for n_agents in n_agents_list:
#                 x_list.append(n_agents)
#                 n_agents_number = n_agents
#                 if is_json:
#                     n_agents = str(n_agents)
#                 nei_list.append(np.mean(statistics_dict[alg_name][n_agents]['n_nei'])/n_agents_number)
#
#             if len(nei_list) > 0:
#                 if 'color' in alg_info:
#                     ax.plot(x_list, nei_list, line_style, label=f'{alg_name}', alpha=0.55, color=alg_info['color'])
#
#     # ax.set_ylabel('sum of neighbours', labelpad=-1)
#     set_plot_title(ax, f'# of neighbours per agent')
#     ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
#     ax.set_xticks(n_agents_list)
#     ax.set_xlabel('N agents', labelpad=-1)
#     set_legend(ax)


# def plot_avr_distance(ax, info):
#     # stats_dict[alg_name][n_agents]['n_messages_per_agent'].extend(alg_info['n_messages_per_agent'])
#     ax.cla()
#     statistics_dict = info['statistics_dict']
#     algs_to_test_dict = info['algs_to_test_dict']
#     n_agents_list = info['n_agents_list']
#     is_json = info['is_json']
#
#     counter = 0
#     for alg_name, (alg_func, alg_info) in algs_to_test_dict.items():
#         line_style = f"-{get_line_or_marker(counter, 'm')}"
#         counter += 1
#         x_list = []
#         y_list = []
#         for n_agents in n_agents_list:
#             x_list.append(n_agents)
#             n_agents_number = n_agents
#             if is_json:
#                 n_agents = str(n_agents)
#             y_list.append(np.mean(statistics_dict[alg_name][n_agents]['avr_distances']))
#
#         if len(y_list) > 0:
#             if 'color' in alg_info:
#                 # ax.plot(x_list, y_list, '-o', label=f'{alg_name}', alpha=0.75, color=alg_info['color'])
#                 ax.plot(x_list, y_list, line_style, label=f'{alg_name}', alpha=0.55, color=alg_info['color'])
#             else:
#                 ax.plot(x_list, y_list, '-o', label=f'{alg_name}', alpha=0.75)
#
#     # ax.set_ylabel('n_messages_per_agent')
#     ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
#     ax.set_xticks(n_agents_list)
#     ax.set_xlabel('N agents', labelpad=-1)
#     set_plot_title(ax, f'Avr. Distances From Neighbours')
#     set_legend(ax)


# def plot_avr_nearby_agents(ax, info):
#     # stats_dict[alg_name][n_agents]['n_messages_per_agent'].extend(alg_info['n_messages_per_agent'])
#     ax.cla()
#     statistics_dict = info['statistics_dict']
#     algs_to_test_dict = info['algs_to_test_dict']
#     n_agents_list = info['n_agents_list']
#     is_json = info['is_json']
#
#     counter = 0
#     for alg_name, (alg_func, alg_info) in algs_to_test_dict.items():
#         line_style = f"-{get_line_or_marker(counter, 'm')}"
#         counter += 1
#         x_list = []
#         y_list = []
#         for n_agents in n_agents_list:
#             x_list.append(n_agents)
#             n_agents_number = n_agents
#             if is_json:
#                 n_agents = str(n_agents)
#             y_list.append(np.mean(statistics_dict[alg_name][n_agents]['avr_nearby_agents']))
#
#         if len(y_list) > 0:
#             if 'color' in alg_info:
#                 # ax.plot(x_list, y_list, '-o', label=f'{alg_name}', alpha=0.75, color=alg_info['color'])
#                 ax.plot(x_list, y_list, line_style, label=f'{alg_name}', alpha=0.55, color=alg_info['color'])
#             else:
#                 ax.plot(x_list, y_list, '-o', label=f'{alg_name}', alpha=0.75)
#
#     # ax.set_ylabel('n_messages_per_agent')
#     ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
#     ax.set_xticks(n_agents_list)
#     ax.set_xlabel('N agents', labelpad=-1)
#     set_plot_title(ax, f'Avr. N of Nearby Agents')
#     set_legend(ax)



