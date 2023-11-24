from tools_for_graph import *
from tools_for_plotting import *
from tools_for_heuristics import *
from functions import *


def check_future_constr(node_current, v_constr_dict, e_constr_dict, perm_constr_dict, ignore_dict, start):
    # NO NEED FOR wasted waiting
    if node_current.xy_name in ignore_dict:
        return False
    new_t = node_current.t + 1
    time_window = node_current.t - start.t
    if time_window < 20:
        return False

    future_constr = False
    for nei_xy_name in node_current.neighbours:
        if v_constr_dict and new_t in v_constr_dict[nei_xy_name]:
            future_constr = True
            break
        if e_constr_dict and (node_current.x, node_current.y, new_t) in e_constr_dict[nei_xy_name]:
            future_constr = True
            break
        if perm_constr_dict:
            if len(perm_constr_dict[nei_xy_name]) > 0:
                if new_t >= perm_constr_dict[nei_xy_name][0]:
                    future_constr = True
                    break
    return future_constr


def get_max_final(perm_constr_dict):
    if perm_constr_dict:
        final_list = [v[0] for k, v in perm_constr_dict.items() if len(v) > 0]
        max_final_time = max(final_list) if len(final_list) > 0 else 1
        return max_final_time
    return 1


def get_node(successor_xy_name, node_current, nodes, nodes_dict, open_nodes, closed_nodes, v_constr_dict, e_constr_dict,
             perm_constr_dict, max_final_time, **kwargs):
    xyt_problem = kwargs['xyt_problem']
    if not xyt_problem:
        if successor_xy_name == node_current.xy_name:
            return None, ''
    new_t = node_current.t + 1

    if v_constr_dict:
        if new_t in v_constr_dict[successor_xy_name]:
            return None, ''

    if e_constr_dict:
        if (node_current.x, node_current.y, new_t) in e_constr_dict[successor_xy_name]:
            return None, ''

    if perm_constr_dict:
        if len(perm_constr_dict[successor_xy_name]) > 0:
            if len(perm_constr_dict[successor_xy_name]) != 1:
                raise RuntimeError('len(perm_constr_dict[successor_xy_name]) != 1')
            final_time = perm_constr_dict[successor_xy_name][0]
            if new_t >= final_time:
                return None, ''

    if max_final_time:
        if node_current.t >= max_final_time:
            new_t = max_final_time + 1

    new_ID = f'{successor_xy_name}_{new_t}'
    if new_ID in open_nodes.dict:
        return open_nodes.dict[new_ID], 'open_nodes'
    if new_ID in closed_nodes.dict:
        return closed_nodes.dict[new_ID], 'closed_nodes'

    node = nodes_dict[successor_xy_name]
    return Node(x=node.x, y=node.y, t=new_t, neighbours=node.neighbours), 'new'


def do_reset_for_one_node(curr_node):
    curr_node.reset()


def reset_nodes(start, goal, nodes, **kwargs):
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     executor.map(do_reset_for_one_node, nodes)
    # _ = [do_reset_for_one_node(node) for node in nodes]
    _ = [node.reset() for node in nodes]
    start.reset(**kwargs)
    return start, goal, nodes


def k_time_check(node_current, **kwargs):
    if 'k_time' in kwargs and kwargs['k_time'] is not None:
        return node_current.t >= kwargs['k_time'] - 1
    return False
    # return node_current.t >= kwargs['k_time'] if 'k_time' in kwargs else False


def build_successor_current_g(nei_pfs, node_successor, successor_current_t, node_current):
    if nei_pfs is not None and successor_current_t < nei_pfs.shape[2]:
        mag_cost = nei_pfs[node_successor.x, node_successor.y, successor_current_t]
        successor_current_g = node_current.g + 1 + mag_cost
    else:
        successor_current_g = successor_current_t
    return successor_current_g


def a_star_xyt(start, goal, nodes, h_func,
               v_constr_dict=None, e_constr_dict=None, perm_constr_dict=None,
               plotter=None, middle_plot=False,
               iter_limit=1e100, nodes_dict=None, **kwargs):
    """
    new_t in v_constr_dict[successor_xy_name]
    """
    start_time = time.time()

    # magnets part
    nei_pfs = kwargs['nei_pfs'] if 'nei_pfs' in kwargs else None

    # start, goal, nodes = deepcopy_nodes(start, goal, nodes)  # heavy!
    start, goal, nodes = reset_nodes(start, goal, nodes, **kwargs)
    print('\rStarted A*...', end='')
    open_nodes = ListNodes()
    closed_nodes = ListNodes()
    node_current = start
    node_current.h = h_func(node_current, goal)
    open_nodes.add(node_current)
    max_final_time = get_max_final(perm_constr_dict)
    future_constr = False
    succeeded = False
    iteration = 0
    while len(open_nodes) > 0:
        iteration += 1
        if iteration > iter_limit:
            print(f'\n[ERROR]: out of iterations (more than {iteration})')
            return None, {'runtime': time.time() - start_time, 'n_open': len(open_nodes.heap_list), 'n_closed': len(closed_nodes.heap_list)}
        node_current = open_nodes.pop()

        time_check = k_time_check(node_current, **kwargs)
        if node_current.xy_name == goal.xy_name or time_check:
            # break
            succeeded = True
            # if there is a future constraint of a goal
            if len(v_constr_dict[node_current.xy_name]) > 0:
                # we will take the maximum time out of all constraints
                max_t = max(v_constr_dict[node_current.xy_name])
                # and compare to the current time
                # if it is greater, we will continue to expand the search tree
                if node_current.t > max_t:
                    # otherwise break
                    break
            else:
                break
        succeeded = False
        for successor_xy_name in node_current.neighbours:
            node_successor, node_successor_status = get_node(
                successor_xy_name, node_current, nodes, nodes_dict, open_nodes, closed_nodes,
                v_constr_dict, e_constr_dict, perm_constr_dict, max_final_time, **kwargs
            )

            if node_successor is None:
                continue

            successor_current_t = node_current.t + 1  # h(now, next)
            successor_current_g = build_successor_current_g(nei_pfs, node_successor, successor_current_t, node_current)

            # INSIDE OPEN LIST
            if node_successor_status == 'open_nodes':
                if node_successor.t <= successor_current_t:
                    continue
                open_nodes.remove(node_successor)

            # INSIDE CLOSED LIST
            elif node_successor_status == 'closed_nodes':
                if node_successor.t <= successor_current_t:
                    continue
                closed_nodes.remove(node_successor)

            # NOT IN CLOSED AND NOT IN OPEN LISTS
            else:
                node_successor.h = h_func(node_successor, goal)
            node_successor.t = successor_current_t
            node_successor.g = successor_current_g
            node_successor.parent = node_current
            open_nodes.add(node_successor)

        # open_nodes.remove(node_current)
        closed_nodes.add(node_current)

        # if plotter and middle_plot and iteration % 10 == 0:
        #     plotter.plot_lists(open_list=open_nodes.get_nodes_list(),
        #                        closed_list=closed_nodes.get_nodes_list(),
        #                        start=start, goal=goal, nodes=nodes, a_star_run=True)
        print(f'\r(a_star) iter: {iteration}, closed: {len(closed_nodes.heap_list)}', end='')

    path = None
    if succeeded:
        path = []
        while node_current is not None:
            path.append(node_current)
            node_current = node_current.parent
        path.reverse()

    if plotter and middle_plot:
        plotter.plot_lists(open_list=open_nodes.get_nodes_list(),
                           closed_list=closed_nodes.get_nodes_list(),
                           start=start, goal=goal, path=path, nodes=nodes, a_star_run=True,
                           agent_name=kwargs['agent_name'])
    # print('\rFinished A*.', end='')
    # if path is None:
    #     print()
    return path, {'runtime': time.time() - start_time, 'n_open': len(open_nodes.heap_list), 'n_closed': len(closed_nodes.heap_list), 'future_constr': future_constr}


def try_a_map_from_pic():
    # img_dir = 'lak109d.png'
    # img_dir = '19_20_warehouse.png'
    # img_dir = 'den101d.png'
    # img_dir = 'rmtst.png'
    # img_dir = 'lak505d.png'
    # img_dir = 'lak503d.png'
    # img_dir = 'ost003d.png'
    # img_dir = 'brc202d.png'
    # img_dir = 'den520d.png'
    img_dir = 'warehouse-10-20-10-2-1.png'
    map_dim = get_dims_from_pic(img_dir=img_dir, path='../maps')
    nodes, nodes_dict, img_np = build_graph_nodes(img_dir=img_dir, path='../maps', show_map=False)
    # ------------------------- #
    # x_start, y_start = 97, 99
    # x_goal, y_goal = 38, 89
    # node_start = [node for node in nodes if node.x == x_start and node.y == y_start][0]
    # node_goal = [node for node in nodes if node.x == x_goal and node.y == y_goal][0]
    # ------------------------- #
    # node_start = nodes[100]
    # node_goal = nodes[-1]
    # ------------------------- #
    node_start = random.choice(nodes)
    node_goal = random.choice(nodes)
    print(f'start: {node_start.x}, {node_start.y} -> goal: {node_goal.x}, {node_goal.y}')
    # ------------------------- #
    # ------------------------- #
    plotter = Plotter(map_dim=map_dim, subplot_rows=1, subplot_cols=3)
    # plotter = None
    # ------------------------- #
    # ------------------------- #
    h_dict = build_heuristic_for_multiple_targets([node_goal], nodes, map_dim, plotter=plotter, middle_plot=False)
    h_func = h_func_creator(h_dict)
    # ------------------------- #
    # h_func = dist_heuristic
    # ------------------------- #
    # ------------------------- #
    # constraint_dict = None
    v_constr_dict = {node.xy_name: [] for node in nodes}
    # v_constr_dict = {'30_12': [69], '29_12': [68, 69]}
    perm_constr_dict = {node.xy_name: [] for node in nodes}
    perm_constr_dict['74_9'].append(10)
    # ------------------------- #
    # ------------------------- #
    # result = a_star(start=node_start, goal=node_goal, nodes=nodes, h_func=h_func, plotter=plotter, middle_plot=False)
    profiler.enable()
    result, info = a_star_xyt(start=node_start, goal=node_goal, nodes=nodes, h_func=h_func,
                              v_constr_dict=v_constr_dict, perm_constr_dict=perm_constr_dict,
                              plotter=plotter, middle_plot=True, nodes_dict=nodes_dict,
                              )
    profiler.disable()
    if result:
        print('The result is:', *[node.xy_name for node in result], sep='->')
        print('The result is:', *[node.ID for node in result], sep='->')
    # ------------------------- #
    # ------------------------- #
    plt.show()
    # plt.close()


if __name__ == '__main__':
    # random_seed = True
    random_seed = False
    seed = random.choice(range(1000)) if random_seed else 121
    random.seed(seed)
    np.random.seed(seed)
    print(f'SEED: {seed}')
    profiler = cProfile.Profile()
    # profiler.enable()
    try_a_map_from_pic()
    # profiler.disable()
    # stats.print_stats()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.dump_stats('../stats/results_a_star.pstat')

# def deepcopy_nodes(start, goal, nodes):
#     copy_nodes_dict = {node.xy_name: copy.deepcopy(node) for node in nodes}
#     copy_start = copy_nodes_dict[start.xy_name]
#     copy_goal = copy_nodes_dict[goal.xy_name]
#     copy_nodes = heap_list(copy_nodes_dict.values())
#     return copy_start, copy_goal, copy_nodes

# for open_node in open_list:
#     if open_node.ID == new_ID:
#         return open_node
# for closed_node in close_list:
#     if closed_node.ID == new_ID:
#         return closed_node


# def get_node_from_open(open_nodes):
#     # v_list = open_list
#     # f_dict = {}
#     # f_vals_list = []
#     # for node in open_nodes.heap_list:
#     #     curr_f = node.f()
#     #     f_vals_list.append(curr_f)
#     #     if curr_f not in f_dict:
#     #         f_dict[curr_f] = []
#     #     f_dict[curr_f].append(node)
#     #
#     # smallest_f_nodes = f_dict[min(f_vals_list)]
#     #
#     # h_dict = {}
#     # h_vals_list = []
#     # for node in smallest_f_nodes:
#     #     curr_h = node.h
#     #     h_vals_list.append(curr_h)
#     #     if curr_h not in h_dict:
#     #         h_dict[curr_h] = []
#     #     h_dict[curr_h].append(node)
#     #
#     # smallest_h_from_smallest_f_nodes = h_dict[min(h_vals_list)]
#     # next_node = random.choice(smallest_h_from_smallest_f_nodes)
#
#     next_node = open_nodes.pop()
#     return next_node


# NO NEED FOR wasted waiting
# if 'a_star_mode' in kwargs and kwargs['a_star_mode'] == 'fast':
#     if successor_xy_name == node_current.xy_name:
#         no_constraints = True
#         for nei_xy_name in node_current.neighbours:
#             if v_constr_dict and new_t in v_constr_dict[nei_xy_name]:
#                 no_constraints = False
#                 break
#
#             if no_constraints and e_constr_dict and (node_current.x, node_current.y, new_t) in e_constr_dict[nei_xy_name]:
#                 no_constraints = False
#                 break
#         if no_constraints:
#             return None, 'future_constr'

# def main():
#     nodes = [
#         Node(x=1, y=1, neighbours=[]),
#         Node(x=1, y=2, neighbours=[]),
#         Node(x=1, y=3, neighbours=[]),
#         Node(x=1, y=4, neighbours=[]),
#         Node(x=2, y=1, neighbours=[]),
#         Node(x=2, y=2, neighbours=[]),
#         Node(x=2, y=3, neighbours=[]),
#         Node(x=2, y=4, neighbours=[]),
#         Node(x=3, y=1, neighbours=[]),
#         Node(x=3, y=2, neighbours=[]),
#         Node(x=3, y=3, neighbours=[]),
#         Node(x=3, y=4, neighbours=[]),
#         Node(x=4, y=1, neighbours=[]),
#         Node(x=4, y=2, neighbours=[]),
#         Node(x=4, y=3, neighbours=[]),
#         Node(x=4, y=4, neighbours=[]),
#     ]
#     make_neighbours(nodes)
#     node_start = nodes[0]
#     node_goal = nodes[-1]
#     plotter = Plotter(map_dim=(5, 5), subplot_rows=1, subplot_cols=3)
#     result = a_star_xyt(start=node_start, goal=node_goal, nodes=nodes, h_func=dist_heuristic, plotter=plotter,
#                         middle_plot=True)
#
#     plt.show()
#     print(result)
#     plt.close()