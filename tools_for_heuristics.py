from globals import *
import concurrent.futures
from tools_for_graph import ListNodes
from tools_for_graph import build_graph_nodes, get_dims_from_pic


def load_h_dict(possible_dir):
    if os.path.exists(possible_dir):
        # Opening JSON file
        with open(possible_dir, 'r') as openfile:
            # Reading from json file
            h_dict = json.load(openfile)
            for k, v in h_dict.items():
                h_dict[k] = np.array(v)
            return h_dict
    return None


def save_h_dict(h_dict, possible_dir):
    for k, v in h_dict.items():
        h_dict[k] = v.tolist()
    json_object = json.dumps(h_dict, indent=2)
    with open(possible_dir, "w") as outfile:
        outfile.write(json_object)


def dist_heuristic(from_node, to_node):
    return np.abs(from_node.x - to_node.x) + np.abs(from_node.y - to_node.y)
    # return np.sqrt((from_node.x - to_node.x) ** 2 + (from_node.y - to_node.y) ** 2)


def h_func_creator(h_dict):
    def h_func(from_node, to_node):
        if to_node.xy_name in h_dict:
            h_value = h_dict[to_node.xy_name][from_node.x, from_node.y]
            if h_value > 0:
                return h_value
        return np.abs(from_node.x - to_node.x) + np.abs(from_node.y - to_node.y)
        # return np.sqrt((from_node.x - to_node.x) ** 2 + (from_node.y - to_node.y) ** 2)
    return h_func


def get_node(successor_xy_name, node_current, nodes_dict):
    if node_current.xy_name == successor_xy_name:
        return None
    return nodes_dict[successor_xy_name]
    # for node in nodes:
    #     if node.xy_name == successor_xy_name and node_current.xy_name != successor_xy_name:
    #         return node
    # return None


def parallel_update_h_table(node, nodes, map_dim, h_dict, node_index, **kwargs):
    print(f'[HEURISTIC]: Thread {node_index} started.')
    h_table = build_heuristic_for_one_target(node, nodes, map_dim, **kwargs)
    h_dict[node.xy_name] = h_table
    print(f'[HEURISTIC]: Thread {node_index} finished.')


def parallel_build_heuristic_for_entire_map(nodes, map_dim, **kwargs):
    print('Started to build heuristic...')
    path = kwargs['path']
    possible_dir = f"{path}/h_dict_of_{kwargs['img_dir'][:-4]}.json"

    # if there is one
    h_dict = load_h_dict(possible_dir)
    if h_dict is not None:
        print(f'\nFinished to build heuristic for all nodes.')
        return h_dict

    # else, create one
    h_dict = {}
    reset_nodes(nodes, nodes)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as executor:
        for node_index, node in enumerate(nodes):
            executor.submit(parallel_update_h_table, node, nodes, map_dim, h_dict, node_index, **kwargs)
    save_h_dict(h_dict, possible_dir)
    print(f'\nFinished to build heuristic for all nodes.')
    return h_dict


# def parallel_build_heuristic_for_multiple_targets(target_nodes, nodes, map_dim, **kwargs):
#     print('Started to build heuristic...')
#     h_dict = {}
#     reset_nodes(nodes, target_nodes)
#     with concurrent.futures.ThreadPoolExecutor(max_workers=len(target_nodes)) as executor:
#         for node_index, node in enumerate(target_nodes):
#             executor.submit(parallel_update_h_table, node, nodes, map_dim, h_dict, node_index, **kwargs)
#
#     print(f'\nFinished to build heuristic for target nodes.')
#     return h_dict


def build_heuristic_for_multiple_targets(target_nodes, nodes, map_dim, **kwargs):
    print('Started to build heuristic...')
    h_dict = {}
    reset_nodes(nodes, target_nodes)
    iteration = 0
    for node in target_nodes:
        h_table = build_heuristic_for_one_target(node, nodes, map_dim, **kwargs)
        h_dict[node.xy_name] = h_table

        print(f'\nFinished to build heuristic for node {iteration}.')
        iteration += 1
    return h_dict


def reset_nodes(nodes, target_nodes=None):
    _ = [node.reset(target_nodes) for node in nodes]


def build_heuristic_for_one_target(target_node, nodes, map_dim, **kwargs):
    # print('Started to build heuristic...')
    copy_nodes = nodes
    nodes_dict = {node.xy_name: node for node in copy_nodes}
    target_name = target_node.xy_name
    target_node = nodes_dict[target_name]
    # target_node = [node for node in copy_nodes if node.xy_name == target_node.xy_name][0]
    # open_list = []
    # close_list = []
    open_nodes = ListNodes(target_name=target_node.xy_name)
    closed_nodes = ListNodes(target_name=target_node.xy_name)
    # open_list.append(target_node)
    open_nodes.add(target_node)
    iteration = 0
    # while len(open_list) > 0:
    while len(open_nodes) > 0:
        iteration += 1
        # node_current = get_node_from_open(open_list, target_name)
        node_current = open_nodes.pop()
        # if node_current.xy_name == '30_12':
        #     print()
        for successor_xy_name in node_current.neighbours:
            node_successor = get_node(successor_xy_name, node_current, nodes_dict)
            if node_successor:
                successor_current_g = node_current.g_dict[target_name] + 1  # h(now, next)

                # INSIDE OPEN LIST
                if node_successor.xy_name in open_nodes.dict:
                    if node_successor.g_dict[target_name] <= successor_current_g:
                        continue
                    open_nodes.remove(node_successor)
                    node_successor.g_dict[target_name] = successor_current_g
                    node_successor.parent = node_current
                    open_nodes.add(node_successor)

                # INSIDE CLOSED LIST
                elif node_successor.xy_name in closed_nodes.dict:
                    if node_successor.g_dict[target_name] <= successor_current_g:
                        continue
                    closed_nodes.remove(node_successor)
                    node_successor.g_dict[target_name] = successor_current_g
                    node_successor.parent = node_current
                    open_nodes.add(node_successor)

                # NOT IN CLOSED AND NOT IN OPEN LISTS
                else:
                    node_successor.g_dict[target_name] = successor_current_g
                    node_successor.parent = node_current
                    open_nodes.add(node_successor)

                # node_successor.g_dict[target_name] = successor_current_g
                # node_successor.parent = node_current

        # open_nodes.remove(node_current, target_name=target_node.xy_name)
        closed_nodes.add(node_current)

        # if plotter and middle_plot and iteration % 1000 == 0:
        #     plotter.plot_lists(open_list=open_nodes.get_nodes_list(),
        #                        closed_list=closed_nodes.get_nodes_list(), start=target_node, nodes=copy_nodes)
        if iteration % 100 == 0:
            print(f'\riter: {iteration}', end='')

    # if plotter and middle_plot:
    #     plotter.plot_lists(open_list=open_nodes.get_nodes_list(),
    #                        closed_list=closed_nodes.get_nodes_list(), start=target_node, nodes=copy_nodes)

    h_table = np.zeros(map_dim)
    for node in copy_nodes:
        h_table[node.x, node.y] = node.g_dict[target_name]
    # h_dict = {target_node.xy_name: h_table}
    # print(f'\rFinished to build heuristic at iter {iteration}.')
    return h_table


def main():
    # img_dir = 'empty-32-32.map'  # 32-32
    # img_dir = 'random-32-32-10.map'  # 32-32          | LNS | Up to 400 agents with w=5, h=2, lim=1min.
    # img_dir = 'random-32-32-20.map'  # 32-32
    # img_dir = 'room-32-32-4.map'  # 32-32
    # img_dir = 'maze-32-32-2.map'  # 32-32
    # img_dir = 'den312d.map'  # 65-81

    # img_dir='empty-48-48.map'  # 48-48              | Up to 580 agents with h=w=5, lim=10sec.
    # img_dir = 'random-64-64-10.map'  # 64-64          | Up to 580 agents with h=w=10, lim=10sec.
    # img_dir='warehouse-10-20-10-2-1.map'  # 63-161  | Up to 330 agents with h=w=30, lim=10sec.
    img_dir='ht_chantry.map'  # 162-141             | Up to 330 agents with h=w=30, lim=10sec.

    map_dim = get_dims_from_pic(img_dir=img_dir, path='maps')
    nodes, nodes_dict, img_np = build_graph_nodes(img_dir=img_dir, path='maps', show_map=False)

    h_dict = parallel_build_heuristic_for_entire_map(nodes, map_dim, img_dir=img_dir, path='logs_for_heuristics')
    pprint(f'{len(h_dict)=}')
    # plotter = Plotter(map_dim=map_dim, subplot_rows=1, subplot_cols=3)
    # plt.show()


if __name__ == '__main__':
    main()

