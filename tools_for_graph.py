from functions import *


class Node:
    def __init__(self, x, y, t=0, neighbours=None, new_ID=None):
        if new_ID:
            self.ID = new_ID
        else:
            self.ID = f'{x}_{y}_{t}'
        self.xy_name = f'{x}_{y}'
        self.x = x
        self.y = y
        self.t = t
        if neighbours is None:
            self.neighbours = []
        else:
            self.neighbours = neighbours
        # self.neighbours = neighbours

        self.h = 0
        self.g = t
        self.parent = None
        self.g_dict = {}

    def f(self):
        # return self.t + self.h
        return self.g + self.h

    def reset(self, target_nodes=None, **kwargs):
        if 'start_time' in kwargs:
            self.t = kwargs['start_time']
        else:
            self.t = 0
        self.h = 0
        self.g = self.t
        self.ID = f'{self.x}_{self.y}_{self.t}'
        self.parent = None
        if target_nodes is not None:
            self.g_dict = {target_node.xy_name: 0 for target_node in target_nodes}
        else:
            self.g_dict = {}


class ListNodes:
    def __init__(self, target_name=None):
        self.heap_list = []
        # self.nodes_list = []
        self.dict = {}
        self.h_func_bool = False
        if target_name:
            self.h_func_bool = True
            self.target_name = target_name

    def __len__(self):
        return len(self.heap_list)

    def remove(self, node):
        if self.h_func_bool:
            self.heap_list.remove((node.g_dict[self.target_name], node.xy_name))
            del self.dict[node.xy_name]
        else:
            if node.ID not in self.dict:
                raise RuntimeError('node.ID not in self.dict')
            self.heap_list.remove(((node.f(), node.h), node.ID))
            del self.dict[node.ID]
        # self.nodes_list.remove(node)

    def add(self, node):
        if self.h_func_bool:
            heapq.heappush(self.heap_list, (node.g_dict[self.target_name], node.xy_name))
            self.dict[node.xy_name] = node
        else:
            heapq.heappush(self.heap_list, ((node.f(), node.h), node.ID))
            self.dict[node.ID] = node
        # self.nodes_list.append(node)

    def pop(self):
        heap_tuple = heapq.heappop(self.heap_list)
        node = self.dict[heap_tuple[1]]
        if self.h_func_bool:
            del self.dict[node.xy_name]
        else:
            del self.dict[node.ID]
        # self.nodes_list.remove(node)
        return node

    def get(self, ID):
        return self.dict[ID]

    def get_nodes_list(self):
        return [self.dict[item[1]] for item in self.heap_list]


def get_dims_from_pic(img_dir, path='maps'):
    with open(f'{path}/{img_dir}') as f:
        lines = f.readlines()
        height = int(re.search(r'\d+', lines[1]).group())
        width = int(re.search(r'\d+', lines[2]).group())
    return height, width


def get_np_from_dot_map(img_dir, path='maps'):
    with open(f'{path}/{img_dir}') as f:
        lines = f.readlines()
        height, width = get_dims_from_pic(img_dir, path)
        img_np = np.zeros((height, width))
        for height_index, line in enumerate(lines[4:]):
            for width_index, curr_str in enumerate(line):
                if curr_str == '.':
                    img_np[height_index, width_index] = 1
        return img_np, (height, width)


def distance_nodes(node1, node2, h_func: dict = None):
    if h_func is None:
        # print('regular distance')
        return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)
    else:
        heuristic_dist = h_func[node1.x][node1.y][node2.x][node2.y]
        # direct_dist = np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)
        return heuristic_dist


def set_nei(name_1, name_2, nodes_dict):
    if name_1 in nodes_dict and name_2 in nodes_dict and name_1 != name_2:
        node1 = nodes_dict[name_1]
        node2 = nodes_dict[name_2]
        dist = distance_nodes(node1, node2)
        if dist == 1:
            node1.neighbours.append(node2.xy_name)
            node2.neighbours.append(node1.xy_name)


def make_self_neighbour(nodes):
    for node_1 in nodes:
        node_1.neighbours.append(node_1.xy_name)


def build_graph_from_np(img_np, show_map=False):
    # 0 - wall, 1 - free space
    nodes = []
    nodes_dict = {}

    x_size, y_size = img_np.shape
    # CREATE NODES
    for i_x in range(x_size):
        for i_y in range(y_size):
            if img_np[i_x, i_y] == 1:
                node = Node(i_x, i_y)
                nodes.append(node)
                nodes_dict[node.xy_name] = node

    # CREATE NEIGHBOURS
    # make_neighbours(nodes)

    name_1, name_2 = '', ''
    for i_x in range(x_size):
        for i_y in range(y_size):
            name_2 = f'{i_x}_{i_y}'
            set_nei(name_1, name_2, nodes_dict)
            name_1 = name_2

    print('finished rows')

    for i_y in range(y_size):
        for i_x in range(x_size):
            name_2 = f'{i_x}_{i_y}'
            set_nei(name_1, name_2, nodes_dict)
            name_1 = name_2
    make_self_neighbour(nodes)
    print('finished columns')

    if show_map:
        plt.imshow(img_np, cmap='gray', origin='lower')
        plt.show()
        # plt.pause(1)
        # plt.close()

    return nodes, nodes_dict, img_np


def build_graph_nodes(img_dir, path='maps', show_map=False):
    print('Start to build_graph_from_png...')
    img_np, (height, width) = get_np_from_dot_map(img_dir, path)
    return build_graph_from_np(img_np, show_map)


def main():
    # img_dir = 'empty-32-32.map'  # 32-32
    # img_dir = 'random-32-32-10.map'  # 32-32          | LNS | Up to 400 agents with w=5, h=2, lim=1min.
    # img_dir = 'random-32-32-20.map'  # 32-32
    # img_dir = 'room-32-32-4.map'  # 32-32
    img_dir = 'maze-32-32-2.map'  # 32-32
    # img_dir = 'den312d.map'  # 65-81
    # img_dir = 'room-64-64-8.map'  # 64-64
    # img_dir = 'warehouse-10-20-10-2-1.map'  # 63-161
    # img_dir = 'warehouse-10-20-10-2-2.map'  # 84-170
    # img_dir = 'warehouse-20-40-10-2-1.map'  # 123-321
    # img_dir = 'ht_chantry.map'  # 141-162
    # img_dir = 'lt_gallowstemplar_n.map'  # 180-251
    # img_dir = 'lak303d.map'  # 194-194
    # img_dir = 'warehouse-20-40-10-2-2.map'  # 164-340
    # img_dir = 'Berlin_1_256.map'  # 256-256
    # img_dir = 'den520d.map'  # 257-256
    # img_dir = 'ht_mansion_n.map'  # 270-133
    # img_dir = 'brc202d.map'  # 481-530
    nodes, nodes_dict, img_np = build_graph_nodes(img_dir=img_dir, path='maps', show_map=True)
    print()


if __name__ == '__main__':
    main()
