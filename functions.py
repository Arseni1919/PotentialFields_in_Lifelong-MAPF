from globals import *


def set_h_and_w(obj):
    if 'h' not in obj.params:
        return None, None
    else:
        return obj.params['h'], obj.params['w']


def set_pf_weight(obj):
    if 'pf_weight' not in obj.params:
        return 0
    else:
        return obj.params['pf_weight']


def use_profiler(save_dir):
    def decorator(func):
        def inner1(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            # getting the returned value
            returned_value = func(*args, **kwargs)
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats('cumtime')
            stats.dump_stats(save_dir)
            # returning the value to the original frame
            return returned_value
        return inner1
    return decorator


def set_seed(random_seed_bool, i_seed=1):
    if random_seed_bool:
        seed = random.randint(0, 1000)
    else:
        seed = i_seed
    random.seed(seed)
    np.random.seed(seed)
    print(f'[SEED]: --- {seed} ---')


def check_if_nei_pos(agents):
    for agent in agents:
        if agent.curr_node.xy_name not in agent.prev_node.neighbours:
            raise RuntimeError('wow wow wow! Not nei pos!')


def two_plans_have_no_confs(plan1, plan2):
    min_len = min(len(plan1), len(plan2))
    prev1 = plan1[0]
    prev2 = plan2[0]
    for i in range(min_len):
        vertex1 = plan1[i]
        vertex2 = plan2[i]
        if i > 0:
            edge1 = (prev1.xy_name, vertex1.xy_name)
            edge2 = (vertex2.xy_name, prev2.xy_name)
            if edge1 == edge2:
                return False
        prev1 = vertex1
        prev2 = vertex2
        if vertex1.xy_name == vertex2.xy_name:
            return False
    return True


def check_actions_if_vc(agents, actions):
    for agent1, agent2 in combinations(agents, 2):
        vertex1 = actions[agent1.name]
        vertex2 = actions[agent2.name]
        if vertex1 == vertex2:
            raise RuntimeError(f'vertex collision: {agent1.name} and {agent2.name} in {vertex1}')
            # print(f'\nvertex collision: {agent1.name} and {agent2.name} in {vertex1}')


def check_if_vc(agents):
    for agent1, agent2 in combinations(agents, 2):
        vertex1 = agent1.curr_node.xy_name
        vertex2 = agent2.curr_node.xy_name
        if vertex1 == vertex2:
            raise RuntimeError(f'vertex collision: {agent1.name} and {agent2.name} in {vertex1}')
            # print(f'\nvertex collision: {agent1.name} and {agent2.name} in {vertex1}')


def check_actions_if_ec(agents, actions):
    for agent1, agent2 in combinations(agents, 2):
        edge1 = (agent1.curr_node.xy_name, actions[agent1.name])
        edge2 = (actions[agent2.name], agent2.curr_node.xy_name)
        if edge1 == edge2:
            raise RuntimeError(f'edge collision: {agent1.name} and {agent2.name} in {edge1}')
            # print(f'\nedge collision: {agent1.name} and {agent2.name} in {edge1}')


def check_if_ec(agents):
    for agent1, agent2 in combinations(agents, 2):
        edge1 = (agent1.prev_node.xy_name, agent1.curr_node.xy_name)
        edge2 = (agent2.curr_node.xy_name, agent2.prev_node.xy_name)
        if edge1 == edge2:
            raise RuntimeError(f'edge collision: {agent1.name} and {agent2.name} in {edge1}')
            # print(f'\nedge collision: {agent1.name} and {agent2.name} in {edge1}')


def create_sub_results(h_agents):
    # sub results
    sub_results = {}
    for agent in h_agents:
        # h_plan = agent.plan
        h_plan = [agent.curr_node]
        h_plan.extend(agent.plan)
        sub_results[agent.name] = h_plan
    # sub_results = {agent.name: agent.plan for agent in h_agents}
    return sub_results


def build_constraints(nodes, other_paths):
    v_constr_dict = {node.xy_name: [] for node in nodes}
    e_constr_dict = {node.xy_name: [] for node in nodes}
    perm_constr_dict = {node.xy_name: [] for node in nodes}

    for agent_name, path in other_paths.items():
        if len(path) > 0:
            final_node = path[-1]
            final_t = len(path) - 1
            perm_constr_dict[final_node.xy_name].append(final_t)
            perm_constr_dict[final_node.xy_name] = [max(perm_constr_dict[final_node.xy_name])]

            prev_node = path[0]
            for t, node in enumerate(path):
                # vertex
                v_constr_dict[f'{node.x}_{node.y}'].append(t)
                # edge
                if prev_node.xy_name != node.xy_name:
                    e_constr_dict[f'{prev_node.x}_{prev_node.y}'].append((node.x, node.y, t))
                prev_node = node
    return v_constr_dict, e_constr_dict, perm_constr_dict


@lru_cache(maxsize=128)
def manhattan_distance_nodes(node1, node2):
    return abs(node1.x-node2.x) + abs(node1.y-node2.y)


def get_nei_nodes(curr_node, nei_r, nodes_dict):
    nei_nodes_dict = {}
    open_list = [curr_node]
    while len(open_list) > 0:
        i_node = open_list.pop()
        i_node_distance = euclidean_distance_nodes(curr_node, i_node)
        if i_node_distance <= nei_r:
            nei_nodes_dict[i_node.xy_name] = i_node
            for node_nei_name in i_node.neighbours:
                if node_nei_name not in nei_nodes_dict:
                    open_list.append(nodes_dict[node_nei_name])
    nei_nodes = list(nei_nodes_dict.values())
    return nei_nodes, nei_nodes_dict


@lru_cache(maxsize=128)
def euclidean_distance_nodes(node1, node2):
    # p = [node1.x, node1.y]
    # q = [node2.x, node2.y]
    return math.dist([node1.x, node1.y], [node2.x, node2.y])
    # return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

