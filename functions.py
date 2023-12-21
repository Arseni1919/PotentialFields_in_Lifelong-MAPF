from globals import *
from concurrent.futures import ThreadPoolExecutor


def get_color(i):
    index_to_pick = i % len(color_names)
    return color_names[index_to_pick]


def check_stay_at_same_node(plan, the_node):
    for i_node in plan:
        if i_node.xy_name != the_node.xy_name:
            return False
    return True


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


def check_time_limit():
    def decorator(func):
        def inner1(*args, **kwargs):
            start_time = time.time()
            # getting the returned value
            returned_value = func(*args, **kwargs)
            end_time = time.time() - start_time
            if end_time > args[0].time_to_think_limit + 1:
                raise RuntimeError(f'[{args[0].alg_name}] crossed the time limit of {args[0].time_to_think_limit} s.')
            # returning the value to the original frame
            return returned_value
        return inner1
    return decorator


def set_seed(random_seed_bool, seed=1):
    if random_seed_bool:
        seed = random.randint(0, 1000)
    random.seed(seed)
    np.random.seed(seed)
    print(f'[SEED]: --- {seed} ---')


def check_if_nei_pos(agents):
    for agent in agents:
        if agent.curr_node.xy_name not in agent.prev_node.neighbours:
            raise RuntimeError('wow wow wow! Not nei pos!')


def plan_has_no_conf_with_vertex(plan, vertex):
    for plan_v in plan:
        if plan_v.xy_name == vertex.xy_name:
            return False
    return True


def two_plans_have_no_confs(plan1, plan2):

    min_len = min(len(plan1), len(plan2))
    assert len(plan1) == len(plan2)
    prev1 = None
    prev2 = None
    for i, (vertex1, vertex2) in enumerate(zip(plan1[:min_len], plan2[:min_len])):
        if vertex1.xy_name == vertex2.xy_name:
            return False
        if i > 0:
            # edge1 = (prev1.xy_name, vertex1.xy_name)
            # edge2 = (vertex2.xy_name, prev2.xy_name)
            if (prev1.xy_name, vertex1.xy_name) == (vertex2.xy_name, prev2.xy_name):
                return False
        prev1 = vertex1
        prev2 = vertex2
    return True


def two_plans_have_confs_at(plan1, plan2):

    min_len = min(len(plan1), len(plan2))
    assert len(plan1) == len(plan2)
    prev1 = None
    prev2 = None
    for i, (vertex1, vertex2) in enumerate(zip(plan1[:min_len], plan2[:min_len])):
        if vertex1.xy_name == vertex2.xy_name:
            return True, i
        if i > 0:
            # edge1 = (prev1.xy_name, vertex1.xy_name)
            # edge2 = (vertex2.xy_name, prev2.xy_name)
            if (prev1.xy_name, vertex1.xy_name) == (vertex2.xy_name, prev2.xy_name):
                return True, i
        prev1 = vertex1
        prev2 = vertex2
    return False, -1


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


def create_sds_sub_results(h_agents, nei_plans_dict):
    # sub results
    sub_results = {}
    for agent in h_agents:
        # h_plan = agent.plan
        h_plan = [agent.curr_node]
        h_plan.extend(nei_plans_dict[agent.name])
        sub_results[agent.name] = h_plan
    # sub_results = {agent.name: agent.plan for agent in h_agents}
    return sub_results


def build_constraints(nodes, other_paths):
    v_constr_dict = {node.xy_name: [] for node in nodes}
    e_constr_dict = {node.xy_name: [] for node in nodes}
    perm_constr_dict = {node.xy_name: [] for node in nodes}
    xyt_problem = False

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
                    xyt_problem = True
                prev_node = node
    return v_constr_dict, e_constr_dict, perm_constr_dict, xyt_problem


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


def get_nei_nodes_times(curr_node, nei_r, nodes_dict):
    nei_nodes_dict = {}
    curr_node.t = 0
    open_list = [curr_node]
    while len(open_list) > 0:
        i_node = open_list.pop()
        i_node_distance = euclidean_distance_nodes(curr_node, i_node)
        if i_node_distance <= nei_r:
            nei_nodes_dict[i_node.xy_name] = i_node
            for node_nei_name in i_node.neighbours:
                if node_nei_name not in nei_nodes_dict:
                    node_nei = nodes_dict[node_nei_name]
                    node_nei.t = i_node.t + 1
                    open_list.append(node_nei)
    nei_nodes = list(nei_nodes_dict.values())
    return nei_nodes, nei_nodes_dict


@lru_cache(maxsize=128)
def euclidean_distance_nodes(node1, node2):
    # p = [node1.x, node1.y]
    # q = [node2.x, node2.y]
    return math.dist([node1.x, node1.y], [node2.x, node2.y])
    # return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)


@lru_cache(maxsize=128)
def manhattan_distance_nodes(node1, node2):
    return abs(node1.x-node2.x) + abs(node1.y-node2.y)

