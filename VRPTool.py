import os
from copy import deepcopy

import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from sklearn.metrics import pairwise_distances
from multiprocessing.pool import Pool


PATH = '../VRPInstances/GEN/'
def check_and_calc(seq_lst, dist, demands, capacity, no_demand=True):
    total_cost = 0
    total_demand = 0
    for seq in seq_lst:
        if(len(seq) < 1): continue
        cost, demand = 0, 0
        all_seq = [0] + seq + [0]
        for i in range(len(all_seq) - 1):
            v1 = all_seq[i]
            v2 = all_seq[i+1]
            cost += dist[v1, v2]
            demand += demands[v2]
        if demand > capacity:
            return -1
        assert demand <= capacity, 'demands in this route exceed'
        total_cost += cost
        total_demand += demand
    if no_demand:
        return total_cost
    else:
        return total_cost, total_demand
def CVRP(num_node, low, high, cnt):
    data = {
        'NAME': f'GEN-{cnt}-{num_node}',
        'COMMENT': 'None',
        'TYPE': 'CVRP',
        'DIMENSION': f'{num_node}',
        'EDGE_WEIGHT_TYPE': 'EUC_2D',
        'CAPACITY': '100',
        'NODE_COORD_SECTION': None,
        'DEMAND_SECTION': None,
        'DEPOT_SECTION': None
    }
    vertexes = np.random.randint(low=low, high=high, size=(num_node, 3))
    vertexes[0, :] = np.round(vertexes[0, :] * 2 / 3, 0)
    vertexes[0, 2] = 0
    data['NODE_COORD_SECTION'] = vertexes[:, :2].astype(int).tolist()
    data['DEMAND_SECTION'] = vertexes[:, 2].astype(int).tolist()
    data['DEPOT_SECTION'] = 1
    return data

# low; high -> [x, y, demand]
def instance(min_node, max_node, bsz, num_instance):
    num_counter = 0
    nums = np.linspace(min_node, max_node, bsz)
    graphs = []
    while num_counter < num_instance:
        for kk in nums:
            num_node = np.random.randint(low=int(kk-5), high=int(kk+5))
            if num_counter >= num_instance: break
            num_counter += 1
            data = CVRP(num_node=num_node, low=[-100,-100,1], high=[100,100,25], cnt=num_counter)
            graphs.append(data)
    return graphs
            # print(data)

def generator():
    instances = instance(min_node=72, max_node=200, bsz=32, num_instance=1280)
    for data in instances:
        filename = PATH + data['NAME'] + '.vrp'
        with open(filename, 'w+', encoding='utf-8') as f:
            KEYS = ['NAME', 'COMMENT', 'TYPE', 'DIMENSION', 'EDGE_WEIGHT_TYPE', 'CAPACITY', 'NODE_COORD_SECTION',
                    'DEMAND_SECTION', 'DEPOT_SECTION']
            output = f''
            for key in KEYS:
                if isinstance(data[key], str):
                    output += f'{key} : {data[key]}\n'
                elif isinstance(data[key], list):
                    output += f'{key}\n'
                    for vid, vals in enumerate(data[key]):
                        output += f'{vid + 1}'
                        if isinstance(vals, list):
                            for val in vals:
                                output += f' {val}'
                        else:
                            output += f' {vals}'
                        output += '\n'
                elif isinstance(data[key], int):
                    output += f'{key}\n{data[key]}\n-1\nEOF\n'
            f.write(output)
def create_data_model(graph):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = pairwise_distances(X=graph['VERTEX'], metric='euclidean').astype(int).tolist()
    data['demands'] = np.reshape(graph['DEMANDS'], -1).astype(int).tolist()
    data['num_vehicles'] = len(data['demands']) // 2
    data['vehicle_capacities'] = [100] * data['num_vehicles']
    data['depot'] = 0
    return data

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    res = {
        'tot_cost': None,
        'seq_list': [],
    }
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = f'Route for vehicle {vehicle_id}:\n'
        route_distance = 0
        route_load = 0
        temp = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            if data['demands'][node_index] > 0:
                temp.append(node_index)
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            # previous_index = index
            index = solution.Value(routing.NextVar(index))
            # route_distance += routing.GetArcCostForVehicle(
            #     previous_index, index, vehicle_id
            # )
            route_distance += data['distance_matrix'][node_index][manager.IndexToNode(index)]
        if len(temp) > 0:
            res['seq_list'].append(temp)
        # plan_output += f" {manager.IndexToNode(index)} Load({route_load})\n"
        # plan_output += f"Distance of the route: {route_distance}m\n"
        # plan_output += f"Load of the route: {route_load}\n"
        # print(plan_output)
        total_distance += route_distance
        total_load += route_load
    # print('Total distance of all routes: {}m'.format(total_distance))
    # print('Total load of all routes: {}'.format(total_load))
    res['tot_cost'] = total_distance
    return res
# subsequent solver for better routing quality
def main_solver(graph=None, data=None, solution_seqs=None, time_lmt=10000):
    """Solve the CVRP problem."""
    assert graph is not None or data is not None
    # Instantiate the data problem.
    if data is None:
        data = create_data_model(graph=graph)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']),
        data['num_vehicles'],
        data['depot']
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data["vehicle_capacities"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromMilliseconds(time_lmt)


    # When an initial solution is given for search, the model will be closed with
    # the default search parameters unless it is explicitly closed with the custom
    # search parameters.

    # init_solution = None
    # if solution_seqs is not None:
    #     routing.CloseModelWithParameters(search_parameters)

    #     init_solution = routing.ReadAssignmentFromRoutes(solution_seqs, True)
    #     print("Status after:", routing.status())
    #     assert init_solution is not None, 'init solution errors'

    init_solution = None
    if solution_seqs is not None:
        routing.CloseModelWithParameters(search_parameters)
        init_solution = routing.ReadAssignmentFromRoutes(solution_seqs, True)
        assert init_solution is not None, 'init routing solution errors'

    # Solve the problem.
    if init_solution is None:
        solution = routing.SolveWithParameters(search_parameters)
    else:
        solution = routing.SolveFromAssignmentWithParameters(init_solution, search_parameters)
    # Print solution on console.
    # if solution:
    res = print_solution(data, manager, routing, solution)
    return res
