import base64
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances
import logging
import re
from VRPTool import check_and_calc
from ga_src.env import RUNS
# import google.ai.generativelanguage as glm

from ga_src.local_search import local_search


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    # Remove all handlers if there are any
    while logger.hasHandlers() and len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[0])

    # File handler for outputting to a file
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def obt_labels(seq_lst, n_customer):
    labels = [0] * (n_customer + 1)
    for idx, route in enumerate(seq_lst):
        for nid in route:
            labels[nid] = idx + 1
    return labels

# def obtain_y(seq_lst):
#     y = [0]
#     map = {}
#     for idx, seq in enumerate(seq_lst):
#         for val in seq:
#             map[val] = idx + 1
#     for val in map.values():
#         y.append(val)
#     return y

def draw(transformed, title, filename=None, seq_lst=None, y=None, bias=1, font=50, color=False, addr='jpeg', demands=None):
    balance = 1
    if y is None and color == True:
        y = obt_labels(seq_lst, n_customer=len(transformed)-1)
    plt.clf()
    plt.figure(figsize=(6.4/balance, 4.8/balance))
    data = np.array(transformed)
    if color == True:
        plt.scatter(data[0, 0], data[0, 1], c=y[0], marker='*', s=np.pi * font)
    else:
        plt.scatter(data[0, 0], data[0, 1], c='red', marker='*', s=np.pi * font)
    x_axis = data[1:, 0]
    y_axis = data[1:, 1]
    IDS = list(range(len(data)))
    if color == True:
        plt.scatter(x_axis, y_axis, c=y[1:], s=np.pi * font/10)
    else:
        plt.scatter(x_axis, y_axis, c='black', s=np.pi * font/10)
    for i in range(len(data)):
        if demands is None:
            plt.annotate(IDS[i], xy=(data[i, 0], data[i, 1]), xytext=(data[i, 0]+bias, data[i, 1]+bias), fontsize=5)
        else:
            plt.annotate(f"{IDS[i]}({demands[i]})", xy=(data[i, 0], data[i, 1]), xytext=(data[i, 0] + bias, data[i, 1] + bias), fontsize=5)
    if seq_lst is not None:
        for seq in seq_lst:
            for l in range(len(seq)-1):
                r = l + 1
                l, r = seq[l], seq[r]
                x1, y1, x2, y2 = data[l, 0], data[l, 1], data[r, 0], data[r, 1]
                plt.arrow(x1, y1, x2-x1, y2-y1, width=bias/100)
    buffer = io.BytesIO()
    plt.title(title)
    if filename is not None:
        plt.savefig(fname=filename+f'.{addr}', dpi=1200, bbox_inches='tight')
    plt.savefig(buffer, format='jpeg')
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return b64
    # img = Image.open(io.BytesIO(base64.b64decode(b64)))
    # img.show()
    # plt.show()
def draw_both(transformed, seq_lst, filename=None, y=None, bias=1, font=50, color=False, demands=None):
    balance = 1
    if y is None and color == True:
        y = obt_labels(seq_lst, n_customer=len(transformed)-1)
    data = np.array(transformed)

    plt.clf()
    plt.figure(figsize=(12.8/balance,4.8/balance))
    plt.subplot(1, 2, 1)
    plt.title('Sub-figure A: Original Vertex Layout')
    if color == True:
        plt.scatter(data[0, 0], data[0, 1], c=y[0], marker='*', s=np.pi * font)
    else:
        plt.scatter(data[0, 0], data[0, 1], c='red', marker='*', s=np.pi * font)
    x_axis = data[1:, 0]
    y_axis = data[1:, 1]
    IDS = list(range(len(data)))
    if color == True:
        plt.scatter(x_axis, y_axis, c=y[1:], s=np.pi * font / 10)
    else:
        plt.scatter(x_axis, y_axis, c='black', s=np.pi * font / 10)
    for i in range(len(data)):
        if demands is None:
            plt.annotate(IDS[i], xy=(data[i, 0], data[i, 1]), xytext=(data[i, 0] + bias, data[i, 1] + bias), fontsize=5)
        else:
            plt.annotate(f"{IDS[i]}({demands[i]})", xy=(data[i, 0], data[i, 1]), xytext=(data[i, 0] + bias, data[i, 1] + bias), fontsize=5)

    plt.subplot(1,2,2)
    plt.title('Sub-figure B: Optimal Traverse Routes')
    if color == True:
        plt.scatter(data[0, 0], data[0, 1], c=y[0], marker='*', s=np.pi * font)
    else:
        plt.scatter(data[0, 0], data[0, 1], c='red', marker='*', s=np.pi * font)
    x_axis = data[1:, 0]
    y_axis = data[1:, 1]
    IDS = list(range(len(data)))
    if color == True:
        plt.scatter(x_axis, y_axis, c=y[1:], s=np.pi * font/10)
    else:
        plt.scatter(x_axis, y_axis, c='black', s=np.pi * font/10)
    for i in range(len(data)):
        if demands is None:
            plt.annotate(IDS[i], xy=(data[i, 0], data[i, 1]), xytext=(data[i, 0] + bias, data[i, 1] + bias), fontsize=5)
        else:
            plt.annotate(f"{IDS[i]}({demands[i]})", xy=(data[i, 0], data[i, 1]), xytext=(data[i, 0] + bias, data[i, 1] + bias), fontsize=5)
    if seq_lst is not None:
        for seq in seq_lst:
            for l in range(len(seq)-1):
                r = l + 1
                l, r = seq[l], seq[r]
                x1, y1, x2, y2 = data[l, 0], data[l, 1], data[r, 0], data[r, 1]
                plt.arrow(x1, y1, x2-x1, y2-y1, width=bias/100)
    # plt.show()
    if filename is not None:
        plt.savefig(fname=filename+'.jpeg', dpi=1200, bbox_inches='tight')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpeg')
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return b64

def information_to_sol_seq(information):
    routes_inf = re.findall(r"<route[^>]*>([^<]+)</route>", information)
    separators = ",[]{} "
    pattern = "[" + re.escape(separators) + "]+"
    seq_lst = []
    for route_inf in routes_inf:
        seq_inf = re.split(pattern, route_inf)
        seq = []
        for val_inf in seq_inf:
            if val_inf.isdigit():
                seq.append(int(val_inf))
        seq_lst.append(seq)
    return seq_lst
def information_to_oper_seq(information):
    opts_inf = re.findall(r"<execution.*?>(.*?)</execution>", information)
    separators = ",[]{} "
    pattern = "[" + re.escape(separators) + "]+"
    opt_lst = []
    for opt_inf in opts_inf:
        seq_inf = re.split(pattern, opt_inf)
        seq = []
        for val_inf in seq_inf:
            if val_inf.isdigit():
                seq.append(int(val_inf))
        opt_lst.append(seq)
    return opt_lst
# check if all the nodes in the information, length = 1 + n_customer

def sol_in_information(information, length):
    length -= 1 # number of customer vertexes
    seq_lst = information_to_sol_seq(information)
    all_values = []
    for seq in seq_lst:
        for val in seq:
            all_values.append(val)
    all_values = sorted(all_values)
    if length != len(all_values):
        return False
    for i in range(length):
        if all_values[i] != i + 1:
            return False
    return True


def modity_anyway(information, length):
    seq_lst = information_to_sol_seq(information)
    all_values = []
    for seq in seq_lst:
        for val in seq:
            if val > 0 and val < length and all_values.count(val) < 1:
                all_values.append(val)
    for val in range(1, length):
        if all_values.count(val) < 1:
            all_values.append(val)
    return [all_values]
def duplicated_and_missed(information, length):
    # length -= 1  # number of customer vertexes
    seq_lst = information_to_sol_seq(information)
    all_values = []
    for seq in seq_lst:
        for val in seq:
            all_values.append(val)
    missed_IDs = []
    duplicates_IDs = []
    errors = [x for x in all_values if x < 1 or x >= length]
    for idx in range(1, length):
        if all_values.count(idx) > 1:
            duplicates_IDs.append(idx)
        elif all_values.count(idx) < 1:
            missed_IDs.append(idx)
    return duplicates_IDs, missed_IDs, errors
    # all_values = sorted(all_values)
    # duplicates = [x for x in all_values if all_values.count(x) > 1]
    # errors = [x for x in all_values if x < 1 or x ]


    # for idx in range(1, length):
    #     if not f'{idx}' in information:
    #         return False



def to_seq_lst(single_route):
    result = []
    temp = []
    for cid in single_route[1:]:
        if cid == 0:
            result.append(temp)
            temp = []
        else:
            temp.append(cid)
    return result
def beam_search(DM, vehicle_capacity, demands, beam_width):
    num_nodes = len(DM)
    # Initialize the beam with the depot node
    beam = [(0, [0], 0, 0)]  # (cost, route, load, last_node)

    for _ in range(num_nodes - 1):
        new_beam = []
        for cost, route, load, last_node in beam:
            # Sort the nodes based on the distance matrix
            next_nodes = sorted(range(1, num_nodes), key=lambda x: DM[last_node][x]) # check the ranking!!!
            # next_nodes = list(range(1, num_nodes))
            for next_node in next_nodes:
                if next_node not in route:
                    if load + demands[next_node] > vehicle_capacity:
                        # If the load exceeds the vehicle capacity, return to depot and start a new route
                        new_cost = cost + DM[last_node][0] + DM[0][next_node]
                        new_route = route + [0, next_node]
                        new_load = demands[next_node]
                    else:
                        new_cost = cost + DM[last_node][next_node]
                        new_route = route + [next_node]
                        new_load = load + demands[next_node]
                    new_last_node = next_node
                    new_beam.append((new_cost, new_route, new_load, new_last_node))
        # Keep the top 'beam_width' routes
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]

        # Add the depot node to the end of each route
    final_beam = []
    for cost, route, load, last_node in beam:
        final_cost = cost + DM[last_node][0]
        final_route = route + [0]
        final_beam.append((final_cost, final_route))

    # Return the route with the minimum cost
    final_beam.sort(key=lambda x: x[0])
    return to_seq_lst(final_beam[0][1])

def repair(llm_seq_lst, dist, demands, capacity):
    llm_cost = check_and_calc(llm_seq_lst, dist, demands, capacity)
    if llm_cost != -1:
        return llm_cost, llm_seq_lst
    new_llm_seq_lst = []

    for seq in llm_seq_lst:
        temp = []
        temp_demand = 0
        for cid in seq:
            if temp_demand + demands[cid] <= capacity:
                temp_demand += demands[cid]
                temp.append(cid)
            else:
                temp_demand = demands[cid]
                new_llm_seq_lst.append(temp)
                temp = [cid]
        if len(temp) > 0:
            new_llm_seq_lst.append(temp)

    return check_and_calc(new_llm_seq_lst, dist, demands, capacity), new_llm_seq_lst

def get_random_index(x):
    n = len(x)
    return np.random.choice(x, size=n, replace=False)

def seqlst_to_routes(seq_lst):
    result = [0]
    for seq in seq_lst:
        if len(seq) > 0:
            result = result + seq + [0]
    return result
def to_search(dist, demands, capacity, tm_lmt, sol_seqs=None):
    data = {}
    data['depot'] = 0
    # dist[data['depot'], :] = dist[:, data['depot']] = np.max(dist)
    data['distance_matrix'] = (dist * 100).astype(int).tolist()
    data['demands'] = np.reshape(demands, -1).astype(int).tolist()
    data['num_vehicles'] = len(data['demands'])
    # data['num_vehicles'] = len(sol_seqs)
    data['vehicle_capacities'] = [int(capacity)] * data['num_vehicles']
    from VRPTool import main_solver
    OPT = main_solver(data=data, time_lmt=tm_lmt, solution_seqs=sol_seqs)['seq_list']
    cost = check_and_calc(OPT, dist, demands, capacity)
    return cost, OPT
    # labels = [0] * len(data['demands'])
    # for idx, route in enumerate(OPT['seq_list']):
    #     for nid in route:
    #         labels[nid] = idx + 1
    # return labels, OPT['seq_list']

def obt_result(data, novision, vision, tm_lmt):
    dist = pairwise_distances(X=data['VERTEX'], metric='euclidean')
    opt_seq_lst = data['OPTIMAL']
    demands = np.array(data['DEMANDS']).reshape(-1)
    capacity = int(data['CAPACITY'])

    cost_novision, novision = repair(novision, dist, demands, capacity)
    cost_vision, vision = repair(vision, dist, demands, capacity)

    # costor_novision, or_novision = to_search(dist, demands, capacity, tm_lmt=tm_lmt, sol_seqs=novision)
    # costor_vision, or_vision = to_search(dist, demands, capacity, tm_lmt=tm_lmt, sol_seqs=vision)

    or_novision, costor_novision, _ = local_search(seqlst_to_routes(novision), dist, capacity, demands, np.inf)
    or_vision, costor_vision, _ = local_search(seqlst_to_routes(vision), dist, capacity, demands, np.inf)

    # cost_init, init = to_search(dist, demands, capacity, tm_lmt=tm_lmt)
    cost_init = check_and_calc(opt_seq_lst, dist, demands, capacity)
    return cost_init, cost_novision, cost_vision, costor_novision, costor_vision
    # return cost_init, cost_novision, cost_vision

def get_random_solution(data):
    dist = pairwise_distances(X=data['VERTEX'], metric='euclidean')
    opt_seq_lst = data['OPTIMAL']
    demands = np.array(data['DEMANDS']).reshape(-1)
    capacity = int(data['CAPACITY'])

    # random generate
    length = len(demands)
    ind = list(range(1, length))
    best_random_lst = None
    best_random_opt = np.inf
    for i in range(1):
        ran_lst = get_random_index(ind)
        ran_cost, ran_seq_lst = repair([ran_lst], dist, demands, capacity)
        if ran_cost < best_random_opt:
            best_random_opt = ran_cost
            best_random_lst = ran_seq_lst
    return best_random_opt, best_random_lst
def process(data, LLM_SEQ_LST_NOVISION=None, LLM_SEQ_LST_VISION=None, color=False):
    dist = pairwise_distances(X=data['VERTEX'], metric='euclidean')
    opt_seq_lst = data['OPTIMAL']
    demands = np.array(data['DEMANDS']).reshape(-1)
    capacity = int(data['CAPACITY'])
    beam_seq_lst = beam_search(dist, capacity, demands, beam_width=100)
    # data['img_opt'] = draw(transformed=data['VERTEX'], title=f'{data["NAME"]} img_opt cost: {check_and_calc(opt_seq_lst, dist, demands, capacity):.2f}', seq_lst=opt_seq_lst, color=color)
    # data['img_bs'] = draw(transformed=data['VERTEX'], title=f'{data["NAME"]} img_bs cost: {check_and_calc(beam_seq_lst, dist, demands, capacity):.2f}', seq_lst=beam_seq_lst, color=color)
    if LLM_SEQ_LST_NOVISION is not None:
        llm_cost1, llm_seq_lst1 = repair(LLM_SEQ_LST_NOVISION, dist, demands, capacity)
        llm_cost2, llm_seq_lst2 = repair(LLM_SEQ_LST_VISION, dist, demands, capacity)
        # data['img_llm_novision'] = draw(transformed=data['VERTEX'], title=f'{data["NAME"]} img_llm-no-vision cost: {check_and_calc(llm_seq_lst1, dist, demands, capacity):.2f}', seq_lst=llm_seq_lst1, color=color)
        # data['img_llm_vision'] = draw(transformed=data['VERTEX'], title=f'{data["NAME"]} img_llm-with-vision cost: {check_and_calc(llm_seq_lst2, dist, demands, capacity):.2f}', seq_lst=llm_seq_lst2, color=color)



    cost_opt, cost_bs = check_and_calc(opt_seq_lst, dist, demands, capacity), check_and_calc(beam_seq_lst, dist, demands, capacity)
    # llm_cost1 llm_cost2 best_random_opt

    # data['img_rand'] = draw(transformed=data['VERTEX'], title=f'{data["NAME"]} img_rand cost: {check_and_calc(best_random_lst, dist, demands, capacity):.2f}', seq_lst=best_random_lst, color=color)
    print(f'OPT cost: {cost_opt:.2f} Beam Search cost: {cost_bs:.2f} gap ({(cost_bs-cost_opt)/cost_opt:.3f}) LLM-no-vision cost: {llm_cost1:.2f} gap ({(llm_cost1-cost_opt)/cost_opt:.3f}) LLM-with-vision cost: {llm_cost2:.2f} gap ({(llm_cost2-cost_opt)/cost_opt:.3f}) random cost: {best_random_opt:.2f} gap ({(best_random_opt-cost_opt)/cost_opt:.3f}) \n')