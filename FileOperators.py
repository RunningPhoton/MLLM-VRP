import os
from copy import deepcopy
import numpy as np

from ga_src.env import POP_SIZE
from utils import check_and_calc
from sklearn.metrics import pairwise_distances


def read_vrp_instance(filename, **kwargs):
    data = {
        'NAME': None,
        'DIMENSION': None,
        'CAPACITY': None,
        'COMMENT': None,
        'OPTIMAL': [],
        'VERTEX': [],
        'DEMANDS': []
    }
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        NODE_COORD_SECTION, DEMAND_SECTION = False, False
        for i, line in enumerate(lines):
            if 'NODE_COORD_SECTION' in line:
                NODE_COORD_SECTION = True
            if 'DEMAND_SECTION' in line:
                DEMAND_SECTION = True
            if 'DEPOT_SECTION' in line:
                assert int(lines[i+1]) == 1, 'Depot must be the first vertex'
            for key in data.keys():
                if key in line:
                    data[key] = line.split()[-1]
            seq = line.split()
            if NODE_COORD_SECTION == True and len(seq) == 3 and seq[0].isdigit():
                # dataset['VERTEX'].append(Vertex(x=float(seq[1]), y=float(seq[2])))
                data['VERTEX'].append([float(val) for val in seq[1:]])
            if DEMAND_SECTION == True and len(seq) == 2 and seq[0].isdigit():
                data['DEMANDS'].append([float(seq[-1])])
    return data

def read_vrp_solution(filename, **kwargs):
    data = kwargs.get('data')
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if 'Route' in line:
                temp = [int(val) for val in line.split()[2:]]
                data['OPTIMAL'].append(temp)
    return data

def rotate_coordinates(mat, anchor, theta):
    # Step 1: Subtract the anchor coordinates from mat
    mat = mat - anchor

    # Step 3: Create rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    # Step 4: Rotate the points
    rotated_mat = np.dot(mat, R)

    # Step 5: Add the anchor coordinates back
    rotated_mat = rotated_mat + anchor

    return rotated_mat

def data_enhancement(data, anchor, theta):
    new_data = deepcopy(data)
    # len_vertex = len(new_data['VERTEX'])
    # anchor = np.array(data['VERTEX'][np.random.choice(len_vertex, size=1)])
    new_data['VERTEX'] = rotate_coordinates(np.array(new_data['VERTEX']), anchor, theta).tolist()
    return new_data


response_sample = '''
<SOLUTION name={}>
    <route id={} travel_cost={} travel_demand={}>[{}]</route>
    ......
    <route id={} travel_cost={} travel_demand={}>[{}]</route>
</SOLUTION>
'''



def data_to_description(data, vision, solution=False, failed=False, illustration=None, metaLS=False):
    n_customer = int(data['DIMENSION'])-1
    stream = ''
    stream += f"\n<CVRP name={data['NAME']} n_customer={n_customer} capacity={data['CAPACITY']}>"
    stream += f"\n    <Depot index_id=0 X-axis={data['VERTEX'][0][0]} Y-axis={data['VERTEX'][0][1]} demand=0></Depot>"
    stream += f"\n    <Customers>"
    for i in range(1, len(data['VERTEX'])):
        stream += f"\n        <customer index_id={i} X-axis={data['VERTEX'][i][0]} Y-axis={data['VERTEX'][i][1]} demand={data['DEMANDS'][i][0]}></customer>"
    stream += f"\n    </Customers>"
    if solution:
        dist = pairwise_distances(X=data['VERTEX'], metric='euclidean')
        demands = np.array(data['DEMANDS']).reshape(-1)
        stream += f"\n    <SOLUTION>"
        for i, seq in enumerate(data['OPTIMAL']):
            travel_cost, travel_demand = check_and_calc([seq], dist, demands, int(data['CAPACITY']), no_demand=False)
            stream += f"\n        <route id={i+1} travel_cost={travel_cost:.1f} travel_demand={travel_demand}>{[val for val in seq]}</route>"
        stream += f"\n    </SOLUTION>"
    stream += f"\n</CVRP>\n"
    if failed:
        return stream
    if solution == False:
        if illustration is None:
            if metaLS == True:
                stream += f"Kindly return me the operation sequence of Local Search methods to be applied for improving the solution quality (comprising at least one ID of Local Search methods ranging from 0 to 9) in XML format, adhering to the heuristics that you have previously acquired.\n"
            else:
                stream += f"Kindly return me the complete preliminary solution of {data['NAME']} (comprising all the customer vertex IDs **ranging from 1 to {n_customer} without repetitions** present in the above XML text) in XML format, adhering to the heuristics that you have previously acquired.\n"
            stream += f"**No Explanation Needed**\n"
        else:
            stream += f"Please first return me the complete preliminary solution of {data['NAME']} (comprising all the customer vertex IDs **ranging from 1 to {n_customer} without repetitions** present in the above XML text) in XML format, adhering to the heuristics that you have previously acquired.\n"
            stream += f"Next below the normal XML text of the solution, please also return me your explanations of constructing each route accordingly.\n{illustration}"
    else:
        if metaLS == True:
            if vision == True:
                stream += f'''For the figure uploaded, the left sub-figure displays the original vertex distribution layout, while the right sub-figure exhibits the layout with optimal traverse routes. In both sub-figures, the depot vertex is marked by a red ‘star’, the customer nodes are signified by black ‘circles’. The index ID of each vertex is displayed at the upper right-hand side of the index. The arrows depicted in the right sub-figure indicates the optimal traverse routes (the edges between the depot and the customer have been removed for clarity).
                \nI trust that these figures of solved CVRPs will enable you to grasp the heuristics for developing novel sequence of Local Search methods to construct high-quality routing solutions.\n'''
                stream += f"You may start by finding the accurate customer mapping between the XML document and the sub-figures according to the IDs, and then return the observations you found to develop novel combination of Local Search methods according to the XML description, the original vertex distribution layout and the optimal traverse routes of the CVRP namely {data['NAME']}.\n"
            else:
                stream += f"Please return the observations you found to develop novel combination of Local Search methods for constructing high-quality routing solution according to the XML description of CVRP namely {data['NAME']}.\n"
        else:
            if vision == True:
                stream += f'''For the figure uploaded, the left sub-figure displays the original vertex distribution layout, while the right sub-figure exhibits the layout with optimal traverse routes. In both sub-figures, the depot vertex is marked by a red ‘star’, the customer nodes are signified by black ‘circles’. The index ID of each vertex is displayed at the upper right-hand side of the index. The arrows depicted in the right sub-figure indicates the optimal traverse routes (the edges between the depot and the customer have been removed for clarity).
                \nI trust that these figures of solved CVRPs will enable you to grasp the heuristics to construct high-quality initial routing solutions.\n'''
                stream += f"You may start by finding the accurate customer mapping between the XML document and the sub-figures according to the IDs, and then return the observations you found to construct high-quality routing solutions according to the XML description, the original vertex distribution layout and the optimal traverse routes of the CVRP namely {data['NAME']}.\n"
            else:
                stream += f"Please return the observations you found to construct high-quality routing solution according to the XML description of CVRP namely {data['NAME']}.\n"

    return stream

def data_loading(input_dirs, **kwargs):
    graphs = []
    vrps, sols = [], []
    for path in input_dirs:
        total = os.listdir(path)
        for file in total:
            if '.vrp' in file:
                vrps.append(path + file)
            if '.sol' in file:
                sols.append(path + file)
    for i in range(len(vrps)):
        vrp_file = vrps[i]
        sol_file = sols[i]
        data = read_vrp_instance(vrp_file)
        data = read_vrp_solution(sol_file, data=data)
        graphs.append(data)
    return graphs


def single_data_loading(input_file, **kwargs):
    vrp_file = input_file + '.vrp'
    sol_file = input_file + '.sol'
    data = read_vrp_instance(vrp_file)
    data = read_vrp_solution(sol_file, data=data)
    return data
def read_seq_lst(filename):
    result = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        temp = []
        for line in lines:
            for val in line.split():
                temp.append(int(val))
        result.append(temp)
    return result
def save_seq_lst(seq_lst, output):
    with open(output, 'w+', encoding='utf-8') as f:
        info = f""
        for seq in seq_lst:
            if len(seq) == 0: continue
            for val in seq:
                info += f"{val} "
            info += '\n'
        f.write(info)
