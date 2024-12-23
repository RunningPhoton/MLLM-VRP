import os

import openai

from VRPTool import check_and_calc
from ga_src.env import apikey, temperature, max_tokens, RUNS, HRUNS, PLS, POP_SIZE, MAX_GEN
from openai_tools import openai_VRPTEST1, openai_VRPTEST2, sol_test, draw_test, obt_table, illustration, \
    illustrate_draw, write_pkl, open_pkl
from FileOperators import data_loading, read_seq_lst, single_data_loading
from utils import repair, seqlst_to_routes
from sklearn.metrics import pairwise_distances
import numpy as np
from multiprocessing.pool import Pool
from ga_src.Solver import solve
# Set the model parameters
model_name = "gpt-4-vision-preview"
# RUNS = [1]
# stop = ["\n"]
# response_sample = '''
# <SOLUTION name={}>
#     <route id={} total_cost={} total_demand={}>{}</route>
#     ......
#     <route id={} total_cost={} total_demand={}>{}</route>
# </SOLUTION>
# '''
def walk():
    cvrps = data_loading(input_dirs=['vrp_instance/small/', 'vrp_instance/large/', 'vrp_instance/vlarge/'])
    for cvrp in cvrps:
        print(f' & {cvrp["NAME"]} & {len(cvrp["DEMANDS"])} & {len(cvrp["OPTIMAL"])} & {cvrp["CAPACITY"]} \\\\')

def ill_draw():
    # captured from knowledge/source-illustration-novision and knowledge/source-illustration-novision
    novision = [[1, 10, 4, 11, 14, 12, 3, 8, 16, 17], [2, 7, 5, 18, 6, 13, 15], [9]]
    vision = [[1, 10, 4, 11, 14, 12, 3, 8, 16, 17], [2, 7, 9, 15, 13, 5, 18], [6]]
    illustrate_draw(['vrp_instance/illustration/'], novision, vision)


def MLLM_init():
    base_url1 = 'https://openai.talentianai.com/v1/'
    # base_url2 = 'https://api.openai.com/v1/'
    # walk()
    # ill_draw()
    # meta_heuristic()
    for scale in ['vlarge']:
        file_dir = f'vrp_instance/{scale}/'
        source_dir = 'vrp_instance/solved/'
        client = openai.OpenAI(api_key=apikey, base_url=base_url1)
        # illustration(client=client, source_dir=[source_dir], file_name=f'{file_dir.split("/")[1]}-vision', input_dirs=[file_dir], model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        # illustration(client=client, source_dir=[source_dir], file_name=f'{file_dir.split("/")[1]}-novision', input_dirs=[file_dir], model_name=model_name, temperature=temperature, max_tokens=max_tokens)

        # draw_test([source_dir])

        # draw_test(file_dir_lst=[file_dir])
        # RUNS
        for run_id in [6, 7, 8, 9, 10]:
            openai_VRPTEST1(run_id=run_id, client=client, source_dir=[source_dir], file_name=f'{file_dir.split("/")[1]}-novision', input_dirs=[file_dir], model_name=model_name, temperature=temperature, max_tokens=max_tokens)
            openai_VRPTEST2(run_id=run_id, client=client, source_dir=[source_dir], file_name=f'{file_dir.split("/")[1]}-vision', input_dirs=[file_dir], model_name=model_name, temperature=temperature, max_tokens=max_tokens)

        # obt_table(benchmark=f'{scale}/', tm_lmt=150)

def parallel_eval(file, res_files):
    run_id = int(file.split('-')[-1])
    data_file = file.replace(f'-{run_id}', "")
    data = single_data_loading(data_file)
    dist = pairwise_distances(X=data['VERTEX'], metric='euclidean')
    demands = np.array(data['DEMANDS']).reshape(-1).tolist()
    capacity = int(data['CAPACITY'])
    cvrpname = data_file.split('/')[-1]
    init_sols = []
    mode = 'random'
    opt_seq = data['OPTIMAL']
    opt_cost = check_and_calc(opt_seq, dist, demands, capacity)
    for res_file in res_files:
        mode = res_file.split('/')[-1].replace(cvrpname+'-', "").split('-')[0]
        # run_id = int(res_file.split('/')[-1].replace(cvrpname+'-', "").split('-')[1])
        init_cost, init_sol = repair(read_seq_lst(res_file), dist, demands, capacity)
        sol = seqlst_to_routes(init_sol)
        init_sols.append(sol)

    result = {
        'name': f'{cvrpname}-{mode}-{run_id}',
        'optimal': f'{opt_cost}',
        'res': solve(data, run_id, PLS=PLS, init_sols=init_sols, pop_size=POP_SIZE, max_gen=MAX_GEN)
    }

    return result

def assessment(scale, mode, result_to_save, THREAD):
    output_dir = f'outputs/{scale}/'
    input_dir = f'vrp_instance/{scale}/'
    file_lst = sorted(list(set([filename.split('.')[0] for filename in os.listdir(input_dir)])))
    data_file_lst = []
    res_file_lst = []
    for cvrpname in file_lst:
        for hrun in HRUNS:
            data_file_lst.append(f'{input_dir}{cvrpname}-{hrun}')
            temp = []
            if mode != 'random':
                for run_id in RUNS:
                    temp.append(f'{output_dir}{cvrpname}-{mode}-{run_id}')
            res_file_lst.append(temp)
    # data_file_lst = data_file_lst[:2]
    # res_file_lst = res_file_lst[:2]
    see1 = parallel_eval(data_file_lst[0], res_file_lst[0])
    with Pool(THREAD) as pool:
        results = pool.starmap(parallel_eval, zip(data_file_lst, res_file_lst))
    results = sorted(results, key=lambda x: x['name'])
    # print(results)
    write_pkl(results, result_to_save)

def process_data(datas, mode):
    result = {}
    for data in datas:
        name = data['name'].split(f'-{mode}-')[0]
        if name not in result.keys():
            result[name] = {
                'sql': [],
                'opt': data['optimal']
            }
        travel_cost = data['res'].F[0]
        result[name]['sql'].append(travel_cost)
    return result
def obt_result(benchmark):

    inputs = [
        f'outputs/{benchmark}/random_EA',
        f'outputs/{benchmark}/novision_EA',
        f'outputs/{benchmark}/vision_EA',
    ]
    datas = []
    modes = []
    for input_file in inputs:
        mode = input_file.split('/')[-1].split('_')[0]
        temp = process_data(open_pkl(input_file), mode=mode)
        modes.append(mode)
        datas.append(temp)

    avgs_all = []
    for name in datas[0].keys():
        output = f"{name} & {float(datas[0][name]['opt']):.0f}"
        means = []
        mins = []

        for data in datas:
            means.append(np.mean(data[name]['sql']))
            mins.append(np.min(data[name]['sql']))
        gaps = []
        argmin_idx = np.argmin(means)
        for idx, data in enumerate(datas):
            optimal = int(float(data[name]['opt']))
            optimal = min([optimal, np.min(mins)])
            # if idx == argmin_idx:
            #     output += f" & {(mins[idx] - optimal) * 100 / optimal:.2f} & \\textbf{{{means[idx]:.1f}}}"
            # else:
            #     output += f" & {(mins[idx] - optimal) * 100 / optimal:.2f} & {means[idx]:.1f}"
            gaps.append((means[idx]-optimal)*100/optimal)
            if idx == argmin_idx:
                output += f" & {int(mins[idx])} & \\textbf{{{(means[idx]-optimal)*100/optimal:.1f}}}\%"
            else:
                output += f" & {int(mins[idx])} & {(means[idx]-optimal)*100/optimal:.1f}\%"
        avgs_all.append(gaps)

        output += '\\\\'
        print(output)
    # res = np.mean(np.array(avgs_all), axis=1)
    # print(f" Average & - & - & {res[0]*100:.2f}\% & - & {res[0]*100:.2f}\% & - & {res[0]*100:.2f}\% \\\\")


    # input_vision = f'outputs/{benchmark}/vision_EA'
    # input_novision = f'outputs/{benchmark}/novision_EA'
    # input_random = f'outputs/{benchmark}/random_EA'
    # vision = process_data(open_pkl(input_vision), mode='vision')
    # novision = process_data(open_pkl(input_novision), mode='novision')
    # random = process_data(open_pkl(input_random), mode='random')
    #
    # for name in vision.keys():
    #     print(f"{name} & {random[name]['opt']} & {np.min(random[name]['sql']):.1f} & {np.mean(random[name]['sql']):.1f} & {np.min(novision[name]['sql']):.1f} & {np.mean(novision[name]['sql']):.1f} & {np.min(vision[name]['sql']):.1f} & {np.mean(vision[name]['sql']):.1f} \\\\")
    #     # print(f'random best: {np.min(random[name]):.1f} random avg: {np.mean(random[name]):.1f} travel cost with vision: best: {np.min(novision[name]):.1f} avg: {np.mean(novision[name]):1.f} travel cost without vision: best: {np.min(vision[name]):.1f} avg: {np.mean(vision[name]):.1f}')

if __name__ == '__main__':
    # MLLM_init()
    # for benchmark in ['small', 'large']:
    # for benchmark in ['small', 'large', 'vlarge']:
    #     for mode in ['random', 'vision', 'novision']:
    #         assessment(scale=benchmark, mode=mode, result_to_save=f'outputs/{benchmark}/{mode}_EA', THREAD=10)

    for benchmark in ['small', 'large', 'vlarge']:
        obt_result(benchmark)

# ssh -p 34622 root@172.26.0.1
# srXtbgNYvQ
# nohup python main_MLLM_SOL.py >/dev/null 2>&1 &
# ps -def | grep main_MLLM_SOL | grep -v grep