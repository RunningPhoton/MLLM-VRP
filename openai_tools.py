import time
from copy import deepcopy
import openai
from sklearn.metrics import pairwise_distances
import os
from descriptions import *
from FileOperators import data_loading, data_to_description, save_seq_lst, read_seq_lst
from ga_src.env import RUNS
from utils import process, draw, get_logger, sol_in_information, information_to_sol_seq, draw_both, \
    duplicated_and_missed, modity_anyway, obt_result, repair
import base64
import io
from PIL import Image
import pickle
import numpy as np


# from verypy.classic_heuristics.gapvrp import gap_init
def write_pkl(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def open_pkl(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
        return data
def openai_session(client, messages, model_name, temperature, max_tokens, add=None, solving=False):
    try:
        time.sleep(5)
        # response_format = 'text'
        # prompt = None
        new_messages = deepcopy(messages)
        if add is not None:
            new_messages.append({"role": "user", "content": add})
            # response_format = 'xml'
            # prompt = response_sample
        res = client.chat.completions.create(
            model=model_name,
            messages=new_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        infomation = res.choices[0].message
        new_messages.append(infomation)
        if solving == False:
            messages = new_messages
        return infomation, messages
        # messages.append(infomation)
        # message = completion.choices[0].message
        # text = message.content
        # print(text)
        # print()
        # problem_dict = xmltodict.parse(message)
    except openai.APIError as e:
        # Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        return None
    except openai.APIConnectionError as e:
        # Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")
        return None
    except openai.RateLimitError as e:
        # Handle rate limit error (we recommend using exponential backoff)
        print(f"OpenAI API request exceeded rate limit: {e}")
        return None

def logger_init(filename, messages, run_id, full=False):
    logger = get_logger(f'responses/{filename}-{run_id}.log', name=f'responses/{filename}-{run_id}')
    logger.info('start')
    for msg in messages:
        if hasattr(msg, 'role') and msg.role == 'assistant':
            logger.info(msg.content)
        elif full == True and 'role' in msg.keys():
            logger.info(msg['content'])
    return logger

FAILED_TO_MODIFY = 3

def openai_VRPTEST1(run_id, client, source_dir, file_name, input_dirs, model_name, temperature, max_tokens, max_failed=5):
    odir_name, mode = file_name.split('-')
    output_dir = f'outputs/{odir_name}/'
    knowledge_to_save = f'knowledge/source-{mode}-{run_id}'
    def init():
        system_content = "You are an expert in the field of vehicle routing."
        user_input = init_cvrp_learning_1
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_input},
        ]
        information, messages = openai_session(client, messages, model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        solved_CVRPS = data_loading(input_dirs=source_dir)
        for data in solved_CVRPS:
            problem_desc = data_to_description(data, vision=False, solution=True)
            solved_message = [{
                "type": "text",
                "text": problem_desc
            }]
            information, messages = openai_session(client, messages=messages, model_name=model_name, temperature=temperature, max_tokens=max_tokens, add=solved_message)

        information, messages = openai_session(client, messages, model_name=model_name, temperature=temperature, max_tokens=max_tokens, add=last_cvrp_learning_1)
        write_pkl(messages, knowledge_to_save)
    if not os.path.isfile(knowledge_to_save):
        init()
    messages = open_pkl(knowledge_to_save)
    CVRPS = data_loading(input_dirs=input_dirs)
    logger = logger_init(file_name, messages, run_id)

    # CVRPs to solve
    for data in CVRPS[:]:
        problem_desc = data_to_description(data, vision=False, solution=False)
        solved_message = [{
            "type": "text",
            "text": problem_desc
        }]
        information = None
        failed = -1
        seq_lst = None
        # print(sol_in_information(problem_desc, len(data['VERTEX'])))
        while information is None or not sol_in_information(information.content, len(data['VERTEX'])):
            failed += 1
            if failed > FAILED_TO_MODIFY:
                seq_lst = modity_anyway(information.content, len(data['VERTEX']))
                break
            if failed > 0:
                print(f'failed: {failed}')
                print(information.content)

                duplicates_IDs, missed_IDs, errors = duplicated_and_missed(information.content, len(data['VERTEX']))
                more_message = f'''\nThe duplicated customer IDs are given by: {duplicates_IDs}, the missed customer IDs are given by: {missed_IDs} and the customer IDs which should not appear are given by: {errors}\n
                Please remove the duplicated IDs and the IDs should not appear, and add the missed IDs to the route with minimum customers\n**No Explanations Needed!!!**\n'''
                failed_message = [{
                    "type": "text",
                    "text": failed_refine+information.content+more_message
                }]
                information, messages = openai_session(client, messages=messages, model_name=model_name, temperature=temperature, max_tokens=max_tokens, add=failed_message, solving=True)
            else:
                information, messages = openai_session(client, messages=messages, model_name=model_name, temperature=temperature, max_tokens=max_tokens, add=solved_message, solving=True)
            assert failed < max_failed, 'failed too many times'
        if failed <= FAILED_TO_MODIFY:
            seq_lst = information_to_sol_seq(information.content)
        save_seq_lst(seq_lst, output_dir+f'{data["NAME"]}-{mode}-{run_id}')
        data['llm_seq_lst'] = seq_lst
        logger.info(information.content)
    logger.info('finish')

def openai_VRPTEST2(run_id, client, source_dir, file_name, input_dirs, model_name, temperature, max_tokens, max_failed=5):
    odir_name, mode = file_name.split('-')
    output_dir = f'outputs/{odir_name}/'
    knowledge_to_save = f'knowledge/source-{mode}-{run_id}'
    def init():
        system_content = "You are an expert in the field of vehicle routing."
        user_input = init_cvrp_learning_2
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_input},
        ]
        information, messages = openai_session(client, messages, model_name=model_name, temperature=temperature, max_tokens=max_tokens)

        # load solved CVRPs
        solved_CVRPS = data_loading(input_dirs=source_dir)
        for data in solved_CVRPS:
            problem_desc = data_to_description(data, vision=True, solution=True)
            base64_image = draw_both(data['VERTEX'], seq_lst=data['OPTIMAL'], filename=data['NAME'])
            if not client:
                img = Image.open(io.BytesIO(base64.b64decode(base64_image)))
                img.show()
            solved_message = [
                {
                    "type": "text",
                    "text": problem_desc
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
            information, messages = openai_session(client, messages=messages, model_name=model_name, temperature=temperature, max_tokens=max_tokens, add=solved_message)
            # logger.info(information.content)
            # print()
        information, messages = openai_session(client, messages, model_name=model_name, temperature=temperature, max_tokens=max_tokens, add=last_cvrp_learning_2)
        write_pkl(messages, knowledge_to_save)
    # CVRPs to solve
    if not os.path.isfile(knowledge_to_save):
        init()
    CVRPS = data_loading(input_dirs=input_dirs)
    messages = open_pkl(knowledge_to_save)
    logger = logger_init(file_name, messages, run_id)
    for data in CVRPS[:]:
        problem_desc = data_to_description(data, vision=True, solution=False)
        base64_image = draw(data['VERTEX'], title=data['NAME'])
        solved_message = [
            {
                "type": "text",
                "text": problem_desc
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ]
        information = None
        failed = -1
        seq_lst = None
        # print(sol_in_information(problem_desc, len(data['VERTEX'])))
        while information is None or not sol_in_information(information.content, len(data['VERTEX'])):
            failed += 1
            if failed > FAILED_TO_MODIFY:
                seq_lst = modity_anyway(information.content, len(data['VERTEX']))
                break
            if failed > 0:
                print(f'failed: {failed}')
                print(information.content)
                duplicates_IDs, missed_IDs, errors = duplicated_and_missed(information.content, len(data['VERTEX']))
                more_message = f'''\nThe duplicated customer IDs are given by: {duplicates_IDs}, the missed customer IDs are given by: {missed_IDs} and the customer IDs which should not appear are given by: {errors}\n
                                Please remove the duplicated IDs and the IDs should not appear, and add the missed IDs to the route with minimum customers\n**No Explanations Needed!!!**\n'''
                failed_message = [{
                    "type": "text",
                    "text": failed_refine + information.content + more_message
                }]
                information, messages = openai_session(client, messages=messages, model_name=model_name, temperature=temperature, max_tokens=max_tokens, add=failed_message, solving=True)
            else:
                information, messages = openai_session(client, messages=messages, model_name=model_name, temperature=temperature, max_tokens=max_tokens, add=solved_message, solving=True)

            assert failed < max_failed, 'failed too many times'
        if failed <= FAILED_TO_MODIFY:
            seq_lst = information_to_sol_seq(information.content)
        save_seq_lst(seq_lst, output_dir + f'{data["NAME"]}-{mode}-{run_id}')
        data['llm_seq_lst'] = seq_lst
        logger.info(information.content)
    logger.info('finish')

def illustration(client, source_dir, file_name, input_dirs, model_name, temperature, max_tokens, max_failed=5):
    odir_name, mode = file_name.split('-')
    knowledge_to_save = f'knowledge/source-illustration-test-{mode}'
    lnum, rnum = 0, 1
    def init():
        system_content = "You are an expert in the field of vehicle routing."
        user_input = None
        if mode == 'novision':
            user_input = init_cvrp_learning_1
        elif mode == 'vision':
            user_input = init_cvrp_learning_2
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_input},
        ]
        information, messages = openai_session(client, messages, model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        # load solved CVRPs
        solved_CVRPS = data_loading(input_dirs=source_dir)
        for data in solved_CVRPS:

            Cap = int(data['CAPACITY'])
            demand = True
            demands = np.array(data['DEMANDS']).reshape(-1)
            demands = (demands * 100 / Cap).astype(np.int32)

            solved_message = None
            if mode == 'novision':
                problem_desc = data_to_description(data, vision=False, solution=True)
                solved_message = [{
                    "type": "text",
                    "text": problem_desc
                }]
            elif mode == 'vision':
                problem_desc = data_to_description(data, vision=True, solution=True, demand=demand)
                base64_image = draw_both(data['VERTEX'], filename=f'figures/{data["NAME"]}-vision-test-demandis{demand}', seq_lst=data['OPTIMAL'], demands=demands)
                solved_message = [
                    {
                        "type": "text",
                        "text": problem_desc
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            information, messages = openai_session(client, messages=messages, model_name=model_name, temperature=temperature, max_tokens=max_tokens, add=solved_message)
            # logger.info(information.content)
            # print()
        if mode == 'novision':
            information, messages = openai_session(client, messages, model_name=model_name, temperature=temperature, max_tokens=max_tokens, add=last_cvrp_learning_1)
        elif mode == 'vision':
            information, messages = openai_session(client, messages, model_name=model_name, temperature=temperature, max_tokens=max_tokens, add=last_cvrp_learning_2)
        write_pkl(messages, knowledge_to_save)

    # CVRPs to solve
    if not os.path.isfile(knowledge_to_save):
        init()
    CVRPS = data_loading(input_dirs=input_dirs)
    messages = open_pkl(knowledge_to_save)
    logger = logger_init(file_name, messages, 'test', full=False)

    for data in CVRPS[lnum:rnum]:
        solved_message = None
        problem_desc = None

        Cap = int(data['CAPACITY'])
        demand = True
        demands = np.array(data['DEMANDS']).reshape(-1)
        demands = (demands * 100 / Cap).astype(np.int32)

        if mode == 'novision':
            more_msg = f'''
            \nYou can explain it according to the pairwise distances between customers (Euclidean distance), which can be easily obtained via 'calculator' in terms of 'X-axis' and 'Y-axis' that are given in the XML text.
            \nHere is an example of route constructing for a route given by [A, B, C, D]:
            \nThe vehicle serving sequence started with A as A is the customer that near the depot. Further, the vehicle served B and C in sequence, following the principle of proximity. Finally, the vehicle visited D and back to depot, where the entire route can be viewed as a convex polygon.
            '''
            problem_desc = data_to_description(data, vision=False, solution=False, illustration=more_msg)
            solved_message = [{
                "type": "text",
                "text": problem_desc
            }]
        elif mode == 'vision':
            more_msg = f'''
            \nYou can explain it according to the pairwise distances between customers (Euclidean distance), which can be easily obtained via 'calculator' in terms of 'X-axis' and 'Y-axis', or observed from the layout map of vertexes.
            \nHere is an example of route constructing for a route given by [A, B, C, D]:
            \nThe vehicle serving sequence started with A as A is the customer that near the depot. Further, the vehicle served B and C in sequence, following the principle of proximity. Finally, the vehicle visited D and back to depot, where the entire route can be viewed as a convex polygon.
            '''
            problem_desc = data_to_description(data, vision=True, solution=False, illustration=more_msg)
            base64_image = draw(data['VERTEX'], filename=f'figures/{data["NAME"]}-vision-test-demandis{demand}', title=data['NAME'], demands=demands)
            solved_message = [
                {
                    "type": "text",
                    "text": problem_desc
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        information = None
        failed = -1
        seq_lst = None
        # print(sol_in_information(problem_desc, len(data['VERTEX'])))
        while information is None or not sol_in_information(information.content, len(data['VERTEX'])):
            failed += 1
            if failed > FAILED_TO_MODIFY:
                seq_lst = modity_anyway(information.content, len(data['VERTEX']))
                break
            if failed > 0:
                print(f'failed: {failed}')
                print(information.content)
                duplicates_IDs, missed_IDs, errors = duplicated_and_missed(information.content, len(data['VERTEX']))
                more_message = f'''\nThe duplicated customer IDs are given by: {duplicates_IDs}, the missed customer IDs are given by: {missed_IDs} and the customer IDs which should not appear are given by: {errors}\n
                                Please remove the duplicated IDs and the IDs should not appear, and add the missed IDs to the route with minimum customers\n**No Explanations Needed!!!**\n'''
                failed_message = [{
                    "type": "text",
                    "text": failed_refine + information.content + more_message
                }]
                information, messages = openai_session(client, messages=messages, model_name=model_name, temperature=temperature, max_tokens=max_tokens, add=failed_message, solving=True)
            else:
                information, messages = openai_session(client, messages=messages, model_name=model_name, temperature=temperature, max_tokens=max_tokens, add=solved_message, solving=True)
            assert failed < max_failed, 'failed too many times'
        logger.info(problem_desc)
        logger.info(information.content)
        if failed <= FAILED_TO_MODIFY:
            seq_lst = information_to_sol_seq(information.content)
        # data['llm_seq_lst'] = seq_lst
        dist = pairwise_distances(X=data['VERTEX'], metric='euclidean')
        demands = np.array(data['DEMANDS']).reshape(-1)
        capacity = int(data['CAPACITY'])
        cost, real_lst = repair(seq_lst, dist, demands, capacity)
        logger.info(f"\nRepaired:\n{real_lst}\nCOST: {cost}\n")
        # more_msg = None
        # if mode == 'novision':
        #     more_msg = f'''
        #     \nYou can explain it according to the pairwise distances between customers (Euclidean distance), which can be easily obtained via 'calculator' in terms of 'X-axis' and 'Y-axis' that are given in the XML text of previous session.
        #     \nHere is an example of route constructing for a route given by [3, 5, 10, 2]:
        #     \nThe vehicle serving sequence started with 3 as 3 is the customer that near the depot. Further, the vehicle served 5 and 10 in sequence, following the principle of proximity. Finally, the vehicle visited 2 and back to depot, where the entire route can be viewed as a convex polygon.
        #     '''
        # elif mode == 'vision':
        #     more_msg = f'''
        #     \nYou can explain it according to the pairwise distances between customers (Euclidean distance), which can be easily obtained via 'calculator' in terms of 'X-axis' and 'Y-axis', or observed from the layout map that are given in previous session.
        #     \nHere is an example of route constructing for a route given by [3, 5, 10, 2]:
        #     \nThe vehicle serving sequence started with 3 as 3 is the customer that near the depot. Further, the vehicle served 5 and 10 in sequence, following the principle of proximity. Finally, the vehicle visited 2 and back to depot, where the entire route can be viewed as a convex polygon.
        #     '''
        # ill_msg = [{
        #     "type": "text",
        #     "text": f'The solution you provide for above problem namely {data["NAME"]} is:\n{information.content}\nCan you also explain how you construct each route accordingly?\n{more_msg}'
        # }]
        # logger.info(ill_msg[0]['text'])
        # information, messages = openai_session(client, messages=messages, model_name=model_name,temperature=temperature,max_tokens=max_tokens, add=ill_msg, solving=True)
        # logger.info(information.content)
    logger.info('finish')


# def parallel_evaluate(knowledge_to_save_data, oper_seq, oid):
#     temp = {
#         'id': oid,
#         'seq': oper_seq,
#         'gap': [],
#         'runtime': []
#     }
#     CVRPS = open_pkl(knowledge_to_save_data)[:12]
#     for cvrp in CVRPS:
#         init_sol = cvrp['nearest_init']
#         D = pairwise_distances(X=cvrp['VERTEX'], metric='euclidean')
#         C = int(cvrp['CAPACITY'])
#         d = np.reshape(cvrp['DEMANDS'], -1)
#         L = np.inf
#
#         new_sol, t_cost, run_tm = local_search(init_sol, D, C, d, L, optlst=oper_seq, max_length=LENGTH, iterations=LS_ITERATION)
#         opt_cost = check_and_calc(cvrp['OPTIMAL'], D, d, C)
#         gap = (t_cost - opt_cost) / opt_cost * 100  # x%
#         temp['gap'].append(gap)
#         temp['runtime'].append(run_tm)
#     temp['avg_gap'] = np.mean(temp['gap'])
#     temp['avg_time'] = np.mean(temp['runtime'])
#     return temp
# def openai_heuristic(run_id, client, source_dir, file_name, input_dirs, model_name, temperature, max_tokens, MAX_ITER=10, THREAD=POP_SIZE):
#     odir_name, mode = file_name.split('-')
#     knowledge_to_save = f'knowledge/metaheuristic-{mode}-{run_id}'
#     knowledge_to_save_data = f'knowledge/metaheuristic-{mode}-nearest-{run_id}'
#     result_to_save = f'outputs/heuristics/metaheuristic-{mode}-{run_id}'
#
#     def init():
#         system_content = "You are an expert in the field of vehicle routing."
#         user_input = init_metah_learning[mode]
#         messages = [
#             {"role": "system", "content": system_content},
#             {"role": "user", "content": user_input},
#         ]
#         information, messages = openai_session(client, messages, model_name=model_name, temperature=temperature, max_tokens=max_tokens)
#         # load solved CVRPs
#         solved_CVRPS = data_loading(input_dirs=source_dir)
#         for data in solved_CVRPS:
#
#             solved_message = None
#             if mode == 'novision':
#                 problem_desc = data_to_description(data, vision=False, solution=True, metaLS=True)
#                 solved_message = [{
#                     "type": "text",
#                     "text": problem_desc
#                 }]
#             elif mode == 'vision':
#                 problem_desc = data_to_description(data, vision=True, solution=True, metaLS=True)
#                 base64_image = draw_both(data['VERTEX'], filename=None, seq_lst=data['OPTIMAL'])
#                 solved_message = [
#                     {
#                         "type": "text",
#                         "text": problem_desc
#                     },
#                     {
#                         "type": "image_url",
#                         "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
#                     }
#                 ]
#             information, messages = openai_session(client, messages=messages, model_name=model_name, temperature=temperature, max_tokens=max_tokens, add=solved_message)
#             # logger.info(information.content)
#         information, messages = openai_session(client, messages, model_name=model_name, temperature=temperature, max_tokens=max_tokens, add=mid_metah_learning)
#         # oper_lsts = information_to_oper_seq(information.content)
#         write_pkl(messages, knowledge_to_save)
#     def init2():
#         CVRPS = data_loading(input_dirs=input_dirs)
#         # obtain the initial solution for each CVRP
#         for cvrp in CVRPS:
#             dist = pairwise_distances(X=cvrp['VERTEX'], metric='euclidean')
#             demands = np.array(cvrp['DEMANDS']).reshape(-1)
#             capacity = int(cvrp['CAPACITY'])
#             # cvrp['beam_init'] = beam_search(DM=dist, vehicle_capacity=capacity, demands=demands, beam_width=100)
#             cvrp['nearest_init'] = nearest_neighbor_init(D=dist, d=demands, C=capacity, L=None)
#             # cvrp['gap_init'] = gap_init(D=dist, d=demands, C=capacity, L=None)
#             cvrp['ls_pop'] = []
#         write_pkl(CVRPS, knowledge_to_save_data)
#     # CVRPs to solve
#     if not os.path.isfile(knowledge_to_save):
#         init()
#     if not os.path.isfile(knowledge_to_save_data):
#         init2()
#     messages = open_pkl(knowledge_to_save)
#     # CVRPS = open_pkl(knowledge_to_save_data)[:2]
#     new_oper_lsts = information_to_oper_seq(messages[-1].content)
#     logger = logger_init(file_name+f'-heuristic', messages, run_id, full=False)
#     # previous_oper_lsts = []
#     results = []
#     # result = None
#     with Pool(THREAD) as pool:
#         result = pool.starmap(parallel_evaluate, zip(repeat(knowledge_to_save_data), new_oper_lsts, list(range(1, len(new_oper_lsts) + 1))))
#     result = sorted(result, key=lambda x: x['avg_gap'])
#     results.append(result)
#
#     for iter_id in range(1, MAX_ITER):
#         problem_desc = meta_description(iter_id, result)
#         information, messages = openai_session(client, messages=messages, model_name=model_name, temperature=temperature, max_tokens=max_tokens, add=problem_desc, solving=False)
#         logger.info(problem_desc)
#         logger.info(information.content)
#         # previous_oper_lsts = deepcopy(new_oper_lsts)
#         new_oper_lsts = information_to_oper_seq(information.content)
#         with Pool(THREAD) as pool:
#             result = pool.starmap(parallel_evaluate, zip(repeat(knowledge_to_save_data), new_oper_lsts, list(range(1, len(new_oper_lsts) + 1))))
#         result = results[-1] + result
#         result = sorted(result, key=lambda x: x['avg_gap'])[:POP_SIZE]
#         results.append(result)
#     problem_desc = meta_description(MAX_ITER, result)
#     logger.info(problem_desc)
#         # select top 20 execution sequences
#
#
#     write_pkl(results, result_to_save)
#
#     logger.info('finish')
def illustrate_draw(filedirs, novision, vision):
    cvrp = data_loading(input_dirs=filedirs)[0]
    opt = cvrp['OPTIMAL']
    # novisionls = local_search(novision, cvrp)
    # visionls = local_search(vision, cvrp)

    print('bug')
    # _, rand_lst = get_random_solution(cvrp)
    # draw(cvrp['VERTEX'], title=cvrp['NAME'], filename='outputs/pictures/'+cvrp['NAME']+'-optimal', seq_lst=opt, color=True, addr='eps')
    # draw(cvrp['VERTEX'], title=cvrp['NAME'], filename='outputs/pictures/'+cvrp['NAME']+'-novision', seq_lst=novision, color=True, addr='eps')
    # draw(cvrp['VERTEX'], title=cvrp['NAME'], filename='outputs/pictures/'+cvrp['NAME']+'-vision', seq_lst=vision, color=True, addr='eps')
    # draw(cvrp['VERTEX'], title=cvrp['NAME'], filename='outputs/pictures/'+cvrp['NAME']+'-random', seq_lst=rand_lst, color=True, addr='eps')

def draw_test(file_dir_lst, NUM_SOLVED=3, NUM_TO_SOLVE=0):
    CVRPS = data_loading(input_dirs=file_dir_lst)
    # for data in CVRPS[:1]:
    #     base64_image = draw(data['VERTEX'], seq_lst=data['OPTIMAL'], title=data['NAME'])
    #     img = Image.open(io.BytesIO(base64.b64decode(base64_image)))
    #     img.show()
    for data in CVRPS[:4]:
        base64_image = draw_both(data['VERTEX'], seq_lst=data['OPTIMAL'], filename=f"figures/{data['NAME']}")
        # img = Image.open(io.BytesIO(base64.b64decode(base64_image)))
        # img.show()
    for data in CVRPS[:4]:
        base64_image = draw(data['VERTEX'], seq_lst=data['OPTIMAL'], title=data['NAME'], filename=f"figures/{data['NAME']}-alone-solution")
        base64_image = draw(data['VERTEX'], seq_lst=None, title=data['NAME'], filename=f"figures/{data['NAME']}-alone")
        # img = Image.open(io.BytesIO(base64.b64decode(base64_image)))
        # img.show()
def sol_test(benchmark):
    CVRPS = data_loading(input_dirs=[f'vrp_instance/{benchmark}'])
    file_dir = f'outputs/{benchmark}'
    for idx, data in enumerate(CVRPS):
        no_vision_lst = read_seq_lst(f'{file_dir}{data["NAME"]}-novision-1')
        vision_lst = read_seq_lst(f'{file_dir}{data["NAME"]}-vision-1')
        process(data, LLM_SEQ_LST_NOVISION=no_vision_lst, LLM_SEQ_LST_VISION=vision_lst, color=True)
        for k in ['img_opt', 'img_bs', 'img_rand', 'img_llm_novision', 'img_llm_vision']:
            if k in data.keys():
                img = Image.open(io.BytesIO(base64.b64decode(data[k])))
                img.show()
        # Beam Search

def obt_table(benchmark, tm_lmt):
    file_dir = f'outputs/{benchmark}'
    file = file_dir + 'result.pkl'
    if not os.path.isfile(file):
        CVRPS = data_loading(input_dirs=[f'vrp_instance/{benchmark}'])
        results = {}
        for idx, data in enumerate(CVRPS):
            result = {
                'cost_init': [],
                'cost_novision': [],
                'cost_vision': [],
                'costor_novision': [],
                'costor_vision': [],
            }
            for run_id in RUNS:
                novision = read_seq_lst(f'{file_dir}{data["NAME"]}-novision-{run_id}')
                vision = read_seq_lst(f'{file_dir}{data["NAME"]}-vision-{run_id}')
                cost_init, cost_novision, cost_vision, costor_novision, costor_vision = obt_result(data, novision, vision, tm_lmt=tm_lmt)
                # cost_init, cost_novision, cost_vision = obt_result(data, novision, vision, tm_lmt=10)
                result['cost_init'].append(cost_init)
                result['cost_novision'].append(cost_novision)
                result['cost_vision'].append(cost_vision)
                result['costor_novision'].append(costor_novision)
                result['costor_vision'].append(costor_vision)
            results[data['NAME']] = result
        write_pkl(results, file)
    results = open_pkl(file)
    cmps = ['cost_init','cost_novision','cost_vision','costor_novision','costor_vision']
    # cmps = ['cost_init','cost_novision','cost_vision']
    for name in results.keys():
        result = results[name]
        res = {'name': name}
        for key in cmps:
            res[f'{key}-mean'] = np.mean(result[key])
            res[f'{key}-lowest'] = np.min(result[key])
        for key, val in res.items():
            if key != 'name':
                print(f'{key}: {val:.1f}', end=' ')
            else:
                print(f'{key}: {val}', end=' ')
        print()
        # if res['cost_novision-mean'] < res['cost_vision-mean']:
        #     print(f"{res['name']} & {res['cost_init-mean']:.0f} & "
        #           f"{res['cost_novision-lowest']:.0f} & \\textbf{{{res['cost_novision-mean']:.0f}}} & \\textbf{{{(res['cost_novision-mean']-res['cost_init-mean'])*100/res['cost_init-mean']:.0f}}}\\% & "
        #           f"{res['cost_vision-lowest']:.0f} & {res['cost_vision-mean']:.0f} & {(res['cost_vision-mean']-res['cost_init-mean'])*100/res['cost_init-mean']:.0f}\\% \\\\")
        # else:
        #     print(f"{res['name']} & {res['cost_init-mean']:.0f} & "
        #           f"{res['cost_novision-lowest']:.0f} & {res['cost_novision-mean']:.0f} & {(res['cost_novision-mean'] - res['cost_init-mean']) * 100 / res['cost_init-mean']:.0f}\\% & "
        #           f"{res['cost_vision-lowest']:.0f} & \\textbf{{{res['cost_vision-mean']:.0f}}} & \\textbf{{{(res['cost_vision-mean'] - res['cost_init-mean']) * 100 / res['cost_init-mean']:.0f}}}\\% \\\\")
