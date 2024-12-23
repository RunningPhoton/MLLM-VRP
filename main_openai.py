import os

import openai
from ga_src.env import apikey, temperature, max_tokens, RUNS
from openai_tools import openai_VRPTEST1, openai_VRPTEST2, sol_test, draw_test, obt_table, illustration, illustrate_draw
from FileOperators import data_loading

# Set the model parameters
model_name = "gpt-4-vision-preview"
# model_name = "gpt-3.5-turbo-0613"
# model_name = "gpt-4-1106-preview"
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





if __name__ == '__main__':
    base_url1 = 'https://openai.talentianai.com/v1/'
    # walk()
    # ill_draw()
    for scale in ['large']:
        file_dir = f'vrp_instance/{scale}/'
        source_dir = 'vrp_instance/solved/'
        client = openai.OpenAI(api_key=apikey, base_url=base_url1)
        illustration(client=client, source_dir=[source_dir], file_name=f'{file_dir.split("/")[1]}-vision', input_dirs=[file_dir], model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        illustration(client=client, source_dir=[source_dir], file_name=f'{file_dir.split("/")[1]}-novision', input_dirs=[file_dir], model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        # draw_test([source_dir])
        # draw_test(file_dir_lst=[file_dir])
        for run_id in RUNS:
            openai_VRPTEST1(run_id=run_id, client=client, source_dir=[source_dir], file_name=f'{file_dir.split("/")[1]}-novision', input_dirs=[file_dir], model_name=model_name, temperature=temperature, max_tokens=max_tokens)
            openai_VRPTEST2(run_id=run_id, client=client, source_dir=[source_dir], file_name=f'{file_dir.split("/")[1]}-vision', input_dirs=[file_dir], model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        sol_test(benchmark=f'{scale}/')
        # obt_table(benchmark=f'{scale}/', tm_lmt=150)
