from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from PIL import Image
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.utils import obs_as_tensor
import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('/PSRP/')
from problem_solvers.gnn.PPO.gen_env import model_for_nn
from problem_solvers.gnn.PPO.settings import *
from data_generators.DataBuilder_MPPSRP_Simple import DataBuilder_MPPSRP_Simple
from problem_solvers.gnn.PPO.env.irp_env_custom import IRPEnv_Custom
from problem_solvers.gnn.dataset_utils import get_graph_dict
from problem_solvers.gnn.graph_generation_functions import create_wheel_noised

import numpy as np
import torch as th

def predict_proba(model, state, lstm_states, episode_start):
    states = th.tensor(lstm_states[0], dtype=th.float32, device=model.policy.device), th.tensor(
                lstm_states[1], dtype=th.float32, device=model.policy.device)
    # lstm_states = th.tensor(lstm_states, dtype=th.float32, device=model.policy.device)
    episode_starts = th.tensor(episode_start, dtype=th.float32, device=model.policy.device)
    obs = obs_as_tensor(state, model.policy.device)
    dis, _ = model.policy.get_distribution(obs, states, episode_starts)
    probs = dis.distribution.probs
    probs_np = probs.detach().cpu().numpy()
    return probs_np

def add_bar_chart_to_image(img, probs):
    # 1. Создаем бар график для вероятностей
    fig, ax = plt.subplots(figsize=(5, 2))  # Устанавливаем размер графика
    actions = np.arange(len(probs))
    ax.bar(actions, probs, color='blue')
    ax.set_xlabel('Actions')
    ax.set_ylabel('Probability')
    ax.set_title('Action Probabilities')
    ax.set_ylim(0, 1)  # Вероятности в диапазоне [0, 1]

    # Убираем лишние отступы
    fig.tight_layout()

    # 2. Конвертируем график в изображение
    fig.canvas.draw()
    chart_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    chart_img = chart_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    chart_img = Image.fromarray(chart_img)

    # Закрываем график, чтобы освободить ресурсы
    plt.close(fig)

    # 3. Объединяем основное изображение и бар график
    total_width = max(img.width, chart_img.width)
    total_height = img.height + chart_img.height

    # Создаем новое изображение с дополнительным местом для графика
    combined_img = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
    
    # Вставляем основное изображение
    combined_img.paste(img, (0, 0))
    
    # Вставляем график ниже
    combined_img.paste(chart_img, (0, img.height))

    return combined_img

n_steps = 500
n = 6
veh = 6
text = 'final'
parameters_dict['load_policy'] = 'full_fill'
parameters_dict['max_trips'] = 1
parameters_dict['num_nodes'] = n
parameters_dict['k_vehicles'] = veh





generation_function = create_wheel_noised
synth_data = generation_function(parameters_dict['num_nodes'])

data_builder = DataBuilder_MPPSRP_Simple(synth_data["weight_matrix"],
                                         distance_multiplier=1, travel_time_multiplier=60 * 60,
                                         planning_horizon=parameters_dict['planning_horizon'], # горизонт
                                         safety_level=0.05, max_level=0.95,
                                         initial_inventory_level=0.5, tank_capacity=100,
                                         depot_service_time=15 * 60, station_service_time=15 * 60,
                                         demand=10, products_count=parameters_dict['products_count'],
                                         k_vehicles=parameters_dict["k_vehicles"],
                                         compartments=[parameters_dict["products_count"] * [
                                             parameters_dict["compartment_capacity"]]],
                                         mean_vehicle_speed=60, vehicle_time_windows=[[9 * 60 * 60, 18 * 60 * 60]],
                                         noise_initial_inventory=0.0, noise_tank_capacity=0.0,
                                         noise_compartments=0.0, noise_demand=0.0,
                                         noise_vehicle_time_windows=0.0,
                                         noise_restrictions=0.0,
                                         random_seed=45)

graph_data = data_builder.build_data_model()

model_for_nn = get_graph_dict(synth_data=synth_data,
                              parameters_dict=parameters_dict,
                              graph_data=graph_data)

env_irp = IRPEnv_Custom(model_for_nn, parameters_dict)

model = RecurrentPPO.load(rf"logs/Rec-PPO_NN_old-env_nsteps-{n_steps}_nodes-{n}_veh-{veh}_{text}/best_model.zip", env=env_irp)


eval_env = model.get_env()
obs = eval_env.reset()

done = False

lstm_states = None
num_envs = 1
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)
steps = 0
actions = []
embedings = []
image_arrays = []
capacities_list = []

capacity_rewards = []
distance_rewards = []
time_end_penalties = []
empty_load_penalties = []
restricted_station_penalties = []
revisit_penalties = []
dry_runs_penalties = []
new_day = []
visited_actions = set()
set_completed = False  # Флаг, указывающий, что все значения были посещены
total_distance_fixed = False  # Флаг для фиксации total_distance

while not done:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    probs = predict_proba(model, obs, lstm_states=lstm_states, episode_start=episode_starts)
    obs, rewards, done, info = eval_env.step(action)

    print(f"{info[0]['total_travel_distance'] = }")
    
    current_action = action.item()
    visited_actions.add(current_action)  # Добавляем значение в множество

    # Проверка, если все значения от 0 до n посещены
    if visited_actions == set(range(n)) and not set_completed:
        set_completed = True  # Устанавливаем флаг, что все значения посещены

    # После заполнения множества ожидаем один дополнительный шаг и фиксируем total_distance
    elif set_completed and not total_distance_fixed:
        total_distance = info[0]['total_travel_distance']
        print(f'Все значения от 0 до {n} посещены + 1 шаг. total_distance = {total_distance}')
        total_distance_fixed = True  # Устанавливаем флаг, что total_distance зафиксирован


    time_end_penalties.append(info[0]['time_end_penalties'])
    empty_load_penalties.append(info[0]['empty_load_penalties'])
    restricted_station_penalties.append(info[0]['restricted_station_penalties'])
    revisit_penalties.append(info[0]['revisit_penalties'])
    dry_runs_penalties.append(info[0]['dry_runs_penalties'])
    capacity_rewards.append(info[0]["capacity_rewards"])
    distance_rewards.append(info[0]['distance_rewards'])

    day_end = (eval_env.envs[0].day_end)
    new_day.append(day_end)

    # сохранить в df
    capacities_list.append(eval_env.envs[0].init_capacities[:, 0].tolist())
    # print(eval_env.envs[0].cur_day.item())

    episode_starts = done
    # img = eval_env.envs[0].render()  # Ваше основное изображение
    # combined_img = add_bar_chart_to_image(img, probs[0])
    # image_arrays.append(combined_img)

    # actions.append(int(action))



    # prob = predict_proba(model, obs)

    # print(np.round(prob, 2))
    # print(f"total_travel_distance: {info[0]['total_travel_distance']}")
    # print(f"dry_runs: {info[0]['dry_runs']}")
    # print(f"average_vehicle_utilization: {info[0]['average_vehicle_utilization']}")
    # print(f"average_stops_per_trip: {info[0]['average_stops_per_trip']}")

    steps += 1
    # print(rewards)
    # print(done)
    # if steps == 300:
    if done:
        print(f"{info[0]['total_travel_distance'] = }")
        print(f"{info[0]['total_travel_time'] = }")
        print(f"{info[0]['average_vehicle_utilization'] = }")
        print(f"{info[0]['average_stops_per_trip'] = }")
        print(f"{info[0]['dry_runs'] = }")

        # Сохранение штрафов в соответствующие списки
        model_kpi = info[0]
        
        total = np.array(capacity_rewards) +\
            np.array(dry_runs_penalties) +\
            np.array(distance_rewards) +\
            np.array(time_end_penalties) +\
            np.array(empty_load_penalties)+\
            np.array(restricted_station_penalties) +\
            np.array(revisit_penalties)
            
        def add_vertical_lines(ax, new_day_series, alpha=0.3):
            """Добавляем вертикальные линии, если new_day = True."""
            for idx, day_end in enumerate(new_day_series):
                if day_end:
                    ax.axvline(x=idx, color='red', linestyle='--', label=None, alpha=alpha)

        result = {
            'dry_runs_penalties': dry_runs_penalties,
            'capacity_rewards': capacity_rewards,
            'distance_rewards': distance_rewards,
            'dist_cap': np.array(capacity_rewards) + np.array(distance_rewards),
            'time_end_penalties': time_end_penalties,
            'empty_load_penalties': empty_load_penalties,
            'restricted_station_penalties': restricted_station_penalties,
            'revisit_penalties': revisit_penalties,
            'total_reward': total,
            'new_day': new_day
        }

        df = pd.DataFrame(result)

        # Создание графиков
        fig = plt.figure(figsize=(14, 16))
        gs = gridspec.GridSpec(len(df.columns), 1, height_ratios=[1] * (len(df.columns) - 1) + [2])

        # Для каждой колонки, кроме 'new_day', создаем подграфик
        for i, column in enumerate(df.columns[:-1]):
            ax = fig.add_subplot(gs[i])
            ax.plot(df[column], label=column)
            ax.set_title(column)
            # ax.set_ylim(-1, 1)
            ax.grid()
            add_vertical_lines(ax, df['new_day'])
            ax.legend()

        # Добавление последнего графика внизу
        last_ax = fig.add_subplot(gs[-1])
        pd.DataFrame(capacities_list, columns=[f'{i}' for i in range(n)]).plot(ax=last_ax)
        last_ax.set_title('Capacities Plot')
        last_ax.grid()
        add_vertical_lines(last_ax, df['new_day'])

        # Общий заголовок и отображение графиков
        plt.tight_layout()
        plt.savefig(f"images/rewards_{n}_{n_steps}_{veh}_{text}.png")


    # print()
# print(actions)
# print()

# print('Saving gif...')
# image_arrays[0].save('output.gif', save_all=True, append_images=image_arrays[1:], duration=1000, loop=0)
# print('Gif saved!')

# for idx, img in enumerate(image_arrays):
#     img.save(f'images/output{idx}.png')


from datetime import datetime
from pprint import pprint
import time

# from problem_solvers.gnn.PPO.gen_env import graph_data
# from problem_solvers.gnn.PPO.settings import parameters_dict
from problem_solvers.mppsrp.cpsat.TaskBuilder_MPPSRP_FullMILP_CPSAT import TaskBuilder_MPPSRP_FullMILP_CPSAT
from problem_solvers.mppsrp.cpsat.TaskSolver_MPPSRP_CPSAT import TaskSolver_MPPSRP_CPSAT
# from problem_solvers.mppsrp.data_generators.DataBuilder_MPPSRP_FullMILP_CPSAT import DataBuilder_MPPSRP_FullMILP_CPSAT

from paths_config import interim_dir

task_builder = TaskBuilder_MPPSRP_FullMILP_CPSAT( max_trips_per_day=parameters_dict['max_trips'], verbose=True )
task = task_builder.build_task( graph_data )

print("Solving start time: {}".format( datetime.now() ))
task_solver = TaskSolver_MPPSRP_CPSAT( cache_dir=interim_dir,
                                       cache_all_feasible_solutions=False,
                                       solution_prefix='dataset',
                                       time_limit_milliseconds=60_000)
start_time = time.time()
solution = task_solver.solve( task )
delta_time = time.time() - start_time
print('kpi')
cpsat_kpi = solution.get_kpis()
pprint( cpsat_kpi )

print('routes_schedule')
routes_schedule = solution.get_routes_schedule()
pprint( routes_schedule )

print("done")

import pandas as pd

df_model_kpi = pd.DataFrame(list(model_kpi.items()), columns=['Metric', 'Model KPI'])
df_cpsat_kpi = pd.DataFrame(list(cpsat_kpi.items()), columns=['Metric', 'CPSAT KPI'])

df_kpi = pd.merge(df_model_kpi, df_cpsat_kpi, on='Metric', how='outer')

df_kpi.to_csv('out.csv')