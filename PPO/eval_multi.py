import time
from pprint import pprint
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from PIL import Image
from sb3_contrib import MaskablePPO, RecurrentPPO
from stable_baselines3 import PPO
import numpy as np
import torch as th
import os

import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('/PSRP/')

# Импортируйте ваши функции и классы
from problem_solvers.gnn.PPO.gen_env import model_for_nn
from problem_solvers.gnn.PPO.settings import *
from data_generators.DataBuilder_MPPSRP_Simple import DataBuilder_MPPSRP_Simple
from problem_solvers.gnn.PPO.enviroment_PyG import IRPEnv_Custom
from problem_solvers.gnn.dataset_utils import get_graph_dict
from problem_solvers.gnn.graph_generation_functions import create_wheel_noised
from tqdm import tqdm
from stable_baselines3.common.utils import obs_as_tensor
n_steps = 500
import os
import re


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


# Путь к папке с результатами TensorBoard
base_dir = fr"logs_all/"
# output_dir = r"C:\Users\pc507\Desktop\Sber_PPO\Gnn_VRP\experiments"  # Замените на ваш выходной каталог
#
# # Создание директории, если она не существует
# os.makedirs(output_dir, exist_ok=True)
# Словарь для хранения уникальных значений nodes и veh
unique_params = set()

# Перебор всех папок в указанной директории
for root, dirs, files in os.walk(base_dir):
    for dir_name in dirs:
        print(dir_name)
        # Проверяем наличие ключевого слова 'final' в имени папки
        if 'nsteps-500_nodes-6_veh-6_final' in dir_name:
            # Используем регулярное выражение для извлечения значений nodes и veh
            match = re.search(r'nsteps-(\d+)_nodes-(\d+)_veh-(\d+)', dir_name)
            if match:
                n_steps_value = match.group(1)
                nodes_value = match.group(2)
                veh_value = match.group(3)
                # Добавляем уникальные параметры в множество
                unique_params.add((n_steps_value, nodes_value, veh_value))

# Вывод уникальных комбинаций nodes и veh
# for n_steps, nodes, veh in unique_params:
#     print(f"Nodes: {nodes}, Vehicles: {veh}")

# Перебор значений количества точек и машин
for n_steps, n, veh in tqdm(unique_params):
        print(f"N-Steps: {n_steps}, Nodes: {n}, Vehicles: {veh}")
        n = int(n)
        veh = int(veh)
        text = 'final'
        parameters_dict['load_policy'] = 'full_fill'
        parameters_dict['max_trips'] = 1
        parameters_dict['num_nodes'] = n
        parameters_dict['k_vehicles'] = veh

        generation_function = create_wheel_noised
        synth_data = generation_function(parameters_dict['num_nodes'])

        data_builder = DataBuilder_MPPSRP_Simple(synth_data["weight_matrix"],
                                                 distance_multiplier=1, travel_time_multiplier=60 * 60,
                                                 planning_horizon=parameters_dict['planning_horizon'],
                                                 safety_level=0.05, max_level=0.95,
                                                 initial_inventory_level=0.5, tank_capacity=100,
                                                 depot_service_time=15 * 60, station_service_time=15 * 60,
                                                 demand=10, products_count=parameters_dict['products_count'],
                                                 k_vehicles=parameters_dict["k_vehicles"],
                                                 compartments=[parameters_dict["products_count"] * [
                                                     parameters_dict["compartment_capacity"]]],
                                                 mean_vehicle_speed=60,
                                                 vehicle_time_windows=[[9 * 60 * 60, 18 * 60 * 60]],
                                                 noise_initial_inventory=0.0, noise_tank_capacity=0.0,
                                                 noise_compartments=0.0, noise_demand=0.0,
                                                 noise_vehicle_time_windows=0.0,
                                                 noise_restrictions=0.0,
                                                 random_seed=45)

        graph_data = data_builder.build_data_model()
        model_for_nn = get_graph_dict(synth_data=synth_data, parameters_dict=parameters_dict, graph_data=graph_data)

        env_irp = IRPEnv_Custom(model_for_nn, parameters_dict)
        model = RecurrentPPO.load(rf"logs_all/Rec-PPO_NN_old-env_nsteps-{n_steps}_nodes-{n}_veh-{veh}_{text}/best_model.zip", env=env_irp)

        eval_env = model.get_env()
        obs = eval_env.reset()

        done = False
        lstm_states = None
        num_envs = 1
        episode_starts = np.ones((num_envs,), dtype=bool)
        steps = 0

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
        total_distance = 0

        while not done:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            probs = predict_proba(model, obs, lstm_states=lstm_states, episode_start=episode_starts)
            obs, rewards, done, info = eval_env.step(action)
            
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

            # Сбор данных для метрик
            time_end_penalties.append(info[0]['time_end_penalties'])
            empty_load_penalties.append(info[0]['empty_load_penalties'])
            restricted_station_penalties.append(info[0]['restricted_station_penalties'])
            revisit_penalties.append(info[0]['revisit_penalties'])
            dry_runs_penalties.append(info[0]['dry_runs_penalties'])
            capacity_rewards.append(info[0]["capacity_rewards"])
            distance_rewards.append(info[0]['distance_rewards'])

            day_end = eval_env.envs[0].day_end
            new_day.append(day_end)

            capacities_list.append(eval_env.envs[0].init_capacities[:, 0].tolist())

            episode_starts = done
            steps += 1

        # Обработка результатов
        total = np.array(capacity_rewards) + np.array(dry_runs_penalties) + np.array(distance_rewards) + np.array(time_end_penalties) + np.array(empty_load_penalties) + np.array(restricted_station_penalties) + np.array(revisit_penalties)
        model_kpi = info[0]
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
            'new_day': new_day,
            'total_distance': total_distance
        }

        df = pd.DataFrame(result)

        # Создание папки для сохранения результатов
        output_dir = f"results/n_steps-{n_steps}/nodes-{n}/vehicles-{veh}/"
        os.makedirs(output_dir, exist_ok=True)

        # Сохранение CSV
        df.to_csv(os.path.join(output_dir, f'kpi_n{n}_veh{veh}.csv'), index=False)

        import matplotlib.pyplot as plt
        import os
        import pandas as pd
        from matplotlib import gridspec

        # Создание общего графика
        fig = plt.figure(figsize=(14, 16))
        gs = gridspec.GridSpec(len(df.columns) - 1, 1, height_ratios=[1] * (len(df.columns) - 2) + [2])

        # Создание подграфиков для каждой колонки, кроме 'new_day'
        for i, column in enumerate(df.columns[:-2]):
            ax = fig.add_subplot(gs[i])
            ax.plot(df[column], label=column)
            ax.set_title(column)
            ax.grid()
            ax.legend()

        # Добавление последнего графика внизу
        last_ax = fig.add_subplot(gs[-1])
        pd.DataFrame(capacities_list, columns=[f'{i}' for i in range(n)]).plot(ax=last_ax)
        last_ax.set_title('Capacities Plot')
        last_ax.grid()

        # Общий заголовок и отображение графиков
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'rewards_{n}_{n_steps}_{veh}.png'))  # Сохранение общего графика
        plt.close(fig)  # Закрываем общий график

        # Создание отдельного графика для last_ax
        fig_capacity = plt.figure(figsize=(8, 6))
        last_ax_capacity = fig_capacity.add_subplot(111)
        pd.DataFrame(capacities_list, columns=[f'{i}' for i in range(n)]).plot(ax=last_ax_capacity)
        last_ax_capacity.set_title('Capacities Plot')
        last_ax_capacity.grid()

        # Сохранение графика вместимости
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'capacity_plot_n{n}_veh{veh}.png'))
        plt.close(fig_capacity)  # Закрываем фигуру для last_ax

        model_kpi['total_distance'] = total_distance


        # Запуск CPSAT
        from datetime import datetime
        from problem_solvers.mppsrp.cpsat.TaskBuilder_MPPSRP_FullMILP_CPSAT import TaskBuilder_MPPSRP_FullMILP_CPSAT
        from problem_solvers.mppsrp.cpsat.TaskSolver_MPPSRP_CPSAT import TaskSolver_MPPSRP_CPSAT
        from paths_config import interim_dir

        task_builder = TaskBuilder_MPPSRP_FullMILP_CPSAT(max_trips_per_day=1, verbose=False)
        task = task_builder.build_task(graph_data)

        print("Solving start time: {}".format(datetime.now()))
        task_solver = TaskSolver_MPPSRP_CPSAT(
            cache_dir=interim_dir,
            cache_all_feasible_solutions=False,
            solution_prefix='dataset',
            time_limit_milliseconds=30_000
        )
        start_time = time.time()
        solution = task_solver.solve(task)
        delta_time = time.time() - start_time

        print('kpi')


        # Проверяем, является ли решение допустимым
        if solution is None:  # Предположим, что у объекта solution есть метод is_infeasible()
            # Если решение не допустимо, создаем DataFrame с 'infeasible' значениями
            
            cpsat_kpi = {metric: 'infeasible' for metric in model_kpi.keys()}
        else:
            print('routes_schedule')
            cpsat_kpi = solution.get_kpis()
            pprint(cpsat_kpi)
            routes_schedule = solution.get_routes_schedule()
            pprint(routes_schedule)

        print("done")

        # Обработка KPI
        df_model_kpi = pd.DataFrame(list(model_kpi.items()), columns=['Metric', 'Model KPI'])
        df_cpsat_kpi = pd.DataFrame(list(cpsat_kpi.items()), columns=['Metric', 'CPSAT KPI'])
        df_kpi = pd.merge(df_model_kpi, df_cpsat_kpi, on='Metric', how='outer')

        # Сохранение KPI в CSV
        # output_dir = "your_output_directory"  # Замените на ваш выходной каталог
        df_kpi.to_csv(os.path.join(output_dir, f'kpi_summary_n{n}_veh{veh}.csv'), index=False)
