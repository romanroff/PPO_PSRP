import argparse
import pickle

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from PIL import Image
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.utils import obs_as_tensor

from PPO.settings import PARAMETERS_DICT
from PPO.environment.irp_env_custom import IRPEnv_Custom

import numpy as np
import torch as th

# Функция для обработки аргументов командной строки
def parse_args():
    parser = argparse.ArgumentParser(description="Параметры для запуска модели.")
    parser.add_argument("--n", type=int, required=True, help="Количество узлов.")
    parser.add_argument("--n_steps", type=int, required=True, help="Количество шагов.")
    parser.add_argument("--veh", type=int, required=True, help="Количество транспортных средств.")
    return parser.parse_args()

# Получение аргументов
args = parse_args()

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

PARAMETERS_DICT['num_nodes'] = args.n
PARAMETERS_DICT['k_vehicles'] = args.veh

# Загрузка model_for_nn из файла .pkl
pkl_path = f"data_pkl/nodes-{args.n}_steps-{args.n_steps}_veh-{args.veh}.pkl"
try:
    with open(pkl_path, "rb") as f:
        model_for_nn = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Файл {pkl_path} не найден. Проверьте путь и наличие файла.")
env = IRPEnv_Custom(model_for_nn, PARAMETERS_DICT)

model = RecurrentPPO.load(rf"models/nsteps-{args.n_steps}_nodes-{args.n}_veh-{args.veh}/best_model.zip", env=env)

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
    if visited_actions == set(range(args.n)) and not set_completed:
        set_completed = True  # Устанавливаем флаг, что все значения посещены

    # После заполнения множества ожидаем один дополнительный шаг и фиксируем total_distance
    elif set_completed and not total_distance_fixed:
        total_distance = info[0]['total_travel_distance']
        print(f'Все значения от 0 до {args.n} посещены + 1 шаг. total_distance = {total_distance}')
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
    capacities_list.append(eval_env.envs[0].init_capacities[:, 0].tolist())

    episode_starts = done
    # img = eval_env.envs[0].render()  # Ваше основное изображение
    # combined_img = add_bar_chart_to_image(img, probs[0])
    # image_arrays.append(combined_img)

    # actions.append(int(action))



    # prob = predict_proba(model, obs)

    steps += 1
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
        pd.DataFrame(capacities_list, columns=[f'{i}' for i in range(args.n)]).plot(ax=last_ax)
        last_ax.set_title('Capacities Plot')
        last_ax.grid()
        add_vertical_lines(last_ax, df['new_day'])

        # Общий заголовок и отображение графиков
        plt.tight_layout()
        plt.savefig(f"results/rewards_{args.n}_{args.n_steps}_{args.veh}.png")