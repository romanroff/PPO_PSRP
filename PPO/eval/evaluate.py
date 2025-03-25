import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .kpi_logger import log_kpi_metrics
from .utils import predict_proba, add_bar_chart_to_image

def evaluate_model(model, args):
    eval_env = model.get_env()
    obs = eval_env.reset()
    done = False
    lstm_states = None
    num_envs = 1
    episode_starts = np.ones((num_envs,), dtype=bool)
    visited_actions = set()
    set_completed = False
    total_distance_fixed = False

    steps = 0
    capacities_list, new_day, image_arrays = [], [], []
    kpi_data = {key: [] for key in ['capacity_rewards', 'distance_rewards', 'time_end_penalties', 
                                    'empty_load_penalties', 'restricted_station_penalties', 
                                    'revisit_penalties']}

    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        probs = predict_proba(model, obs, lstm_states=lstm_states, episode_start=episode_starts)
        # Убедимся, что probs — это одномерный массив
        probs = probs.squeeze()  # Удаляем лишние размерности
        obs, rewards, done, info = eval_env.step(action)

        # Получаем список кадров от render
        frames = eval_env.envs[0].render(probs=probs)
        image_arrays.append(frames)  # Добавляем список кадров для текущего шага
        print(action)

        current_action = action[0][1].item()
        visited_actions.add(current_action)

        if visited_actions == set(range(args.n)) and not set_completed:
            set_completed = True
        elif set_completed and not total_distance_fixed:
            total_distance_fixed = True

        for key in kpi_data.keys():
            kpi_data[key].append(info[0][key])

        day_end = eval_env.envs[0].day_end
        new_day.append(day_end)
        capacities_list.append(eval_env.envs[0].init_capacities[:, 0].tolist())

        episode_starts = done
        steps += 1

        if done:
            total_rewards = np.sum([np.array(kpi_data[key]) for key in kpi_data.keys()], axis=0)
            kpi_data.update({'total_reward': total_rewards, 'new_day': new_day})
            df = pd.DataFrame(kpi_data)
            log_kpi_metrics(df, capacities_list, args)

    # Создаем директорию для сохранения, если ее нет
    save_dir = f"results/{args.n}_{args.n_steps}_{args.veh}/steps"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Сохраняем все кадры с учетом шага и индекса кадра
    for step_idx, frame_list in enumerate(image_arrays):
        for frame_idx, img in enumerate(frame_list):
            img.save(f'{save_dir}/env_frame_{step_idx}_{frame_idx}.png')