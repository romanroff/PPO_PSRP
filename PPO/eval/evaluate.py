import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .kpi_logger import log_kpi_metrics
from .utils import predict_proba, add_bar_chart_to_image, predict_proba_trpo

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
                                    'empty_load_penalties', 'restricted_station_penalties', ]}

    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        probs = predict_proba(model, obs, lstm_states=lstm_states, episode_start=episode_starts)
        # Убедимся, что probs — это одномерный массив
        probs = probs.squeeze()  # Удаляем лишние размерности
        obs, rewards, done, info = eval_env.step(action)

        # Получаем список кадров от render
        frames = eval_env.envs[0].render(probs=probs)
        image_arrays.append(frames)  # Добавляем список кадров для текущего шага
        if hasattr(eval_env.envs[0], 'day_end'):
            print(action, eval_env.envs[0].day_end, rewards)

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


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from .kpi_logger import log_kpi_metrics
from .utils import predict_proba, add_bar_chart_to_image
import time  # Добавляем импорт модуля time

def evaluate_model_trpo(model, args):
    """
    Оценка модели TRPO
    Args:
        model: Обученная модель TRPO
        args: Аргументы конфигурации
    """
    eval_env = model.get_env()
    obs = eval_env.reset()
    done = False

    num_envs = 1
    episode_starts = np.ones((num_envs,), dtype=bool)
    visited_actions = set()
    set_completed = False
    total_distance_fixed = False

    steps = 0
    capacities_list, new_day, image_arrays = [], [], []
    kpi_data = {key: [] for key in ['capacity_rewards', 'distance_rewards', 'time_end_penalties', 
                                    'empty_load_penalties', 'restricted_station_penalties']}

    # Засекаем время начала цикла
    start_time = time.time()

    while not done:
        # Предсказание действия с помощью TRPO
        action, _ = model.predict(obs, deterministic=True)
        probs = predict_proba_trpo(model, obs, episode_start=episode_starts)
        probs = probs.squeeze()  # Удаляем лишние размерности

        # Шаг в среде
        obs, rewards, done, info = eval_env.step(action)

        # Получаем список кадров от render
        """Вспомогательный метод для создания одного кадра."""
        # Создаем пустое изображение с размерами, соответствующими figsize
        width, height = [int(x * 100) for x in [100,100]]  # Предполагаем, что 1 единица figsize = 100 пикселей
        img = np.zeros((height, width, 3), dtype=np.uint8)  # Черное изображение (RGB)
        from PIL import Image
        frames = [Image.fromarray(img)]
        # frames = eval_env.envs[0].render(probs=probs)
        image_arrays.append(frames)  # Добавляем список кадров для текущего шага
        
        if hasattr(eval_env.envs[0], 'day_end'):
            print(action, eval_env.envs[0].day_end, rewards)

        # Обработка действия
        current_action = action.item() if np.isscalar(action) else action[0][1].item()
        visited_actions.add(current_action)

        # Логика завершения набора действий
        if visited_actions == set(range(args.n)) and not set_completed:
            set_completed = True
        elif set_completed and not total_distance_fixed:
            total_distance_fixed = True

        # Сбор KPI данных
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

    # Засекаем время окончания и вычисляем разницу
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Время выполнения цикла while not done: {execution_time:.2f} секунд")

    # Создаем директорию для сохранения с префиксом trpo
    save_dir = f"results/trpo_{args.n}_{args.n_steps}_{args.veh}/steps"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Сохраняем все кадры
    for step_idx, frame_list in enumerate(image_arrays):
        for frame_idx, img in enumerate(frame_list):
            img.save(f'{save_dir}/env_frame_{step_idx}_{frame_idx}.png')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .kpi_logger import log_kpi_metrics
from .utils import predict_proba, add_bar_chart_to_image, predict_proba_trpo

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
                                    'empty_load_penalties', 'restricted_station_penalties', ]}

    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        probs = predict_proba(model, obs, lstm_states=lstm_states, episode_start=episode_starts)
        # Убедимся, что probs — это одномерный массив
        probs = probs.squeeze()  # Удаляем лишние размерности
        obs, rewards, done, info = eval_env.step(action)

        # Получаем список кадров от render
        frames = eval_env.envs[0].render(probs=probs)
        image_arrays.append(frames)  # Добавляем список кадров для текущего шага
        if hasattr(eval_env.envs[0], 'day_end'):
            print(action, eval_env.envs[0].day_end, rewards )

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


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from .kpi_logger import log_kpi_metrics
from .utils import predict_proba, add_bar_chart_to_image

# def evaluate_model_trpo(model, args):   
#     """
#     Evaluates a TRPO model using the same environment configuration as in training.
    
#     Args:
#         model: Trained TRPO model
#         args: Arguments containing parameters like n, veh, n_steps
    
#     Returns:
#         float: Mean reward over evaluation episodes
#     """
#     import pickle
#     import gymnasium
#     from stable_baselines3.common.monitor import Monitor
#     from PPO.environment.irp_env_custom import IRPEnv_Custom
#     from PPO.settings import PARAMETERS_DICT
#     from stable_baselines3.common.vec_env import DummyVecEnv
    
#     # Load the same data as in training
#     pkl_path = f"data_pkl/nodes-{args.n}_steps-{500}_veh-{args.veh}.pkl"
#     try:
#         with open(pkl_path, "rb") as f:
#             model_for_nn = pickle.load(f)
#     except FileNotFoundError:
#         raise FileNotFoundError(f"File {pkl_path} not found. Check path and file existence.")
    
#     # Create evaluation environment exactly like in training
#     eval_env = model.get_env()
    
#     # If VecNormalize was used in training, uncomment these lines and provide the path
#     # from stable_baselines3.common.vec_env import VecNormalize
#     # stats_path = f"models/trpo_GNN_nsteps-{args.n_steps}_nodes-{args.n}_veh-{args.veh}/vec_normalize_stats.pkl"
#     # eval_env = VecNormalize.load(stats_path, eval_env)
#     # eval_env.training = False  # Don't update running mean/std during evaluation
#     # eval_env.norm_reward = False  # Don't normalize rewards during evaluation
    
#     # Evaluation loop
#     n_eval_episodes = 1  # Same as in EvalCallback
#     episode_rewards = []
    
#     for i in range(n_eval_episodes):
#         obs = eval_env.reset()
#         done = False
#         episode_reward = 0
        
#         while not done:
#             # Use deterministic=True as in EvalCallback
#             action, _ = model.predict(obs, deterministic=True)
            
#             # Step the environment
#             obs, rewards, dones, infos = eval_env.step(action)
#             print(action, rewards) 
            
#             # Update rewards
#             episode_reward += rewards[0]
            
#             # Check if episode is done
#             done = dones[0]
            
#             # Handle episode end if needed
#             if done:
#                 episode_rewards.append(episode_reward)
#                 print(f"Episode {i+1}/{n_eval_episodes}: Reward = {episode_reward}")
#                 break
    
#     # Calculate mean reward
#     mean_reward = sum(episode_rewards) / len(episode_rewards)
#     print(f"Mean evaluation reward: {mean_reward}")
    
#     return mean_reward  