import pickle
import numpy as np
from PPO.environment.irp_env_custom import IRPEnv_Custom
from PPO.settings import PARAMETERS_DICT
import torch

# Настройка параметров среды
PARAMETERS_DICT['num_nodes'] = 5
PARAMETERS_DICT['k_vehicles'] = 2
PARAMETERS_DICT['products_count'] = 2  # Укажем явно, так как delivery имеет размерность 2

# Загрузка данных
pkl_path = r'C:\Users\rkozl\Documents\PythonProjects\PPO_PSRP\data_pkl\vrp_data.pkl'#f"data_pkl/nodes-{3}_steps-{500}_veh-{2}.pkl"
with open(pkl_path, "rb") as f:
    model_for_nn = pickle.load(f)

# Создание среды
env = IRPEnv_Custom(model_for_nn, PARAMETERS_DICT)
print(env.weight_matrixes)

# Функция для вывода результатов шага
def print_step_info(step_name, action, revisit_penalty, expected_revisit):
    print(f"\n{step_name}:")
    print(f"Action: {action}")
    print(f"Revisit Penalty 1: {env.revisit_1}")
    print(f"Revisit Penalty 2: {env.revisit_2}")
    print(f"Revisit Penalty 3: {env.revisit_3}")
    print(f"Revisit Penalty 4: {env.revisit_4}")
    print(f"Overfill: {env.overfill_penalty}") 
    print(f"Dist: {env.dist}")
    # print(f"Dist: {env.edge_features}")
    # print(env.edge_indices)

    # print(env.get_state())
    print()

print("============ Тест 1: таже станция ===============")
env.reset()
action1 = np.array([0, 1, 2, 2, 0])  # Машина 0, станция 1, доставка [10, 10], end_day=0
obs, reward, done, truncated, info = env.step(action1)
print_step_info("Шаг 0", action1, env.revisit, 0)  # Первый шаг, штрафа нет

action1 = np.array([0, 2, 2, 2, 0])  # Машина 0, станция 1, доставка [10, 10], end_day=0
obs, reward, done, truncated, info = env.step(action1)
print_step_info("Шаг 1", action1, env.revisit, 0)  # Первый шаг, штрафа нет

action1 = np.array([0, 3, 2, 2, 0])  # Машина 0, станция 1, доставка [10, 10], end_day=0
obs, reward, done, truncated, info = env.step(action1)
print_step_info("Шаг 2", action1, env.revisit, 0)  # Первый шаг, штрафа нет

action1 = np.array([0, 4, 2, 2, 0])  # Машина 0, станция 1, доставка [10, 10], end_day=0
obs, reward, done, truncated, info = env.step(action1)
print_step_info("Шаг 3", action1, env.revisit, 0)  # Первый шаг, штрафа нет


action1 = np.array([0, 0, 2, 2, 0])  # Машина 0, станция 1, доставка [10, 10], end_day=0
obs, reward, done, truncated, info = env.step(action1)
print_step_info("Шаг 3", action1, env.revisit, 0)  # Первый шаг, штрафа нет
