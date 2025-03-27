import pickle
import numpy as np
from PPO.environment.irp_env_custom import IRPEnv_Custom
from PPO.settings import PARAMETERS_DICT

# Настройка параметров среды
PARAMETERS_DICT['num_nodes'] = 3
PARAMETERS_DICT['k_vehicles'] = 2
PARAMETERS_DICT['products_count'] = 2  # Укажем явно, так как delivery имеет размерность 2

# Загрузка данных
pkl_path = f"data_pkl/nodes-{3}_steps-{500}_veh-{2}.pkl"
with open(pkl_path, "rb") as f:
    model_for_nn = pickle.load(f)

# Создание среды
env = IRPEnv_Custom(model_for_nn, PARAMETERS_DICT)

# Функция для вывода результатов шага
def print_step_info(step_name, action, revisit_penalty, expected_revisit):
    print(f"\n{step_name}:")
    print(f"Action: {action}")
    print(f"Revisit Penalty: {env.revisit}")
    try:
        assert env.revisit == expected_revisit, f"Ожидался revisit = {expected_revisit}, получено {env.revisit}"
        print("Тест пройден успешно!")
    except AssertionError as e:
        print(f"Тест провален: {e}")

# Тест 1: Два действия на станции без завершения дня
print("=== Тест 1: Два действия на станции без завершения дня ===")
env.reset()
action1 = np.array([0, 1, 10, 10, 0])  # Машина 0, станция 1, доставка [10, 10], end_day=0
obs, reward, done, truncated, info = env.step(action1)
print_step_info("Шаг 1", action1, env.revisit, 0)  # Первый шаг, штрафа нет

action2 = np.array([0, 1, 10, 10, 0])  # То же самое
obs, reward, done, truncated, info = env.step(action2)
print_step_info("Шаг 2", action2, env.revisit, -5)  # Ожидаем штраф -5

# Тест 2: Действие без завершения дня, затем с завершением дня
print("\n=== Тест 2: Станция без завершения, затем с завершением дня ===")
env.reset()
action3 = np.array([0, 1, 10, 10, 0])  # Машина 0, станция 1, доставка [10, 10], end_day=0
obs, reward, done, truncated, info = env.step(action3)
print_step_info("Шаг 1", action3, env.revisit, 0)  # Первый шаг, штрафа нет

action4 = np.array([0, 1, 10, 10, 1])  # Завершаем день
obs, reward, done, truncated, info = env.step(action4)
print_step_info("Шаг 2", action4, env.revisit, -5)  # Ожидаем штраф -5

# Тест 3: Действие в депо без завершения дня
print("\n=== Тест 3: Депо без завершения дня ===")
env.reset()
action5 = np.array([0, 0, 10, 10, 0])  # Машина 0, депо (0), доставка [10, 10], end_day=0
obs, reward, done, truncated, info = env.step(action5)
print_step_info("Шаг 1", action5, env.revisit, -5)  # Ожидаем штраф -5

# Тест 4: Действие в депо с завершением дня
print("\n=== Тест 4: Депо с завершением дня ===")
env.reset()
action6 = np.array([0, 1, 10, 10, 1])  # Машина 0, депо (0), доставка [10, 10], end_day=1
obs, reward, done, truncated, info = env.step(action6)
print_step_info("Шаг 1", action6, env.revisit, 0)  # Ожидаем 0
action6 = np.array([0, 0, 10, 10, 0])  # Машина 0, депо (0), доставка [10, 10], end_day=1
obs, reward, done, truncated, info = env.step(action6)
print_step_info("Шаг 2", action6, env.revisit, 0)  # Ожидаем 0
action6 = np.array([0, 1, 10, 10, 0])  # Машина 0, депо (0), доставка [10, 10], end_day=1
obs, reward, done, truncated,  info = env.step(action6)
print_step_info("Шаг 3", action6, env.revisit, 0)  # Ожидаем 0