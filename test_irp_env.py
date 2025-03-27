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
    print(f"Revisit Penalty 1: {env.revisit_1}")
    print(f"Revisit Penalty 2: {env.revisit_2}")
    print(f"Revisit Penalty 3: {env.revisit_3}")
    print(f"Revisit Penalty 4: {env.revisit_4}")
    print(env.dist)
    # try:
    #     assert env.revisit == expected_revisit, f"Ожидался revisit = {expected_revisit}, получено {env.revisit}"
    #     print("Тест пройден успешно!")
    # except AssertionError as e:
    #     print(f"Тест провален: {e}")

# Тест 1: Два действия на станции без завершения дня
print("============ Тест 1: таже станция ===============")
env.reset()
action1 = np.array([0, 1, 10, 10, 0])  # Машина 0, станция 1, доставка [10, 10], end_day=0
obs, reward, done, truncated, info = env.step(action1)
print_step_info("Шаг 1", action1, env.revisit, 0)  # Первый шаг, штрафа нет

action2 = np.array([0, 1, 10, 10, 0])  # То же самое
obs, reward, done, truncated, info = env.step(action2)
print_step_info("Шаг 2", action2, env.revisit, -5)  # Ожидаем штраф -5

print("============ Тест 2: повтор в депо без конца дня ===============")
env.reset()
action1 = np.array([0, 0, 10, 10, 0])  # Машина 0, станция 1, доставка [10, 10], end_day=0
obs, reward, done, truncated, info = env.step(action1)
print_step_info("Шаг 1", action1, env.revisit, 0)  # Первый шаг, штрафа нет

action2 = np.array([0, 0, 10, 10, 0])  # То же самое
obs, reward, done, truncated, info = env.step(action2)
print_step_info("Шаг 2", action2, env.revisit, -5)  # Ожидаем штраф -5



print("============ Тест 4: повтор на станции конец дня ===============")
env.reset()
action1 = np.array([0, 1, 10, 10, 0])  # Машина 0, станция 1, доставка [10, 10], end_day=0
obs, reward, done, truncated, info = env.step(action1)
print_step_info("Шаг 1", action1, env.revisit, 0)  # Первый шаг, штрафа нет

action2 = np.array([0, 1, 10, 10, 1])  # То же самое
obs, reward, done, truncated, info = env.step(action2)
print_step_info("Шаг 2", action2, env.revisit, -5)  # Ожидаем штраф -5


print("============ Тест 5: повтор в депо и конец дня ===============")
env.reset()
action1 = np.array([0, 0, 10, 10, 0])  # Машина 0, станция 1, доставка [10, 10], end_day=0
obs, reward, done, truncated, info = env.step(action1)
print_step_info("Шаг 1", action1, env.revisit, 0)  # Первый шаг, штрафа нет

action2 = np.array([0, 0, 10, 10, 1])  # То же самое
obs, reward, done, truncated, info = env.step(action2)
print_step_info("Шаг 2", action2, env.revisit, -5)  # Ожидаем штраф -5



print("============ Тест 6: из станции в депо и конец дня ===============")
env.reset()
action1 = np.array([0, 1, 10, 10, 1])  # Машина 0, станция 1, доставка [10, 10], end_day=0
obs, reward, done, truncated, info = env.step(action1)
print_step_info("Шаг 1", action1, env.revisit, 0)  # Первый шаг, штрафа нет

action2 = np.array([0, 0, 10, 10, 0])  # То же самое
obs, reward, done, truncated, info = env.step(action2)
print_step_info("Шаг 2", action2, env.revisit, -5)  # Ожидаем штраф -5


print("============ Тест 7: из депо в депо и конец дня ===============")
env.reset()
action1 = np.array([0, 0, 10, 10, 1])  # Машина 0, станция 1, доставка [10, 10], end_day=0
obs, reward, done, truncated, info = env.step(action1)
print_step_info("Шаг 1", action1, env.revisit, 0)  # Первый шаг, штрафа нет

action2 = np.array([0, 0, 10, 10, 0])  # То же самое
obs, reward, done, truncated, info = env.step(action2)
print_step_info("Шаг 2", action2, env.revisit, -5)  # Ожидаем штраф -5


print("============ Тест 8: туда сюда ===============")
env.reset()
action1 = np.array([0, 1, 10, 10, 1])  # Машина 0, станция 1, доставка [10, 10], end_day=0
obs, reward, done, truncated, info = env.step(action1)
print_step_info("Шаг 1", action1, env.revisit, 0)  # Первый шаг, штрафа нет

action2 = np.array([0, 1, 10, 10, 1])  # То же самое
obs, reward, done, truncated, info = env.step(action2)
print_step_info("Шаг 2", action2, env.revisit, -5)  # Ожидаем штраф -5



print("============ Тест 9: пропуск дня ===============")
env.reset()
action1 = np.array([0, 0, 10, 10, 1])  # Машина 0, станция 1, доставка [10, 10], end_day=0
obs, reward, done, truncated, info = env.step(action1)
print_step_info("Шаг 1", action1, env.revisit, 0)  # Первый шаг, штрафа нет

action2 = np.array([0, 0, 10, 10, 1])  # То же самое
obs, reward, done, truncated, info = env.step(action2)
print_step_info("Шаг 2", action2, env.revisit, -5)  # Ожидаем штраф -5


print("============ Тест 10: Сбор машин ===============")
env.reset()
action1 = np.array([0, 1, 10, 10, 0])  # Машина 0, станция 1, доставка [10, 10], end_day=0
obs, reward, done, truncated, info = env.step(action1)
print_step_info("Шаг 1", action1, env.revisit, 0)  # Первый шаг, штрафа нет

action2 = np.array([1, 2, 10, 10, 1])  # То же самое
obs, reward, done, truncated, info = env.step(action2)
print_step_info("Шаг 2", action2, env.revisit, -5)  # Ожидаем штраф -5


