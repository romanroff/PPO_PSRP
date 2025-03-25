import argparse
import pickle
import gymnasium
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
from pprint import pprint

from PPO.environment.irp_env_custom import IRPEnv_Custom
from PPO.logger import InfoLoggerCallback, RewardsCallback, TensorboardGradientCallback
from PPO.settings import KEYS_TO_LOG, REWARDS_TO_LOG, PARAMETERS_DICT
from PPO.GNN.GNN_Gat import GATFeatureExtractor
from stable_baselines3.common.utils import Schedule
import numpy as np

def parse_and_print_state(state):
    print("=== Разбор состояния ===")
    
    # # 1. normalized_remaining_time
    # if 'normalized_remaining_time' in state:
    #     nrt = state['normalized_remaining_time']
    #     print(f"Normalized Remaining Time: shape={nrt.shape}")
    #     print(f"Values: {nrt}\n")
    
    # 2. node_features
    if 'node_features' in state:
        nf = state['node_features']
        num_stations, features_per_station = nf.shape
        print(f"Node Features: shape={nf.shape}")
        print(f"Number of stations: {num_stations}")
        print(f"Features per station: {features_per_station}")
        
        # Попробуем разделить node_features на логические части
        products_count = None
        base_features = 5  # Количество базовых признаков на продукт (из кода)
        
        # Предполагаем, что первые признаки - это базовые характеристики
        for possible_products in range(1, 11):  # Ограничим до 10 продуктов
            base_size = base_features * possible_products
            if features_per_station >= base_size:
                remaining = features_per_station - base_size
                if remaining % (possible_products + 1) == 0:  # +1 для temp_delivery
                    products_count = possible_products
                    break
        
        if products_count:
            base_size = base_features * products_count
            vehicles_features = (features_per_station - base_size - products_count)
            k_vehicles = vehicles_features // products_count if vehicles_features > 0 else 0
            
            print(f"Detected products count: {products_count}")
            print(f"Detected k_vehicles: {k_vehicles}")
            
            # Разделяем node_features
            base_features_tensor = nf[:, :base_size]
            vehicle_loads_tensor = nf[:, base_size:base_size + vehicles_features] if k_vehicles > 0 else None
            delivery_tensor = nf[:, -products_count:]
            
            print("Base Features (max_capacities, min_capacities, demands, etc):")
            print(base_features_tensor)
            if vehicle_loads_tensor is not None:
                print("\nVehicle Loads (flattened):")
                print(vehicle_loads_tensor)
            print("\nDelivery Features:")
            print(delivery_tensor)
        else:
            print("Full node features (couldn't split automatically):")
            print(nf)
        print()

    # 5. global_features
    # if 'global_features' in state:
    #     gf = state['global_features']
    #     print(f"Global Features: shape={gf.shape}")
    #     print(f"Values: {gf}")
    #     print(f"Components: vehicles={gf[0]}, cur_day={gf[1]}, vehicles_done={gf[2]}\n")

    # # 6. vehicle_locations
    # if 'vehicle_locations' in state:
    #     vl = state['vehicle_locations']
    #     print(f"Vehicle Locations: shape={vl.shape}")
    #     print(f"Values: {vl}\n")

# Пример использования
state_example = {
    'edge_attr': np.array([[1.], [1.], [1.], [0.8487805], [1.], [0.8487805]], dtype=np.float32),
    'edge_index': np.array([[0, 0, 1, 1, 2, 2], [0, 2, 0, 2, 0, 1]]),
    'global_features': np.array([0.5, 0., 0.], dtype=np.float32),
    'node_features': np.array([
        [1., 1., 0.05263158, 0.05263158, 0.10526316, 0.10526316, 0.94736844, 0.94736844, 0., 0., 
         0.05263158, 0.05263158, 0.05263158, 0.05263158, 0., 0.],
        [1., 1., 0.05263158, 0.05263158, 0.10526316, 0.10526316, 0.47368422, 0.47368422, 0., 0., 
         0.05263158, 0.05263158, 0.05263158, 0.05263158, 0., 0.]
    ]),
    'normalized_remaining_time': np.array([0.56481481]),
    'vehicle_locations': np.array([1, 0])
}

PARAMETERS_DICT['num_nodes'] = 3
PARAMETERS_DICT['k_vehicles'] = 2

pkl_path = f"data_pkl/nodes-{3}_steps-{500}_veh-{2}.pkl"
with open(pkl_path, "rb") as f:
    model_for_nn = pickle.load(f)

import pickle

pkl_path = f"data_pkl/nodes-{3}_steps-{500}_veh-{2}.pkl"
with open(pkl_path, "rb") as f:
    model_for_nn = pickle.load(f)

positions, weight_matrixes, daily_demands, \
    depots, working_time, restriction_matrix, \
    service_times, min_capacities, max_capacities, \
    init_capacities, vehicle_compartments = model_for_nn


env = IRPEnv_Custom(model_for_nn, PARAMETERS_DICT)


state = env.get_state()
node_features = state['node_features']
num_stations = node_features.shape[0]  # Количество станций
current_storage = node_features[:, 6:8] # Извлекаем третью часть

print("Текущий запас:", current_storage)
print()
import numpy as np

test_action = np.array([
    0,  # Фиксированное транспортное средство
    0,  # Фиксированный узел
    *[10] * 2,  # Фиксированное количество товаров
    0   # Фиксированное бинарное значение
])


obs, reward, done, truncated, info = env.step(test_action)
state = env.get_state(1)
node_features = state['node_features']
num_stations = node_features.shape[0]  # Количество станций
current_storage = node_features[:, 6:8] # Извлекаем третью часть
print("DRY RUNS REWARD",env.dry_runs_penalty)
print("DIST REWARD", env.dist)

print("Запас после действия запас:", current_storage)

print('REWARD', reward)
print()

parse_and_print_state(state)


test_action = np.array([
    0,  # Фиксированное транспортное средство
    1,  # Фиксированный узел
    *[10] * 2,  # Фиксированное количество товаров
    0   # Фиксированное бинарное значение
])


obs, reward, done, truncated, info = env.step(test_action)
state = env.get_state(2)
node_features = state['node_features']
num_stations = node_features.shape[0]  # Количество станций
current_storage = node_features[:, 6:8] # Извлекаем третью часть
print("DRY RUNS REWARD",env.dry_runs_penalty)
print("DIST REWARD", env.dist)

print("Запас после действия запас:", current_storage)

print('REWARD', reward)
print()

parse_and_print_state(state)



state = env.get_state()
node_features = state['node_features']
num_stations = node_features.shape[0]  # Количество станций
current_storage = node_features[:, 6:8] # Извлекаем третью часть

print("Запас после действия запас:", current_storage)
print()


print('REWARD', reward)

parse_and_print_state(state)