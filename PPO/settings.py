import torch

KEYS_TO_LOG = [
    'total_travel_distance',
    'total_travel_time',
    'average_stock_levels_percent',
    'dry_runs',
    'average_vehicle_utilization',
    'average_stops_per_trip',
    'average_delivery',
]

REWARDS_TO_LOG = [   
    'distance_rewards',
    'dry_runs_penalty',
    'time_end_penalties',
    'empty_load_penalties',
    # 'restricted_station_penalties',
    'revisit_1',
    'revisit_2',
    'revisit_3',
    'revisit_4',
    'overfill_penalty']

PARAMETERS_DICT = {
    'num_nodes': 5,
    'products_count': 2,
    'num_depots': 1,
    'k_vehicles': 5,
    'compartment_capacity': 50,
    'planning_horizon': 7,
    'noise_demand': 0.2,
    'seed': 6,
    'max_trips': 1,
    'eval_epochs': 3,
    'load_policy':  'full_fill',
    'days_to_fill': 3,
    'working_time': float(18 * 60 * 60 - 9 * 60 * 60),
}