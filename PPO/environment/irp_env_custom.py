import torch
import numpy as np

from typing import Tuple
from gymnasium import spaces, Env
from PIL import Image, ImageDraw, ImageFont

from .mixins.action_management import ActionManagement
from .mixins.kpi_tracking import KPITracking
from .mixins.render_utils import RenderUtils
from .mixins.state_utils import StateUtils
from .mixins.utility_functions import IRPEnvUtilitiesMixin

class IRPEnv_Custom(Env, ActionManagement, KPITracking, RenderUtils, StateUtils, IRPEnvUtilitiesMixin):
    def __init__(self, batch, parameters_dict, seed: int = 69):
        super(IRPEnv_Custom, self).__init__()
        self.render_utils = RenderUtils()

        self._initialize_parameters(parameters_dict)
        self._set_seed(seed)
        self.batch = batch
        self.reset(seed)

        self.action_space = spaces.MultiDiscrete([
            self.k_vehicles,
            self.num_nodes,
            *[6] * self.products_count,
            2
        ])

        self.observation_space = spaces.Dict({
            'normalized_remaining_time': spaces.Box(low=0, high=1, shape=(self.k_vehicles,), dtype=np.float32),
            'node_features': spaces.Box(
                low=0,
                high=1,
                shape=(self.num_nodes, 7 * self.products_count + self.k_vehicles * self.products_count + 2 * self.k_vehicles + 2 + self.products_count * 3 * 2 + 2),
                dtype=np.float32
            ),
            'edge_index': spaces.Box(
                low=0,
                high=self.num_stations,
                shape=(2, self.edge_indices.shape[1]),
                dtype=np.int64
            ),
            'edge_attr': spaces.Box(
                low=0,
                high=float('inf'),
                shape=(self.edge_indices.shape[1], self.edge_features.shape[1]),
                dtype=np.float32
            ),
            'global_features': spaces.Box(low=0, high=float('inf'), shape=(3,), dtype=np.float32),
        })


    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, dict, bool]:
        self.step_count += 1
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions).to(self.device)

        self.vehicle = actions[0]
        self.station_idx = actions[1]
        self.delivery_percents = actions[2:-1].float() * 20.0 / 100
        self.end_day_flag = actions[-1] == 1

        self.action_history.append((self.vehicle, self.station_idx, self.delivery_percents, self.end_day_flag))

        self.render_steps.append({
            'type': 'move',
            'vehicle': self.vehicle.item(),
            'start': self.vehicle_updated_locations[self.vehicle].item(),
            'end': self.station_idx.item(),
            'delivery_percents': self.delivery_percents,
            'end_day_flag': 0
        })

        self._update_time_and_location()
        self._update_load()
        self._handle_depot_visits()
        self._handle_day_end()

        if self.day_end and self.station_idx != self.depots.item():
            self.render_steps.append({
                'type': 'return_to_depot',
                'vehicle': self.vehicle.item(),
                'start': self.station_idx.item(),
                'end': self.depots.item(),
                'delivery_percents': torch.zeros_like(self.delivery_percents),
                'end_day_flag': 1
            })

        done = self.is_done()
        self.calc_step_kpis()
        total_reward = self.get_reward()

        return self.get_state(), total_reward, done, done, self.get_kpis()

    def reset(self, seed=None, options=None, **kwargs):
        if seed is not None:
            self._set_seed(seed)  # Устанавливаем seed, если передан
        self._unpack_batch()
        self._init_tracking_variables()
        self._initialize_tensors()
        self._initialize_variables()
        self._calculate_initial_state()
        return self.get_state(), self.get_kpis()

    def render(self, mode='rgb_array', **kwargs):
        probs = kwargs.get('probs', None)
        return self.render_utils.render(self, probs=probs, mode=mode)

    def action_masks(self):
        # Маска для транспортных средств: True, если осталось время
        vehicles_mask = (self.cur_remaining_time > 0).cpu().numpy().astype(bool)
        
        # Базовая маска для станций (только станции, без депо)
        active_vehicles = np.where(vehicles_mask)[0]
        restriction_matrix = self.restriction_matrix.cpu().numpy()
        if len(active_vehicles) > 0:
            restriction_matrix_active = restriction_matrix[active_vehicles]
            stations_allowed = np.any(restriction_matrix_active == 0, axis=0).astype(bool)
        else:
            stations_allowed = np.zeros(self.num_nodes, dtype=bool)
        
        full_stations = torch.all(self.init_capacities >= self.max_capacities, dim=1).cpu().numpy()
        
        depot_mask = np.array([True])  # Для депо
        full_stations = np.concatenate([depot_mask, full_stations])
        stations_mask = stations_allowed & ~full_stations  
        
        delivery_percent_masks = [np.array([True] * 5) for _ in range(self.products_count)]
        
        end_day_mask = np.array([True, True])
        
        # Объединяем все маски в одну сплющенную маску для MultiDiscrete
        flattened_mask = np.concatenate([
            vehicles_mask,
            stations_mask,
            *[mask for mask in delivery_percent_masks],
            end_day_mask
        ])
        
        return flattened_mask