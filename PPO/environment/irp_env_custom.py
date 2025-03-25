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
            *[5] * self.products_count,
            2
        ])

        self.observation_space = spaces.Dict({
            'normalized_remaining_time': spaces.Box(low=0, high=1, shape=(self.k_vehicles,), dtype=np.float32),
            'node_features': spaces.Box(low=0, high=1, shape=(self.num_stations, 5 * self.products_count + self.products_count * self.k_vehicles + self.products_count ), dtype=np.float32),
            'edge_index': spaces.Box(low=0, high=self.num_nodes - 1, shape=(2, self.edge_indices.shape[1]), dtype=np.int64),
            'edge_attr': spaces.Box(low=0, high=float('inf'), shape=(self.edge_indices.shape[1], self.edge_features.shape[1]), dtype=np.float32),
            'global_features': spaces.Box(low=0, high=float('inf'), shape=(3,), dtype=np.float32),
            'vehicle_locations': spaces.Box(low=0, high=self.num_nodes - 1, shape=(self.k_vehicles,), dtype=np.int64),
        })

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, dict, bool]:
        self.step_count += 1
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions).to(self.device)

        vehicle = actions[0].item()
        station_idx = actions[1].item()
        delivery_percents = actions[2:-1].float() * 25.0
        end_day_flag = actions[-1].item()

        current_vehicle_location = self.vehicle_locations[vehicle]
        self.action_history.append((vehicle, station_idx, delivery_percents, end_day_flag))
        current_vehicle_location = current_vehicle_location.unsqueeze(0)
        traversed_edges = torch.cat([current_vehicle_location, torch.tensor([station_idx], device=self.device)], dim=0).long()

        self.render_steps.append({
            'type': 'move',
            'vehicle': vehicle,
            'start': self.vehicle_locations[vehicle].item(),
            'end': station_idx,
            'delivery_percents': delivery_percents,
            'end_day_flag': 0
        })

        self._update_time_and_location(torch.tensor([station_idx]), vehicle)
        self.delivery = torch.zeros(self.products_count, device=self.device)

        # Обрабатываем доставку, если не депо
        if station_idx != self.depots.item():
            station_idx_for_capacities = station_idx - 1
            for product_idx, delivery_percent in enumerate(delivery_percents):
                self._update_load(torch.tensor([station_idx_for_capacities]), delivery_percent, product_idx, vehicle)

        self._handle_depot_visits(torch.tensor([station_idx]), vehicle)

        # Обрабатываем конец дня
        self._handle_day_end(end_day_flag=end_day_flag)

        if end_day_flag == 1 and station_idx != self.depots.item():
            self.render_steps.append({
                'type': 'return_to_depot',
                'vehicle': vehicle,
                'start': station_idx,
                'end': self.depots.item(),
                'delivery_percents': torch.zeros_like(delivery_percents),
                'end_day_flag': 1
            })

        done = self.is_done()
        self.calc_step_kpis(torch.tensor([station_idx]), traversed_edges, vehicle)
        total_reward = self.get_reward(traversed_edges)

        return self.get_state(station_idx), total_reward, done, done, self.get_kpis()

    def reset(self, seed=None, options=None, **kwargs):
        if seed is not None:
            self._set_seed(seed)  # Устанавливаем seed, если передан
        self._unpack_batch()
        self._init_tracking_variables()
        self._initialize_tensors()
        self._initialize_variables()
        self._calculate_initial_state()
        self.dry_days = torch.zeros(self.num_stations, dtype=torch.float32, device=self.device)  # [num_stations]
        return self.get_state(), self.get_kpis()

    def render(self, mode='rgb_array', **kwargs):
        probs = kwargs.get('probs', None)
        return self.render_utils.render(self, probs=probs, mode=mode)