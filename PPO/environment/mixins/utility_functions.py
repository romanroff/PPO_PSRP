import time
import torch

class IRPEnvUtilitiesMixin:
    def get_state(self, station_idx=0) -> dict:
        # Формируем node_features только из 2D тензоров
        node_features = torch.cat([
            torch.nan_to_num(self.max_capacities / self.max_capacities, 0, posinf=0),  # Спрос
            torch.nan_to_num(self.min_capacities / self.max_capacities, 0, posinf=0),  # Спрос
            torch.nan_to_num(self.demands / self.max_capacities, 0, posinf=0),  # Спрос
            torch.nan_to_num((self.init_capacities - self.min_capacities) / self.max_capacities, 0, posinf=0),  # Остатки
            (self.init_capacities < self.min_capacities).float(),
        ], dim=-1).float()  # Размер: [num_stations, 4 * products_count]

        # Формируем отдельный тензор для загрузки всех машин
        temp_load_all_vehicles = self.temp_load.expand(self.num_stations, self.k_vehicles, self.products_count)
        vehicle_loads = torch.nan_to_num(temp_load_all_vehicles / self.max_capacities.unsqueeze(1), 0, posinf=0)  # [num_stations, k_vehicles, products_count]

        temp_load_flattened = vehicle_loads.reshape(self.num_stations, self.k_vehicles*self.products_count)
        node_features = torch.cat([node_features, temp_load_flattened], dim=1)

        # Формируем отдельный тензор для информации о доставках
        # Обрабатываем доставку, если не депо
        temp_delivery = torch.zeros(self.num_stations, self.products_count)
        if station_idx != self.depots.item():
            station_idx_for_capacities = station_idx - 1
            temp_delivery[station_idx_for_capacities] = self.delivery / self.max_capacities[station_idx_for_capacities]

        node_features = torch.cat([node_features, temp_delivery], dim=1)

        # Формируем global_features с учетом всех машин
        time_for_vehicle = self.working_time  # Вектор времени для всех машин
        normalized_remaining_time = torch.nan_to_num(self.cur_remaining_time / time_for_vehicle, 0, posinf=0)
        global_features = torch.cat([
            self.vehicles.float().unsqueeze(0) / self.k_vehicles * self.max_trips,  # Общее количество доступных машин
            (self.cur_day / self.planning_horizon).float().unsqueeze(0),  # Текущий день
            (self.vehicles <= 0).float().unsqueeze(0)  # Флаг окончания машин
        ]).squeeze().float()

        state = {
            'normalized_remaining_time':normalized_remaining_time,
            'node_features': node_features,  # Только 2D данные о станциях
            'edge_index': self.edge_indices,
            'edge_attr': self.edge_features,
            'global_features': global_features,
            'vehicle_locations': self.vehicle_locations  # Местоположение всех машин
        }
        state_np = self.tensors_to_numpy(state)
        return state_np

    def tensors_to_numpy(self, tensor_dict):
        return {key: value.cpu().numpy() for key, value in tensor_dict.items()}

    def get_kpis(self):
        algorithm_run_time = time.time() - self.algorithm_start_time
    
        kpis = {
            'total_travel_distance': int(self.total_travel_distance.mean().item() / 60),
            'total_travel_time': int(self.total_travel_distance.mean().item()),
            'average_stock_levels': self.total_stock_level / self.products_count,
            'average_stock_levels_percent': self.average_stock_levels_percent.mean().item() / self.planning_horizon * 100,
            'dry_runs': int(self.total_dry_runs.mean().item()),
            'algorithm_run_time': round(algorithm_run_time, 3),
            'average_vehicle_utilization': round(
                torch.nan_to_num(self.total_delivered_quantity / self.total_vehicle_capacities).mean().item() * 100, 1),
            'average_stops_per_trip': (self.total_stops).mean().item(),

            'capacity_rewards': self.capacity_reward,
            'distance_rewards': self.dist,
            'time_end_penalties': self.time_end,
            'empty_load_penalties':  self.empty_load,
            'dry_runs_penalties':self.dry_runs_penalty,
            'closeness':self.closeness,
            'restricted_station_penalties': self.restricted_station,
            'revisit_penalties':  self.revisit
        }
        average_routes = self.average_routes(self.actions_list)
        kpis['average_stops_per_trip'] /= average_routes + 1e-6
        kpis['average_stops_per_trip'] = round(kpis['average_stops_per_trip'], 1)
        kpis['average_stock_levels'] = kpis['average_stock_levels'].cpu().tolist()
        kpis['average_delivery'] = self.total_delivered_quantity.mean().item() / (self.step_count +1)

        return kpis

    def get_distance(self, node_idx_1, node_idx_2) -> float:
        # Приводим к int, если это тензор, иначе используем как есть
        idx_1 = node_idx_1.item() if isinstance(node_idx_1, torch.Tensor) else int(node_idx_1)
        idx_2 = node_idx_2.item() if isinstance(node_idx_2, torch.Tensor) else int(node_idx_2)
        return self.weight_matrixes[idx_1, idx_2]