import time
import torch

class IRPEnvUtilitiesMixin:
    def get_state(self) -> dict:
        # Существующий код для node_features
        node_features = torch.cat([
            self.max_capacities / self.max_capacities,
            self.min_capacities / self.max_capacities,
            self.demands / self.max_capacities,
            self.init_capacities / self.max_capacities, 
            (self.init_capacities - self.min_capacities) / self.max_capacities, 
            (self.init_capacities < self.min_capacities).float(),
            (self.init_capacities >= self.max_capacities).float(),
        ], dim=-1).float()

        temp_load_all_vehicles = self.temp_load.expand(self.num_stations, self.k_vehicles, self.products_count)
        vehicle_loads = torch.nan_to_num(temp_load_all_vehicles / self.max_capacities.unsqueeze(1), 0, posinf=0)
        temp_load_flattened = vehicle_loads.reshape(self.num_stations, self.k_vehicles*self.products_count)
        node_features = torch.cat([node_features, temp_load_flattened], dim=1)

        temp_delivery = torch.zeros(self.num_stations, self.products_count)
        temp_delivery_percent = torch.zeros(self.num_stations, self.products_count)
        if self.station_idx != self.depots.item():
            station_idx_for_capacities = self.station_idx - 1
            temp_delivery[station_idx_for_capacities] = self.delivery / self.max_capacities[station_idx_for_capacities]
            temp_delivery_percent[station_idx_for_capacities] = self.delivery_percents
        node_features = torch.cat([node_features, temp_delivery], dim=1)
        node_features = torch.cat([node_features, temp_delivery_percent], dim=1)

        depot_features = torch.zeros(1, node_features.shape[1], device=node_features.device)
        node_features = torch.cat([depot_features, node_features], dim=0)

        vehicle_indicators = torch.zeros(self.num_nodes, self.k_vehicles, device=node_features.device)
        for k, loc in enumerate(self.vehicle_updated_locations):
            vehicle_indicators[int(loc.item()), k] = 1
        node_features = torch.cat([node_features, vehicle_indicators], dim=-1)

        # Временные признаки
        future_stock_levels = torch.zeros(self.num_nodes, self.products_count, 3, device=self.device)
        current_stock = torch.cat([torch.zeros(1, self.products_count, device=self.device), 
                                self.init_capacities], dim=0)
        depot_demand = torch.zeros(1, self.products_count, device=self.device)
        for day in range(3):
            if self.cur_day.item() + day < self.planning_horizon:
                daily_demand_with_depot = torch.cat([depot_demand, 
                                                self.daily_demands[self.cur_day + day].squeeze(0)], dim=0)
                future_stock = current_stock - daily_demand_with_depot
                future_stock = torch.clamp(future_stock, min=0)
                future_stock_levels[:, :, day] = future_stock / self.max_capacities.max()
            else:
                future_stock_levels[:, :, day] = future_stock_levels[:, :, day-1] if day > 0 else current_stock / self.max_capacities.max()
        # Global features
        time_for_vehicle = self.working_time
        normalized_remaining_time = torch.nan_to_num(self.cur_remaining_time / time_for_vehicle, 0, posinf=0)
        global_features = torch.cat([
            self.vehicles.float().unsqueeze(0) / self.k_vehicles * self.max_trips,
            (self.cur_day / self.planning_horizon).float().unsqueeze(0),
            (self.vehicles <= 0).float().unsqueeze(0)
        ]).squeeze().float()

        state = {
            'normalized_remaining_time': normalized_remaining_time,
            'node_features': node_features,
            'edge_index': self.edge_indices,
            'edge_attr': self.edge_features,
            'global_features': global_features,
            'future_stock_levels': future_stock_levels
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
            'dry_runs_penalty':self.dry_runs_penalty,
            'closeness':self.closeness,
            'restricted_station_penalties': self.restricted_station,
            'revisit_1' : self.revisit_1,
            'revisit_2' : self.revisit_2,
            'revisit_2' : self.revisit_3,
            'revisit_3' : self.revisit_4,
            'overfill_penalty': self.overfill_penalty
        }
        average_routes = self.average_routes(self.station_list)
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

    def average_routes(self, seq):
        seq = [torch.tensor([x.item()], device=self.device) if not isinstance(x, (int, float)) else torch.tensor([x], device=self.device) for x in seq]
        seq = torch.cat(seq).tolist()
        # seq.insert(0, 0)
        # seq.insert(-1, 0)
        count = 0
        in_sequence = False

        for num in seq:
            if num != 0:
                if not in_sequence:
                    count += 1
                    in_sequence = True
            else:
                in_sequence = False

        return count