import time
import torch

class IRPEnvUtilitiesMixin:
    def get_state(self) -> dict:
        node_features = torch.cat([
            torch.nan_to_num(self.demands / self.max_capacities, 0, posinf=0),
            torch.nan_to_num((self.init_capacities - self.min_capacities) / self.max_capacities, 0, posinf=0),
            torch.nan_to_num(self.temp_load / self.max_capacities, 0, posinf=0),
        ], dim=-1).float()

        time_for_vehicle = self.working_time[self.temp_vehicle]
        global_features = torch.tensor([
            self.vehicles.float(),
            (self.cur_remaining_time / time_for_vehicle).float(),
            (self.cur_day / self.planning_horizon).float(),
            (self.vehicles <= 0).float()
        ], device=self.device)

        state = {
            'node_features': node_features,
            'edge_index': self.edge_indices,
            'edge_attr': self.edge_features,
            'global_features': global_features
        }

        return self.tensors_to_numpy(state)

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
            'restricted_station_penalties': self.restricted_station,
            'revisit_penalties':  self.revisit
        }
        average_routes = self.average_routes(self.actions_list)
        kpis['average_stops_per_trip'] /= average_routes + 1e-6
        kpis['average_stops_per_trip'] = round(kpis['average_stops_per_trip'], 1)
        kpis['average_stock_levels'] = kpis['average_stock_levels'].cpu().tolist()

        return kpis

    def generate_mask(self):
        mask = torch.zeros((self.num_nodes), dtype=torch.int32, device=self.device)

        mask[self.current_location] = 1
        filled_nodes = torch.all(self.init_capacities == self.max_capacities, dim=1)
        mask[filled_nodes] = 1
        not_enough_time = self.possible_action_time >= self.cur_remaining_time
        mask[not_enough_time] |= 1
        empty_loads = torch.where(torch.all(self.temp_load <= 0.0, dim=1))[0]
        mask[empty_loads] |= 1
        all_filled = torch.where(torch.all(self.init_capacities == self.max_capacities, dim=1))[0]
        mask[all_filled] |= 1
        vehicle_restriction = self.restriction_matrix[self.temp_vehicle].squeeze()
        mask |= vehicle_restriction
        done_graphs = torch.where(self.cur_day == self.planning_horizon)[0]
        mask[done_graphs] = 1
        mask[self.depots.squeeze()] = 0

        return (~mask.bool()).cpu().numpy()

    def get_distance(self, node_idx_1: int, node_idx_2: int) -> float:
        return self.weight_matrixes[node_idx_1, node_idx_2]
