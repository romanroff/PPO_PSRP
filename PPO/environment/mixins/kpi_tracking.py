import torch

class KPITracking:
    def calc_step_kpis(self, actions, traversed_edges):
        self.actions_list.append(actions)  # actions уже скаляр (station_idx)

        distances = self.get_distance(traversed_edges[0], traversed_edges[1])
        self.total_travel_distance += distances

        # Dry runs
        if self.day_end:
            dry_runs_mask = (self.init_capacities < self.min_capacities)
            dry_runs = dry_runs_mask.sum()
            self.total_dry_runs += dry_runs

            # Average stock level
            stock_levels = self.init_capacities
            stock_levels_masked = torch.nan_to_num(stock_levels)
            avg_stock_level = stock_levels_masked.sum()
            self.total_stock_level[self.cur_day - 1] += avg_stock_level

            stock_levels_percent = torch.nan_to_num(
                self.init_capacities.sum() / self.max_capacities.sum(), 0)
            self.average_stock_levels_percent += stock_levels_percent

        # Vehicle utilisation
        self.total_delivered_quantity += self.delivery.sum()
        if self.delivery.sum().item() > 0:
            self.total_vehicle_capacities += self.vehicle_compartments[self.temp_vehicle, actions].sum()

        # Number of stops and trips
        self.total_stops += torch.sum(actions != self.depots)

        # Update days completed
        self.days_completed += (self.cur_day == self.planning_horizon - 1).float()

    def get_reward(self, traversed_edges):
        # Штраф за пересыхания
        num_dry_stations = torch.any(self.init_capacities < self.min_capacities, dim=1).sum().item()
        self.dry_runs_penalty = -20 * (num_dry_stations / self.num_nodes)
        
        # Штраф за расстояние
        max_distance = self.weight_matrixes.max().item()
        distance = self.get_distance(traversed_edges[0], traversed_edges[1]).item()
        normalized_distance = distance / max_distance if max_distance > 0 else 0
        self.dist = -1 * normalized_distance
        
        # Дополнительные штрафы
        self.get_penalty()  # Вызываем get_penalty, чтобы обновить self.time_end и др.
        
        # Итоговая награда
        total_reward = self.dry_runs_penalty + self.dist + self.time_end + self.empty_load + self.restricted_station + self.revisit
        
        return total_reward

    def get_capacity_reward(self):
        norm_min = torch.nan_to_num(self.min_capacities / self.max_capacities, posinf=0)
        norm_temp = torch.nan_to_num(self.init_capacities / self.max_capacities, posinf=0)

        center = (norm_min + 1) / 2
        width = (1 - norm_min) / 2
        capacity_reward = 1 - ((norm_temp - center) / width) ** 2
        capacity_reward = capacity_reward / (self.products_count * (self.num_nodes - 1))

        dry_runs_mask = self.init_capacities < self.min_capacities
        self.capacity_reward = capacity_reward[~dry_runs_mask].sum().item()

        return self.capacity_reward

    def get_dist_reward(self, traversed_edges):
        time_for_vehicle = self.working_time[self.temp_vehicle]
        dist = self.get_distance(traversed_edges[0], traversed_edges[1]) / time_for_vehicle
        self.dist = -dist.item()
        return self.dist

    def get_penalty(self):
        # Инициализация дополнительных штрафов
        self.time_end = 0
        self.empty_load = 0
        self.restricted_station = 0
        self.revisit = 0
        
        current_action = self.action_history[-1]

        # Штраф за превышение времени
        if self.cur_remaining_time.item() <= 0:
            self.time_end = -2
        # Штраф за пустую загрузку и отсутствие доставки
        elif torch.all(self.temp_load == 0) and torch.all(self.delivery == 0):
            self.empty_load = -2
        # Штраф за посещение запрещённой станции
        elif self.restriction_matrix[self.temp_vehicle, current_action] == 1:
            self.restricted_station = -2
        # Штраф за повторное посещение
        elif len(self.action_history) > 1:
            prev_action = self.action_history[-2]
            if prev_action == current_action and current_action != 0:
                self.revisit = -2

        return self.time_end + self.empty_load + self.restricted_station + self.revisit
