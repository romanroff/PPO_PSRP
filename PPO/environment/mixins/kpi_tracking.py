import torch

class KPITracking:
    def calc_step_kpis(self, actions, traversed_edges, vehicle):
        self.actions_list.append(actions)

        distances = self.get_distance(traversed_edges[0], traversed_edges[1])
        self.total_travel_distance += distances

        if self.day_end:
            dry_runs_mask = (self.init_capacities < self.min_capacities)
            dry_runs = dry_runs_mask.sum()
            self.total_dry_runs += dry_runs

            stock_levels = self.init_capacities
            stock_levels_masked = torch.nan_to_num(stock_levels)
            avg_stock_level = stock_levels_masked.sum()
            self.total_stock_level[self.cur_day - 1] += avg_stock_level

            stock_levels_percent = torch.nan_to_num(
                self.init_capacities.sum() / self.max_capacities.sum(), 0)
            self.average_stock_levels_percent += stock_levels_percent

        self.total_delivered_quantity += self.delivery.sum()
        if self.delivery.sum().item() > 0:
            self.total_vehicle_capacities += self.vehicle_compartments[vehicle].sum()

        self.total_stops += torch.sum(actions != self.depots)  # Депо не в диапазоне станций

        self.days_completed += (self.cur_day == self.planning_horizon - 1).float()

    # def get_reward(self, traversed_edges):
        
    #     current_location = traversed_edges[0]  # Текущая позиция
    #     next_location = traversed_edges[1]  # Следующая позиция

    #     max_distance = self.weight_matrixes.max().item()

    #     # Определяем пересохшие станции
    #     if self.day_end:
    #         dry_runs_mask = self.init_capacities < self.min_capacities  # [num_stations, products_count]
    #         dry_stations_mask = torch.any(dry_runs_mask, dim=1)  # [num_stations], True если хоть один продукт пересох
    #         num_dry_stations = dry_stations_mask.sum().item()

    #         # Обновляем количество дней пересыхания только для станций
    #         if not hasattr(self, 'dry_days'):
    #             self.dry_days = torch.zeros(self.num_stations, dtype=torch.float32, device=self.device)  # [num_stations]
            
    #         # Увеличиваем счетчик дней для пересохших станций
    #         self.dry_days += dry_stations_mask.float()
    #         # Сбрасываем счетчик для станций, где запасы восстановлены
    #         self.dry_days *= dry_stations_mask.float()  # Обнуляем, если станция больше не пересохшая

    #         if num_dry_stations > 0:
    #             dry_stations = torch.where(dry_stations_mask)[0]  # Индексы пересохших станций
    #             dry_stations_in_matrix = dry_stations + 1  # Индексы станций в weight_matrixes смещены на +1 из-за депо
            
    #             # Штраф за пересохшие станции с учетом количества дней
    #             distances_to_dry = self.weight_matrixes[current_location, dry_stations_in_matrix]  # Расстояния до пересохших станций
    #             normalized_distances = distances_to_dry / max_distance
    #             # Множитель штрафа: базовый штраф увеличивается плавно с днями пересыхания
    #             dry_penalty_multiplier = 1 + 0.1 * self.dry_days[dry_stations]  # [num_dry_stations], +0.1 за каждый день
    #             weighted_distances = normalized_distances * dry_penalty_multiplier  # Увеличиваем штраф за долгие пересыхания
    #             sum_weighted_distances = weighted_distances.sum().item()
    #             self.dry_runs_penalty = -4 * sum_weighted_distances
    #         else:
    #             self.dry_runs_penalty = 0
    #             self.dry_days.fill_(0)  # Сбрасываем счетчик, если нет пересохших станций
    #     else:
    #         self.dry_runs_penalty = 0

    #     distance = self.get_distance(current_location, next_location).item()

    #     if self.day_end:
    #         distance += self.get_distance(next_location, self.depots).item()

    #     normalized_distance = distance / max_distance 
    #     self.dist = -1 * normalized_distance

    #     penalties = self.get_penalty()  # Остальные штрафы (time_end, empty_load, restricted_station, revisit)

    #     total_reward = self.dry_runs_penalty + self.dist  - 1 #+ penalties
    #     return total_reward
    
    def get_reward(self, traversed_edges):
        current_location = traversed_edges[0]
        next_location = traversed_edges[1]
        max_distance = self.weight_matrixes.max().item()

        # Штраф за пересыхание
        if self.day_end:
            dry_runs_mask = self.init_capacities < self.min_capacities
            dry_stations_mask = torch.any(dry_runs_mask, dim=1)
            num_dry_stations = dry_stations_mask.sum().item()

            # Обновляем количество дней пересыхания только для станций
            if not hasattr(self, 'dry_days'):
                self.dry_days = torch.zeros(self.num_stations, dtype=torch.float32, device=self.device)  # [num_stations]
            
            # Увеличиваем счетчик дней для пересохших станций
            self.dry_days += dry_stations_mask.float()
            # Сбрасываем счетчик для станций, где запасы восстановлены
            self.dry_days *= dry_stations_mask.float()  # Обнуляем, если станция больше не пересохшая

            if num_dry_stations > 0:
                dry_stations = torch.where(dry_stations_mask)[0]
                dry_penalty_multiplier = 1 + 0.1 * self.dry_days[dry_stations].sum().item()
                self.dry_runs_penalty = -4 * num_dry_stations #* dry_penalty_multiplier
            else:
                self.dry_runs_penalty = 0
                self.dry_days.fill_(0)  # Сбрасываем счетчик, если нет пересохших станций
        else:
            self.dry_runs_penalty = 0

        # Наказание за поездку
        distance = self.get_distance(current_location, next_location).item()
        if self.day_end:
            distance += self.get_distance(next_location, self.depots).item()
        normalized_distance = distance / max_distance
        self.dist = -1 * normalized_distance

        penalties = self.get_penalty()  # Остальные штрафы (time_end, empty_load, restricted_station, revisit)

        total_reward = self.dry_runs_penalty + self.dist + penalties
        return total_reward

    def get_dist_reward(self, traversed_edges):
        vehicle = self.action_history[-1][0]
        time_for_vehicle = self.working_time[vehicle]
        dist = self.get_distance(traversed_edges[0], traversed_edges[1]) / time_for_vehicle
        self.dist = -dist.item()
        return self.dist

    def get_penalty(self):
        self.time_end = 0
        self.empty_load = 0
        self.restricted_station = 0
        self.revisit = 0
        
        current_action = self.action_history[-1]
        vehicle = current_action[0]

        if self.cur_remaining_time[vehicle].item() <= 0: 
            self.time_end = -1
        elif torch.all(self.temp_load[vehicle] == 0) and torch.all(self.delivery == 0):
            self.empty_load = -1
        elif self.restriction_matrix[vehicle, int(current_action[1])] == 1:
            self.restricted_station = -1
        elif len(self.action_history) > 1:
            prev_action = self.action_history[-2]
            same_vehicle = prev_action[0] == current_action[0]
            same_node = prev_action[1] == current_action[1]
            not_in_depot = current_action[1] != self.depots
            not_day_end = prev_action[-1] != 1 and current_action[-1] != 1
            
            if  same_vehicle and same_node and not_in_depot and not_day_end:
                self.revisit = -2
        
        return self.time_end + self.empty_load + self.restricted_station + self.revisit