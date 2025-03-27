import torch

class KPITracking:
    def calc_step_kpis(self, station_idx, vehicle):
        self.station_list.append(station_idx)

        current_location, next_location = self.vehicle_prev_locations[vehicle], self.vehicle_updated_locations[vehicle]

        distances = self.get_distance(current_location, next_location)
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

        self.total_stops += torch.sum(station_idx != self.depots)  # Депо не в диапазоне станций

        self.days_completed += (self.cur_day == self.planning_horizon - 1).float()
    
    def get_reward(self, vehicle):
        prev_location, upd_location = self.vehicle_prev_locations[vehicle], self.vehicle_updated_locations[vehicle]

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
        penalties = self.get_penalty()  # Остальные штрафы (time_end, empty_load, restricted_station, revisit)

        distance = self.get_distance(prev_location, upd_location).item()
        if self.day_end:
            distance += self.get_distance(upd_location, self.depots).item()
            
        normalized_distance = distance / max_distance
        self.dist = -1 * normalized_distance

        total_reward = self.dry_runs_penalty + self.dist + penalties
        return total_reward

    def get_penalty(self):
        # Расчет всех штрафов
        self.time_end = 0
        self.empty_load = 0
        self.restricted_station = 0
        self.revisit = 0

        current_action = self.action_history[-1]
        vehicle, station_idx, end_day_flag = current_action[0], current_action[1], current_action[-1]
    
        # Штраф за превышение времени
        if self.cur_remaining_time[vehicle].item() <= 0:
            self.time_end = -5
        # Штраф за пустую загрузку
        elif torch.all(self.temp_load[vehicle] == 0) and torch.all(self.delivery == 0):
            self.empty_load = -5
        # Штраф за ограниченную станцию
        elif self.restriction_matrix[vehicle, int(station_idx)] == 1:
            self.restricted_station = -1
       
        elif self.step_count > 1:  # Проверяем историю только если есть предыдущие действия
            prev_location, upd_location = self.vehicle_prev_locations[vehicle], self.vehicle_updated_locations[vehicle]
            prev_day_end, upd_day_end = self.vehicles_prev_day_end[vehicle], self.vehicles_updated_day_end[vehicle]

            if prev_location == upd_location and\
                ((prev_day_end == False and upd_day_end == True) or (prev_day_end == False and upd_day_end == False))and\
                    prev_location != self.depots and upd_location != self.depots:
                    self.revisit = -5

            if prev_location == self.depots and upd_location == self.depots and\
                not(prev_day_end and upd_day_end):
                self.depot_revisit = -5

        return self.time_end + self.empty_load + self.restricted_station + self.revisit + self.depot_revisit  