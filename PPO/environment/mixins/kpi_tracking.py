import torch

class KPITracking:
    def calc_step_kpis(self):
        self.station_list.append(self.station_idx)

        current_location, next_location = self.vehicle_prev_locations[self.vehicle], self.vehicle_updated_locations[self.vehicle]

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
            self.total_vehicle_capacities += self.vehicle_compartments[self.vehicle].sum()

        self.total_stops += torch.sum(self.station_idx != self.depots)  # Депо не в диапазоне станций

        self.days_completed += (self.cur_day == self.planning_horizon - 1).float()
    
    def get_reward(self):

        # Штраф за пересыхание
        if self.day_end:
            dry_runs_mask = self.init_capacities < self.min_capacities
            dry_stations_mask = torch.any(dry_runs_mask, dim=1)
            num_dry_stations = dry_stations_mask.sum().item()

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

        penalties = self.get_penalty() 

        total_reward = self.dry_runs_penalty + self.dist + penalties
        return total_reward

    def get_penalty(self):
        # Расчет всех штрафов
        self.time_end = 0
        self.empty_load = 0
        self.restricted_station = 0
        self.revisit = 0

        # Штраф за превышение времени
        if self.cur_remaining_time[self.vehicle].item() <= 0:
            self.time_end = -5
        # Штраф за пустую загрузку
        elif torch.all(self.temp_load[self.vehicle] == 0) and torch.all(self.delivery == 0):
            self.empty_load = -5
        # Штраф за ограниченную станцию
        elif self.restriction_matrix[self.vehicle, int(self.station_idx)] == 1:
            self.restricted_station = -1
       
        # elif self.step_count > 1:  # Проверяем историю только если есть предыдущие действия
        #     prev_location, upd_location = self.vehicle_prev_locations[self.vehicle], self.vehicle_updated_locations[self.vehicle]
        #     prev_day_end, upd_day_end = self.prev_day_end[self.vehicle], self.updated_day_end[self.vehicle]

        #     if prev_location == upd_location and\
        #         ((prev_day_end == False and upd_day_end == True) or (prev_day_end == False and upd_day_end == False))and\
        #             prev_location != self.depots and upd_location != self.depots:
        #             self.revisit = -5

        #     if prev_location == self.depots and upd_location == self.depots and\
        #         not(prev_day_end and upd_day_end):
        #         self.depot_revisit = -5
        revisits = self.revisit_1 + self.revisit_2 + self.revisit_3 + self.revisit_4

        return self.time_end + self.empty_load + self.restricted_station + revisits #+ self.revisit + self.depot_revisit  