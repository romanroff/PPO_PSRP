import torch

class KPITracking:
    def calc_step_kpis(self):
        self.station_list.append(self.station_idx)

        current_location, next_location = self.vehicle_prev_locations[self.vehicle], self.vehicle_updated_locations[self.vehicle]

        distances = self.get_distance(current_location, next_location)


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
        self.delivery_reward = 0
        self.all_routes = 0
        # Обновляем маршрут
        if self.station_idx != self.depots.item() and self.delivery.sum() > 0:
            self.current_route.append(self.station_idx.item())
            delivery_normalized = self.delivery / self.vehicle_compartments[self.vehicle] 
            self.delivery_reward = delivery_normalized.sum().item() / self.products_count * 0.5



        num_stations_visited = len(set(self.current_route))
        if num_stations_visited > 1 and num_stations_visited <=3:
            self.route_reward = -1*self.dist * 0.25 * num_stations_visited
        

        if self.day_end:
            route_data = self.current_route
            num_stations_visited = len(set(route_data))
            if 1 > num_stations_visited <= 3:
                self.all_routes += self.dist * 0.25 * num_stations_visited
                self.route_reward = 0
            self.current_route = []


        self.dry_runs_penalty = 0
        self.overfill_penalty = 0
        if self.day_end:
            dry_runs_mask = self.init_capacities < self.min_capacities
            dry_stations_mask = torch.any(dry_runs_mask, dim=1)
            num_dry_stations = dry_stations_mask.sum().item()
            self.dry_runs_penalty = -8 * num_dry_stations

        overfill_mask = self.init_capacities > self.max_capacities
        overfilled_stations_mask = torch.any(overfill_mask, dim=1)
        num_overfilled_stations = overfilled_stations_mask.sum().item()
        self.overfill_penalty = -1 * num_overfilled_stations

        penalties = self.get_penalty() 

        self.dist = self.dist
        if self.dist < 0:
            self.dist *= 2

        total_reward = self.dry_runs_penalty + self.dist + penalties  + self.overfill_penalty #+ self.delivery_reward
        # if total_reward >= 0:
        #     total_reward += 2
        if self.revisit_1 + self.revisit_2 + self.revisit_3 + self.revisit_4 == 0:
            total_reward+=self.route_reward

        if self.station_idx == self.depots.item() or self.day_end:
            self.route_reward = 0.0
            self.current_route = []
            self.used_vehicles = []


        return total_reward

    def get_penalty(self):
        # Расчет всех штрафов
        self.time_end = 0
        self.empty_load = 0
        self.restricted_station = 0

        # Штраф за превышение времени
        if self.cur_remaining_time[self.vehicle].item() <= 0:
            self.time_end = -4
        # Штраф за пустую загрузку
        elif torch.all(self.temp_load[self.vehicle] == 0) and torch.all(self.delivery == 0):
            self.empty_load = -4
        # Штраф за ограниченную станцию
        elif self.restriction_matrix[self.vehicle, int(self.station_idx)] == 1:
            self.restricted_station = -1
       
        revisits = self.revisit_1 + self.revisit_2 + self.revisit_3 + self.revisit_4


        return self.time_end + self.empty_load + self.restricted_station + revisits