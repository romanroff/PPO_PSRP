import torch
class ActionManagement:
    def _update_time_and_location(self):
        prev_vehicle_location = self.vehicle_updated_locations[self.vehicle]
        self.vehicle_prev_locations[self.vehicle] = prev_vehicle_location
        self.vehicle_updated_locations[self.vehicle] = self.station_idx

        action_time = self.get_distance(prev_vehicle_location, self.station_idx)
        selected_service_times = self.service_times[self.station_idx].squeeze()

        self.cur_remaining_time[self.vehicle] -= action_time + selected_service_times
        self.cur_remaining_time = torch.clamp(self.cur_remaining_time, min=0)
        self.update_edges(self.vehicle)

    def _update_load(self):
        self.overfill_penalty = 0
        self.delivery = torch.zeros(self.products_count, device=self.device)
        if self.station_idx != self.depots.item():
            station_idx_for_capacities = self.station_idx - 1
            # Вычисляем возможное пополнение для всех продуктов сразу
            full_fill_up = self.max_capacities - self.init_capacities
            selected_temp_load = self.temp_load[self.vehicle, :]
            max_possible_delivery = torch.min(full_fill_up[station_idx_for_capacities, :].squeeze(0), selected_temp_load)
            # Вычисляем выбранную доставку на основе процентов
            selected_delivery = self.delivery_percents * selected_temp_load
            if torch.any(selected_delivery > max_possible_delivery):
                self.overfill_penalty = -1
            # Выбираем минимальную возможную доставку
            self.delivery = torch.min(selected_delivery, max_possible_delivery)
            # Обновляем init_capacities для всех продуктов
            self.init_capacities[station_idx_for_capacities, :] += self.delivery
            # Обновляем temp_load для всех продуктов
            self.temp_load[self.vehicle, :] -= self.delivery.type(self.temp_load.dtype)
            # Ограничиваем temp_load снизу, чтобы избежать отрицательных значений
            self.temp_load = torch.clamp(self.temp_load, min=0)

    def _handle_depot_visits(self):
        vehicle_in_depot = self.station_idx == self.depots
        
        if vehicle_in_depot:
            self.temp_load[self.vehicle] = self.vehicle_compartments[self.vehicle].clone()
            self.vehicles -= 1

    def _handle_day_end(self):
        no_more_vehicles = self.vehicles <= 0
        self.day_end = self.end_day_flag or no_more_vehicles

        self.prev_day_end = self.updated_day_end
        self.updated_day_end = self.day_end

        prev_location, upd_location = self.vehicle_prev_locations[self.vehicle], self.vehicle_updated_locations[self.vehicle]

        self.revisit_1 = 0
        self.revisit_2 = 0
        self.revisit_3 = 0
        self.revisit_4 = 0

        if prev_location == upd_location and not self.prev_day_end and not self.updated_day_end:
            self.revisit_1 = -4
        if prev_location == upd_location and not self.prev_day_end and self.updated_day_end:
            self.revisit_2 = -4
        if prev_location != self.depots and upd_location == self.depots and self.prev_day_end:
            self.revisit_3 = -4
        if prev_location == self.depots and upd_location == self.depots and self.prev_day_end and not self.updated_day_end:
            self.revisit_4 = -4

        # Наказание за поездку
        max_distance = self.weight_matrixes.max().item()
        
        distance = (self.get_distance(prev_location, upd_location).item() / max_distance)
        if self.day_end:

            for veh in self.vehicle_updated_locations:
                distance += (self.get_distance(veh, self.depots).item() / max_distance)

            self.vehicles = torch.ones(1, dtype=torch.long, device=self.device) * self.k_vehicles * self.max_trips
            self.demands = self.daily_demands[self.cur_day].squeeze(0)
            self.init_capacities -= self.demands.double()
            self.init_capacities = torch.clamp(self.init_capacities, min=0)
            self.cur_day += 1
            self.temp_load = self.vehicle_compartments.clone()
            self.cur_remaining_time = self.working_time.clone()

            prev_vehicle_location = self.vehicle_updated_locations
            self.vehicle_prev_locations = prev_vehicle_location
            self.vehicle_updated_locations = torch.full((self.k_vehicles,), self.depots.item(), dtype=torch.long, device=self.device)

        normalized_distance = distance
        self.dist = -1 * normalized_distance 

    def is_done(self):
        return (self.cur_day >= self.planning_horizon).item()