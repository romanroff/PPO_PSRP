import torch
class ActionManagement:
    def _update_time_and_location(self, station_idx, vehicle):
        current_vehicle_location = self.vehicle_locations[vehicle]
        action_time = self.get_distance(current_vehicle_location, station_idx)
        selected_service_times = self.service_times[station_idx].squeeze()

        self.vehicle_locations[vehicle] = station_idx
        self.cur_remaining_time[vehicle] -= action_time + selected_service_times
        self.cur_remaining_time = torch.clamp(self.cur_remaining_time, min=0)
        self.update_edges(vehicle)

    def _update_load(self, station_idx, delivery_percent, product_idx, vehicle):
        full_fill_up = self.max_capacities - self.init_capacities
        selected_temp_load = self.temp_load[vehicle, product_idx]
        max_possible_delivery = torch.min(full_fill_up[station_idx, product_idx].squeeze(0), selected_temp_load)
        selected_delivery = delivery_percent * selected_temp_load /100
        self.delivery[product_idx] = torch.min(selected_delivery, max_possible_delivery)
        self.init_capacities[station_idx, product_idx] += self.delivery[product_idx]
        self.temp_load[vehicle, product_idx] -= self.delivery[product_idx].type(self.temp_load.dtype)
        self.temp_load = torch.clamp(self.temp_load, min=0)

    def _handle_depot_visits(self, station_idx, vehicle):
        vehicle_in_depot = station_idx == self.depots
        
        if vehicle_in_depot:
            self.temp_load[vehicle] = self.vehicle_compartments[vehicle].clone()
            self.vehicles -= 1

    def _handle_day_end(self, end_day_flag):
        self.day_end = (end_day_flag == 1) or (self.vehicles <= 0)

        if self.day_end:
            self.vehicles = torch.ones(1, dtype=torch.long, device=self.device) * self.k_vehicles * self.max_trips
            self.demands = self.daily_demands[self.cur_day].squeeze(0)
            self.init_capacities -= self.demands.double()
            self.init_capacities = torch.clamp(self.init_capacities, min=0)
            self.cur_day += 1
            self.temp_load = self.vehicle_compartments.clone()
            self.cur_remaining_time = self.working_time.clone()
            self.vehicle_locations = torch.full((self.k_vehicles,), self.depots.item(), dtype=torch.long, device=self.device)
            self.render_vehicle_locations = self.vehicle_locations.clone()

    def is_done(self):
        return (self.cur_day >= self.planning_horizon).item()

    def average_routes(self, seq):
        seq = [torch.tensor([x], device=self.device) if isinstance(x, (int, float)) else x for x in seq]
        seq = torch.cat(seq).tolist()
        seq.insert(0, 0)
        seq.insert(-1, 0)
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