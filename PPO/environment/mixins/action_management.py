import torch

class ActionManagement:
    def _update_time_and_location(self, actions):
        action_time = self.get_distance(self.current_location, actions)
        action_time = action_time.T.squeeze()
        self.current_location = actions
        selected_service_times = self.service_times[actions].squeeze()
        self.cur_remaining_time -= action_time + selected_service_times
        self.cur_remaining_time[self.cur_remaining_time < 0] = 0
        cur_to_any_time = self.weight_matrixes[self.current_location].squeeze()
        self.possible_action_time = cur_to_any_time + self.service_times + self.dist_to_depot

        self.update_edges()

    def _update_load(self, actions):
        if self.load_policy == 'fill_forward':
            self._fill_forward_load_policy(actions)
        elif self.load_policy == 'full_fill':
            self._full_fill_load_policy(actions)

        self.init_capacities[actions] += self.delivery
        percent = self.delivery / self.load[actions, :].squeeze(0)
        percent = percent
        percent_reduction = self.load * percent
        self.temp_load -= percent_reduction.type(self.temp_load.dtype).squeeze()
        self.temp_load[self.temp_load <= 0.01] = 0

    def _handle_depot_visits(self, actions):
        vehicle_in_depot = actions == self.depots
        if vehicle_in_depot:
            self.temp_load = self.load
            self.vehicles -= 1
            self.temp_vehicle = (self.vehicles - 1) % self.max_trips
            self.load = self.vehicle_compartments[self.temp_vehicle].squeeze()
            time_for_vehicle = self.working_time[self.temp_vehicle]
            self.cur_remaining_time = time_for_vehicle

    def _handle_day_end(self):
        self.day_end = (self.vehicles <= 0).item()
        if self.day_end:
            self.vehicles = torch.ones(1, dtype=torch.long, device=self.device) * self.k_vehicles * self.max_trips
            self.demands = self.daily_demands[self.cur_day].squeeze(0)
            self.init_capacities -= self.demands.double()
            self.init_capacities[self.init_capacities < 0] = 0
            self.cur_day += 1

    def _full_fill_load_policy(self, actions):
        full_fill_up = self.max_capacities - self.init_capacities
        selected_delivery = full_fill_up[actions, :].squeeze(0)
        selected_temp_load = self.temp_load[actions, :].squeeze(0)
        self.delivery = torch.min(selected_delivery, selected_temp_load)

    def is_done(self):
        return (self.cur_day >= self.planning_horizon).item()

    def average_routes(self, seq):
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
