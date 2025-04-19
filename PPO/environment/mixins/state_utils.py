import time
import torch

class StateUtils:
    def _initialize_parameters(self, parameters_dict):
        for key, value in parameters_dict.items():
            setattr(self, key, value)

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        self.device = torch.device("cpu")

    def _unpack_batch(self):
        self.positions, self.weight_matrixes, self.daily_demands, \
            self.depots, self.working_time, self.restriction_matrix, \
            self.service_times, self.min_capacities, self.max_capacities, \
            self.init_capacities, self.vehicle_compartments = self.batch

        self.init_capacities = self.init_capacities.clone()
        self.vehicle_compartments = self.vehicle_compartments.clone()

        self.daily_demands = self.daily_demands[:, 1:, :]
        self.min_capacities = self.min_capacities[1:, :]
        self.max_capacities = self.max_capacities[1:, :]
        self.init_capacities = self.init_capacities[1:, :]
        self.num_stations = self.num_nodes - 1

        self.vehicle_compartments = self.vehicle_compartments[:, 0, :]
        # self.working_time = self.working_time[0].unsqueeze(0)
        # print(self.vehicle_compartments)

    def _init_tracking_variables(self):
        self.action_history = [(0, 0, torch.zeros(self.products_count, device=self.device), 0)]
        self.step_count = 0
        self.total_travel_distance = torch.zeros(1, device=self.device)
        self.total_stock_level = torch.zeros((self.planning_horizon), device=self.device)
        self.average_stock_levels_percent = torch.zeros(1, device=self.device)
        self.total_dry_runs = torch.zeros(1, device=self.device)
        self.total_delivered_quantity = torch.zeros(1, device=self.device)
        self.total_vehicle_capacities = torch.zeros(1, device=self.device)
        self.total_stops = torch.zeros(1, device=self.device)
        self.days_completed = torch.zeros(1, device=self.device)
        self.algorithm_start_time = time.time()
        self.capacity_reward = 0
        self.dist = 0
        self.time_end = 0
        self.empty_load = 0
        self.restricted_station = 0
        self.revisit = 0
        self.revisit_1 = 0
        self.revisit_2 = 0
        self.revisit_3 = 0
        self.revisit_4 = 0
        self.overfill_penalty = 0
        self.depot_revisit = 0
        self.dry_runs_penalty = 0
        self.closeness = 0
        self.render_steps = []
        # Новые переменные
        self.current_route = {v: [] for v in range(self.k_vehicles)}
        self.route_reward = torch.zeros(self.k_vehicles, device=self.device)
        self.delivery_reward  = 0
        self.all_routes = 0

    def _initialize_tensors(self):
        self.positions = self.positions.to(self.device)
        self.depots = self.depots.to(self.device)
        self.restriction_matrix = self.restriction_matrix.to(self.device)
        self.working_time = self.working_time.to(self.device)
        self.daily_demands = torch.stack([demand for demand in self.daily_demands]).to(self.device)
        self.weight_matrixes = self.weight_matrixes.to(self.device)
        self.service_times = self.service_times.to(self.device)
        self.min_capacities = self.min_capacities.to(self.device)
        self.max_capacities = self.max_capacities.to(self.device)
        self.init_capacities = self.init_capacities.to(self.device)
        self.vehicle_compartments = self.vehicle_compartments.to(self.device)
        self.working_hours = self.working_time / (60 * 60)
        self.vehicle = 0

        self.depots = self.depots.long()
        self.daily_demands = torch.stack([demand for demand in self.daily_demands]).to(self.device)

    def _initialize_variables(self):
        self.cur_day = torch.zeros(1, dtype=torch.long, device=self.device)
        self.dry_runs_duration = 0
        self.vehicles = torch.ones(1, dtype=torch.long, device=self.device) * self.k_vehicles * self.max_trips
        self.cur_remaining_time = self.working_time.clone()
        self.temp_load = self.vehicle_compartments.clone()
        self.demands = self.daily_demands[self.cur_day].squeeze()

        self.vehicle_final_locations = torch.full((self.k_vehicles,), self.depots.item(), dtype=torch.long, device=self.device)
        self.vehicle_updated_locations = torch.full((self.k_vehicles,), self.depots.item(), dtype=torch.long, device=self.device)
        self.vehicle_prev_locations = torch.full((self.k_vehicles,), self.depots.item(), dtype=torch.long, device=self.device)
        self.updated_day_end = False
        self.prev_day_end = False

        self.station_idx = self.depots

        self.dry_days = torch.zeros(self.num_stations, dtype=torch.float32, device=self.device)  # [num_stations]

        self.mock_edge_matrix()
        self.update_edges(0)

    def _calculate_initial_state(self):
        self.station_list = [torch.tensor([0], device=self.device)]
        self.avarage_stocks = torch.zeros(self.planning_horizon, device=self.device)
        self.dry_runs_dict = torch.zeros(self.planning_horizon, device=self.device)
        self.actions_daily = torch.zeros(1, device=self.device)
        self.loss_dry_runs = torch.zeros(1, device=self.device)
        self.delivery = torch.zeros(self.products_count, device=self.device)

    def update_edges(self, vehicle):
        temp_hour = self.working_hours[vehicle].item() - self.cur_remaining_time[vehicle].item() / (60 * 60)
        temp_hour = torch.tensor(temp_hour, dtype=torch.int32, device=self.device)
        temp_hour_max = self.daily_matrixes.shape[1] - 1
        temp_hour = torch.clamp(temp_hour, min=0, max=temp_hour_max)
        self.weight_matrixes = self.daily_matrixes[self.cur_day, temp_hour].squeeze(0)
        self.edge_indices = (self.weight_matrixes > 0).nonzero(as_tuple=False).t().contiguous()
        time_for_vehicle = self.working_time[vehicle]
        self.edge_features = (self.weight_matrixes / self.weight_matrixes.max().item())[
            self.edge_indices[0], self.edge_indices[1]].unsqueeze(-1).float()

    def mock_edge_matrix(self):
        temp_hour = int(self.working_hours.max().item())
        if temp_hour <= 0:
            temp_hour = 1
        self.daily_matrixes = torch.stack([self.weight_matrixes] * self.planning_horizon * (temp_hour + 1)).reshape(
            self.planning_horizon, temp_hour + 1, self.num_nodes, self.num_nodes
        ).to(self.device)

        decay_factor = torch.linspace(1.0, 1.0, steps=self.planning_horizon).unsqueeze(1).unsqueeze(2).unsqueeze(3).to(
            self.device)
        self.daily_matrixes *= decay_factor
        self.daily_matrixes = torch.round(self.daily_matrixes)