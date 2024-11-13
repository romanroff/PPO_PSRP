import time
import torch

class StateUtils:
    def _initialize_parameters(self, parameters_dict):
        for key, value in parameters_dict.items():
            setattr(self, key, value)

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        self.device = torch.device("cpu")  #"cuda:0" if torch.cuda.is_available() else

    def _unpack_batch(self):
        self.positions, self.weight_matrixes, self.daily_demands, \
            self.depots, self.working_time, self.restriction_matrix, \
            self.service_times, self.min_capacities, self.max_capacities, \
            self.init_capacities, self.vehicle_compartments = self.batch

    def _init_tracking_variables(self):
        # Initialize KPI tracking variables
        self.action_history = [0]
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
        self.dry_runs_penalty = 0

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

        self.depots = self.depots.long()
        self.daily_demands = torch.stack([demand for demand in self.daily_demands]).to(self.device)

    def _initialize_variables(self):
        self.cur_day = torch.zeros(1, dtype=torch.long, device=self.device)
        self.dry_runs_duration = 0
        self.current_location = self.depots
        self.vehicles = torch.ones(1, dtype=torch.long, device=self.device) * self.k_vehicles * self.max_trips
        self.temp_vehicle = (self.vehicles - 1) % self.max_trips
        self.cur_remaining_time = self.working_time[self.temp_vehicle]
        self.load = self.vehicle_compartments[self.temp_vehicle].squeeze(0)
        self.demands = self.daily_demands[self.cur_day].squeeze()

        self.mock_edge_matrix()

        self.update_edges()

    def _calculate_initial_state(self):
        self.actions_list = [torch.tensor([0], device=self.device)]
        self.avarage_stocks = torch.zeros(self.planning_horizon, device=self.device)
        self.dry_runs_dict = torch.zeros(self.planning_horizon, device=self.device)
        self.actions_daily = torch.zeros(1, device=self.device)
        self.loss_dry_runs = torch.zeros(1, device=self.device)
        self.delivery = torch.zeros((1, self.products_count), device=self.device)

        self.dist_to_depot = self.weight_matrixes[self.depots].squeeze()
        cur_to_any_time = self.weight_matrixes[self.current_location].squeeze()
        self.possible_action_time = cur_to_any_time + self.service_times + self.dist_to_depot

        self.temp_load = self.load.clone().squeeze()

    def update_edges(self):
        temp_hour = self.working_hours[self.temp_vehicle] - self.cur_remaining_time / (60 * 60)
        temp_hour = temp_hour.type(torch.int32)
        self.weight_matrixes = self.daily_matrixes[self.cur_day, temp_hour].squeeze(0)
        self.edge_indices = (self.weight_matrixes > 0).nonzero(as_tuple=False).t().contiguous()
        time_for_vehicle = self.working_time[self.temp_vehicle]
        self.edge_features = (self.weight_matrixes / time_for_vehicle)[
            self.edge_indices[0], self.edge_indices[1]].unsqueeze(-1).float()

    def mock_edge_matrix(self):
        # Create daily_matrixes
        temp_hour = int(self.working_hours[self.temp_vehicle].item())
        self.daily_matrixes = torch.stack([self.weight_matrixes] * self.planning_horizon * (temp_hour + 1)).reshape(
            self.planning_horizon, temp_hour + 1, self.num_nodes, self.num_nodes
        ).to(self.device)

        decay_factor = torch.linspace(1.0, 1.0, steps=self.planning_horizon).unsqueeze(1).unsqueeze(2).unsqueeze(3).to(
            self.device)
        self.daily_matrixes *= decay_factor
        self.daily_matrixes = torch.round(self.daily_matrixes)
