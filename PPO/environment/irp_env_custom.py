import torch
import numpy as np

from typing import Tuple
from gymnasium import spaces, Env
from PIL import Image, ImageDraw, ImageFont

from .mixins.action_management import ActionManagement
from .mixins.kpi_tracking import KPITracking
from .mixins.render_utils import RenderUtils
from .mixins.state_utils import StateUtils
from .mixins.utility_functions import IRPEnvUtilitiesMixin

class IRPEnv_Custom(Env, ActionManagement, KPITracking, RenderUtils, StateUtils, IRPEnvUtilitiesMixin):

    def __init__(self, batch, parameters_dict, seed: int = 69):
        super(IRPEnv_Custom, self).__init__()

        self._initialize_parameters(parameters_dict)
        self._set_seed(seed)
        self.batch = batch
        self.reset(seed)

        self.action_space = spaces.Discrete(self.num_nodes)

        self.observation_space = spaces.Dict({
            'node_features': spaces.Box(low=-np.inf, high=np.inf,
                                        shape=(self.num_nodes, self.products_count * 3), dtype=np.float32),
            'edge_index': spaces.Box(low=0, high=self.num_nodes, shape=(2, self.num_nodes * (self.num_nodes - 1)),
                                     dtype=np.int64),
            'edge_attr': spaces.Box(low=0, high=np.inf, shape=(self.num_nodes * (self.num_nodes - 1), 1),
                                    dtype=np.float32),
            'global_features': spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
        })

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, dict, bool]:
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions).unsqueeze(0).to(self.device)

        self.action_history.append(actions.item())
        self.step_count += 1

        traversed_edges = torch.cat([self.current_location, actions], dim=0).long()

        self._update_time_and_location(actions)
        self._update_load(actions)
        self._handle_depot_visits(actions)
        self._handle_day_end()

        done = self.is_done()

        self.calc_step_kpis(actions, traversed_edges)
        total_reward = self.get_reward(traversed_edges)
        return self.get_state(), total_reward, done, done, self.get_kpis()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._unpack_batch()
        self._init_tracking_variables()
        self._initialize_tensors()
        self._initialize_variables()
        self._calculate_initial_state()
        return self.get_state(), self.get_kpis()

    def render(self, mode='rgb_array', **kwargs):
        probs = kwargs.get('probs', None)
        img_size = 800
        img = Image.new('RGB', (img_size, img_size), color='white')
        draw = ImageDraw.Draw(img)

        scale = (img_size - 500) / max(self.positions[0].max().item(), 1)

        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()

        self.draw_nodes(draw, scale, font)
        self.draw_edges(draw, scale, font)

        current_pos = self.positions[self.current_location].cpu().numpy().squeeze() * scale + 50
        draw.ellipse([current_pos[0] - 7, current_pos[1] - 7, current_pos[0] + 7, current_pos[1] + 7], fill=(255, 0, 0))

        self.draw_text(draw, font)

        # Если переданы вероятности, отображаем их на изображении
        if probs is not None:
            img = self.plot_action_probabilities_on_image(self, img, probs)

        return img
