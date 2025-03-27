import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GAT, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import TransformerConv
class GATFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, embedding_size=64):
        super(GATFeatureExtractor, self).__init__(observation_space, features_dim=embedding_size)

        self.k_vehicles = observation_space['normalized_remaining_time'].shape[0]
        self.num_nodes= observation_space['node_features'].shape[0]
        self.products_count = 2

        node_base_dim = observation_space['node_features'].shape[1]
        global_input_dim = observation_space['global_features'].shape[0]
        edge_attr_dim = observation_space['edge_attr'].shape[1]

        # GAT с четырьмя слоями
        self.gat = nn.Sequential(
            TransformerConv(
                in_channels=node_base_dim,
                out_channels=embedding_size,
                heads=8,
                edge_dim=edge_attr_dim,
                # dropout=0.05,
                beta=True  # Остаточные связи
            ),
            TransformerConv(
                in_channels=embedding_size * 8,  # Учитываем heads=8 из первого слоя
                out_channels=embedding_size,
                heads=4,
                edge_dim=edge_attr_dim,
                # dropout=0.05,
                beta=True
            ),
            TransformerConv(
                in_channels=embedding_size * 4,  # Учитываем heads=4 из второго слоя
                out_channels=embedding_size,
                heads=2,
                edge_dim=edge_attr_dim,
                # dropout=0.05,
                beta=True
            ),
            TransformerConv(
                in_channels=embedding_size * 2,  # Учитываем heads=2 из третьего слоя
                out_channels=embedding_size,
                heads=1,
                edge_dim=edge_attr_dim,
                # dropout=0.05,
                beta=True
            )
        )

        # Линейный слой для глобальных признаков
        self.global_linear = nn.Sequential(
            nn.Linear(global_input_dim, embedding_size),
            # nn.BatchNorm1d(embedding_size),
            nn.Tanh(),
            nn.Linear(embedding_size, embedding_size)
        )

        self.time_linear = nn.Sequential(
            nn.Linear(self.k_vehicles, embedding_size),
            # nn.BatchNorm1d(embedding_size),
            nn.Tanh(),
            nn.Linear(embedding_size, embedding_size)
        )

        # Исправляем размер входа в final_linear: 6 компонентов вместо 12
        self.final_linear = nn.Sequential(
            nn.Linear(embedding_size * 5, embedding_size),  # 3 пулинга + 3 компонента
            nn.Tanh(),
            nn.Linear(embedding_size, embedding_size),
            nn.Tanh(),
            # nn.Dropout(0.05),
            nn.Linear(embedding_size, embedding_size)
        )

    def forward(self, observations):
        node_features, edge_index, edge_attr, batch = self.convert_to_pyg_format(observations)

        # Проходим через все слои GAT
        x = node_features
        for layer in self.gat:
            x = layer(x, edge_index, edge_attr)
            x = F.tanh(x)  # Применяем активацию после каждого слоя
        
        # Комбинированный пулинг
        x_add = global_add_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_add, x_mean, x_max], dim=-1)  # [batch_size, embedding_size * 3]

        global_hidden = self.global_linear(observations['global_features'])  # [batch_size, embedding_size]
        time_hidden = self.time_linear(observations['normalized_remaining_time'])  # [batch_size, embedding_size]

        # Объединяем: 3 от пулинга + 3 от других компонентов
        combined = torch.cat([x, global_hidden, time_hidden], dim=-1)  # [batch_size, embedding_size * 6]
        output = self.final_linear(combined)  # [batch_size, embedding_size]

        return output

    def convert_to_pyg_format(self, observations):
        batch_size = observations['node_features'].shape[0]

        node_features = observations['node_features']  # [batch_size, num_stations, node_base_dim]
        edge_index = observations['edge_index'].to(torch.int64)  # [batch_size, 2, num_edges] или [2, num_edges]
        edge_attr = observations['edge_attr']  # [batch_size, num_edges, edge_attr_dim] или [num_edges, edge_attr_dim]
        
        batch_indices = torch.arange(batch_size, device=node_features.device).repeat_interleave(self.num_nodes)
        data = Data(x=node_features.view(-1, node_features.shape[2]),  # [batch_size * num_nodes, node_base_dim]
                    edge_index=edge_index.view(2, -1),  # [2, batch_size * num_edges]
                    edge_attr=edge_attr.view(-1, edge_attr.shape[-1]),  # [batch_size * num_edges, edge_attr_dim]
                    batch=batch_indices)  # [batch_size * num_nodes]

        return data.x, data.edge_index, data.edge_attr, data.batch