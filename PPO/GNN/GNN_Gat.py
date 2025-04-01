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
        self.num_nodes = observation_space['node_features'].shape[0]
        self.products_count = 2

        node_base_dim = observation_space['node_features'].shape[1]
        global_input_dim = observation_space['global_features'].shape[0]
        edge_attr_dim = observation_space['edge_attr'].shape[1]

        # RNN для обработки временных данных
        self.rnn = nn.GRU(
            input_size=self.products_count,
            hidden_size=embedding_size,
            num_layers=1,
            batch_first=True
        )

        # GAT с учетом увеличенного размера входных данных
        self.gat = nn.Sequential(
            TransformerConv(
                in_channels=node_base_dim + embedding_size,  # Добавляем размер RNN выхода
                out_channels=embedding_size,
                heads=8,
                edge_dim=edge_attr_dim,
                beta=True
            ),
            TransformerConv(
                in_channels=embedding_size * 8,
                out_channels=embedding_size,
                heads=4,
                edge_dim=edge_attr_dim,
                beta=True
            ),
            TransformerConv(
                in_channels=embedding_size * 4,
                out_channels=embedding_size,
                heads=1,
                edge_dim=edge_attr_dim,
                beta=True
            )
        )

        self.global_linear = nn.Sequential(
            nn.Linear(global_input_dim, embedding_size),
            nn.LeakyReLU(0.01),
            nn.Linear(embedding_size, embedding_size)
        )

        self.time_linear = nn.Sequential(
            nn.Linear(self.k_vehicles, embedding_size),
            nn.LeakyReLU(0.01),
            nn.Linear(embedding_size, embedding_size)
        )

        # Оставляем размер final_linear как был (5 компонентов)
        self.final_linear = nn.Sequential(
            nn.Linear(embedding_size * 5, embedding_size),
            nn.LeakyReLU(0.01),
            nn.Linear(embedding_size, embedding_size),
            nn.Linear(embedding_size, embedding_size)
        )

    def forward(self, observations):
        node_features, edge_index, edge_attr, batch = self.convert_to_pyg_format(observations)
        
        # Обработка временных данных через RNN
        future_stock = torch.tensor(observations['future_stock_levels'], dtype=torch.float32)  # [batch_size, num_nodes, products_count, 3]
        batch_size = future_stock.shape[0]
        future_stock = future_stock.view(batch_size * self.num_nodes, 3, self.products_count)  # [batch_size * num_nodes, timesteps, features]
        rnn_out, _ = self.rnn(future_stock)  # [batch_size * num_nodes, timesteps, embedding_size]
        rnn_out = rnn_out[:, -1, :]  # Берем последний выход [batch_size * num_nodes, embedding_size]
        
        # Добавляем RNN признаки к node_features
        node_features = torch.cat([node_features, rnn_out], dim=-1)  # [batch_size * num_nodes, node_base_dim + embedding_size]

        # Проходим через GAT слои
        x = node_features
        # Обработка всех слоев кроме последнего
        for layer in self.gat[:-1]:
            x = layer(x, edge_index, edge_attr)
            x = F.leaky_relu(x,negative_slope=0.01)
        # Обработка последнего слоя без активации
        x = self.gat[-1](x, edge_index, edge_attr)
        
        # Комбинированный пулинг
        x_add = global_add_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_add, x_mean, x_max], dim=-1)

        global_hidden = self.global_linear(observations['global_features'])
        time_hidden = self.time_linear(observations['normalized_remaining_time'])

        combined = torch.cat([x, global_hidden, time_hidden], dim=-1)
        output = self.final_linear(combined)

        return output

    def convert_to_pyg_format(self, observations):
        batch_size = observations['node_features'].shape[0]

        node_features = observations['node_features']
        edge_index = observations['edge_index'].to(torch.int64)
        edge_attr = observations['edge_attr']
        
        batch_indices = torch.arange(batch_size, device=node_features.device).repeat_interleave(self.num_nodes)
        data = Data(x=node_features.view(-1, node_features.shape[2]),
                    edge_index=edge_index.view(2, -1),
                    edge_attr=edge_attr.view(-1, edge_attr.shape[-1]),
                    batch=batch_indices)

        return data.x, data.edge_index, data.edge_attr, data.batch