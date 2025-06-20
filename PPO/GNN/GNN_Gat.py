import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, SAGPooling, GATConv

class GATFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, embedding_size=64):
        super(GATFeatureExtractor, self).__init__(observation_space, features_dim=embedding_size)

        self.k_vehicles = observation_space['normalized_remaining_time'].shape[0]
        self.num_nodes = observation_space['node_features'].shape[0]
        self.products_count = 2

        node_base_dim = observation_space['node_features'].shape[1]
        global_input_dim = observation_space['global_features'].shape[0]
        edge_attr_dim = observation_space['edge_attr'].shape[1]

        # TransformerConv layers
        self.transformer_conv = nn.Sequential(
            TransformerConv(
                in_channels=node_base_dim,
                out_channels=embedding_size,
                heads=8,
                edge_dim=edge_attr_dim,
                beta=True
            ),
            TransformerConv(
                in_channels=embedding_size * 8,
                out_channels=embedding_size,
                heads=1,
                edge_dim=edge_attr_dim,
                beta=True
            ),
        )

        # Pooling layer with GATConv as GNN
        self.sag_pool = SAGPooling(embedding_size, ratio=1, GNN=GATConv)

        # Global and time feature processing
        self.global_linear = nn.Sequential(
            nn.Linear(global_input_dim, embedding_size),
            nn.LeakyReLU(0.1),
            nn.Linear(embedding_size, embedding_size),
            nn.LeakyReLU(0.1),
        )
        self.time_linear = nn.Sequential(
            nn.Linear(self.k_vehicles, embedding_size),
            nn.LeakyReLU(0.1),
            nn.Linear(embedding_size, embedding_size),
            nn.LeakyReLU(0.1),
        )

        # Final linear layers
        self.final_linear = nn.Sequential(
            nn.Linear(embedding_size * 3, embedding_size),  # x_pooled + global + time
            nn.LeakyReLU(0.1),
            nn.Linear(embedding_size, embedding_size),
            nn.LeakyReLU(0.1),
        )

    def forward(self, observations):
        node_features, edge_index, edge_attr, batch = self.convert_to_pyg_format(observations)

        # TransformerConv processing
        x = node_features
        for layer in self.transformer_conv:
            x = layer(x, edge_index, edge_attr)
            x = F.leaky_relu(x, negative_slope=0.1)

        # Pooling with GATConv
        edge_weight = edge_attr.squeeze()
        x, edge_index, edge_attr, batch, perm, score = self.sag_pool(x, edge_index, edge_weight, batch)

        # Global and time features
        global_hidden = self.global_linear(observations['global_features'])
        time_hidden = self.time_linear(observations['normalized_remaining_time'])

        # Combine all features
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