import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.data import Data
from torch_geometric.nn import GAT, TransformerConv, SAGPooling

class GATFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, embedding_size=64):
        super(GATFeatureExtractor, self).__init__(observation_space, features_dim=embedding_size)

        self.k_vehicles = observation_space['normalized_remaining_time'].shape[0]
        self.num_nodes = observation_space['node_features'].shape[0]
        self.products_count = 2

        node_base_dim = observation_space['node_features'].shape[1]
        global_input_dim = observation_space['global_features'].shape[0]
        edge_attr_dim = observation_space['edge_attr'].shape[1]

        # TransformerConv head with nn.Sequential
        self.transformer_gat = nn.Sequential(
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

        # Single GAT head with num_layers=3
        self.gat = GAT(
            in_channels=node_base_dim,
            hidden_channels=embedding_size,
            out_channels=embedding_size,
            num_layers=2,
            heads=8,
            concat=True,
            edge_dim=edge_attr_dim
        )

        # Pooling layer, adjusted for concatenated input
        self.sag_pool = SAGPooling(embedding_size * 2, ratio=1)

        # Global and time feature processing
        self.global_linear = nn.Sequential(
            nn.Linear(global_input_dim, embedding_size),
            nn.LeakyReLU(0.1),
            nn.Linear(embedding_size, embedding_size)
        )
        self.time_linear = nn.Sequential(
            nn.Linear(self.k_vehicles, embedding_size),
            nn.LeakyReLU(0.1),
            nn.Linear(embedding_size, embedding_size)
        )

        # Final linear layers, fixed for correct input dimension
        self.final_linear = nn.Sequential(
            nn.Linear(embedding_size * 4, embedding_size),  # x_pooled (2*embedding_size), global, time
            nn.LeakyReLU(0.1),
            nn.Linear(embedding_size, embedding_size),
            nn.Linear(embedding_size, embedding_size)
        )

    def forward(self, observations):
        node_features, edge_index, edge_attr, batch = self.convert_to_pyg_format(observations)

        # TransformerConv head
        x_trans = node_features
        for layer in self.transformer_gat[:-1]:
            x_trans = layer(x_trans, edge_index, edge_attr)
            x_trans = F.leaky_relu(x_trans, negative_slope=0.1)
        x_trans = self.transformer_gat[-1](x_trans, edge_index, edge_attr)

        # GAT head
        x_gat = self.gat(node_features, edge_index, edge_attr=edge_attr)

        # Concatenate heads
        x = torch.cat([x_trans, x_gat], dim=-1)

        # Pooling
        edge_weight = edge_attr.squeeze()
        x, _, _, _, _, _ = self.sag_pool(x, edge_index, edge_weight, batch)

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