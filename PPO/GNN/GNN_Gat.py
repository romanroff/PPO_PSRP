import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GAT, SAGPooling


class GATFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, embedding_size=64):
        super(GATFeatureExtractor, self).__init__(observation_space, features_dim=embedding_size)

        # Входные размеры для узлов и глобальных признаков
        node_input_dim = observation_space['node_features'].shape[1]
        global_input_dim = observation_space['global_features'].shape[0]

        # Инициализируем готовую модель GAT
        self.gat = GAT(
            in_channels=node_input_dim,
            hidden_channels=embedding_size,
            num_layers=5,
            out_channels=embedding_size,
            heads=8,
            dropout=0.0,
            edge_dim=1
        )

        self.global_linear = torch.nn.Sequential(
            nn.Linear(global_input_dim, embedding_size),
            torch.nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            torch.nn.ReLU(),
            nn.Linear(embedding_size, embedding_size)
        )

        self.final_linear = torch.nn.Sequential(
            nn.Linear(embedding_size*2, embedding_size),
            torch.nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            torch.nn.ReLU(),
            nn.Linear(embedding_size, embedding_size)
        )

        self.pool = SAGPooling(in_channels=embedding_size, ratio=1)

    def forward(self, observations):
        node_features, edge_index, edge_attr, batch = self.convert_to_pyg_format(observations)
        global_features = observations['global_features']

        x = self.gat(x=node_features,
                     edge_index=edge_index,
                     edge_attr=edge_attr,
                     batch=batch)
        x = self.pool(x=x,
                      edge_index=edge_index,
                      edge_attr=edge_attr,
                      batch=batch)

        global_hidden = F.relu(self.global_linear(global_features))

        combined = torch.cat([x[0], global_hidden], dim=-1)

        output = self.final_linear(combined)
        return output

    def convert_to_pyg_format(self, observations):
        batch_data = observations['node_features']
        edge_index_batch = observations['edge_index']
        edge_attr_batch = observations['edge_attr']

        data_list = []
        batch_size = batch_data.shape[0]

        for i in range(batch_size):
            x = batch_data[i]
            edge_index = edge_index_batch[i].type(torch.int64)
            edge_attr = edge_attr_batch[i]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

            data_list.append(data)

        batch = Batch.from_data_list(data_list)

        return batch.x, batch.edge_index, batch.edge_attr, batch.batch

