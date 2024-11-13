import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.data import Data, Batch

from torch_geometric.nn import SAGPooling, GATConv


class GraphResNetBlockGAT(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super(GraphResNetBlockGAT, self).__init__()
        # Первый графовый attention слой
        self.gat1 = GATConv(in_channels, out_channels // heads, heads=heads, concat=True, edge_dim=1)
        # Второй графовый attention слой
        self.gat2 = GATConv(out_channels, out_channels // heads, heads=heads, concat=True, edge_dim=1)
        # Нормализация
        self.bn = nn.BatchNorm1d(out_channels)
        # Shortcut для выравнивания размерностей
        self.shortcut = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None

    def forward(self, x, edge_index, edge_attr, batch):
        identity = x  # Сохраняем входной тензор для использования в shortcut
        # Первый GAT слой с активацией ReLU
        out = F.relu(self.bn(self.gat1(x, edge_index, edge_attr)))
        # Второй GAT слой
        out = self.bn(self.gat2(out, edge_index, edge_attr))

        # Если размеры не совпадают, пропускаем вход через shortcut
        if self.shortcut is not None:
            identity = self.shortcut(identity)

        # Резюмируем с использованием residual соединения (shortcut)
        out += identity
        return F.relu(out)


class GraphResNetGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=1):
        super(GraphResNetGAT, self).__init__()
        self.layers = nn.ModuleList()
        # Первый блок с преобразованием размерности
        self.layers.append(GraphResNetBlockGAT(in_channels, hidden_channels, heads))

        # Промежуточные блоки с постоянной размерностью
        for _ in range(num_layers - 2):
            self.layers.append(GraphResNetBlockGAT(hidden_channels, hidden_channels, heads))

        # Последний блок с преобразованием в выходное количество каналов
        self.layers.append(GraphResNetBlockGAT(hidden_channels, out_channels, heads))


    def forward(self, x, edge_index, edge_attr, batch):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, batch)

        return x

class GATFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, embedding_size=64):
        super(GATFeatureExtractor, self).__init__(observation_space, features_dim=embedding_size)

        # Входные размеры для узлов и глобальных признаков
        node_input_dim = observation_space['node_features'].shape[1]
        global_input_dim = observation_space['global_features'].shape[0]

        # Инициализируем готовую модель GAT
        # self.gat = GAT(
        #     in_channels=node_input_dim,
        #     hidden_channels=embedding_size,
        #     num_layers=5,
        #     out_channels=embedding_size,
        #     heads=8,
        #     dropout=0.0,
        #     edge_dim=1
        # )
        # Пример данных
        in_channels = node_input_dim
        hidden_channels = embedding_size
        out_channels = embedding_size
        num_layers = 5
        heads = 8  # Используем 4 головы для GAT

        # Создаем графовую сеть
        self.gat = GraphResNetGAT(in_channels, hidden_channels, out_channels, num_layers, heads)

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

