import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch

class GATFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, embedding_size=64):
        # Извлечение размерностей из observation_space
        node_feature_dim = observation_space['node_features'].shape[-1]
        global_feature_dim = observation_space['global_features'].shape[-1]
        time_feature_dim = observation_space['normalized_remaining_time'].shape[-1]
        self.num_nodes = observation_space['node_features'].shape[0]
        
        # Промежуточная размерность перед проекцией
        intermediate_dim = (embedding_size * self.num_nodes) + (node_feature_dim * self.num_nodes) + global_feature_dim + time_feature_dim
        
        # Итоговая размерность равна embedding_size
        features_dim = embedding_size
        
        # Инициализация родительского класса
        super().__init__(observation_space, features_dim=512)
        
        # GAT слой для обработки графа
        self.gat = GATv2Conv(
            in_channels=node_feature_dim,
            out_channels=embedding_size,
            heads=1,
            edge_dim=observation_space['edge_attr'].shape[-1]
        )
        
        # Проекционный слой
        self.projection = nn.Linear(intermediate_dim, 512)


    def convert_to_pyg_format(self, observations):
        """
        Преобразует наблюдения из SB3 в формат, подходящий для GATConv из PyTorch Geometric.

        Parameters:
            observations (dict): словарь с ключами:
                - 'node_features': [B, N, F_node] — признаки узлов
                - 'edge_index': [B, 2, E] — локальные индексы рёбер
                - 'edge_attr': [B, E, F_edge] — признаки рёбер

        Returns:
            x (torch.Tensor): узловые признаки [B*N, F_node]
            edge_index (torch.Tensor): глобальные индексы рёбер [2, B*E]
            edge_attr (torch.Tensor): признаки рёбер [B*E, F_edge]
            batch (torch.Tensor): индикатор принадлежности узла к графу [B*N]
        """
        # Извлечение данных
        node_features = observations['node_features']  # [B, N, F_node]
        edge_index = observations['edge_index'].long()  # [B, 2, E], приводим к int64
        edge_attr = observations['edge_attr']  # [B, E, F_edge]

        # Размеры
        B, N, F_node = node_features.shape
        _, _, E = edge_index.shape
        _, _, F_edge = edge_attr.shape

        # Создаем список объектов Data
        data_list = []
        for i in range(B):
            # Извлекаем данные для i-го графа
            x = node_features[i]  # [N, F_node]
            ei = edge_index[i]   # [2, E]
            ea = edge_attr[i]    # [E, F_edge]
            
            # Создаем объект Data
            data = Data(x=x, edge_index=ei, edge_attr=ea)
            data_list.append(data)

        # Объединяем в батч с помощью Batch.from_data_list
        batch = Batch.from_data_list(data_list)

        # Извлекаем необходимые компоненты
        x = batch.x  # [B*N, F_node]
        edge_index = batch.edge_index  # [2, B*E]
        edge_attr = batch.edge_attr  # [B*E, F_edge]
        batch_idx = batch.batch  # [B*N]

        return x, edge_index, edge_attr, batch_idx


    def forward(self, observations):
        # Преобразование в формат PyG
        node_features, edge_index, edge_attr, batch = self.convert_to_pyg_format(observations)
        
        # Обработка графа через GAT
        x = self.gat(node_features, edge_index, edge_attr=edge_attr)
        
        # Форматирование обработанных узловых признаков (без пулинга)
        batch_size = observations['node_features'].shape[0]
        x = x.view(batch_size, -1)  # [batch_size, num_nodes * embedding_size]
        
        # Сырые узловые признаки
        raw_node_features = observations['node_features'].view(batch_size, -1)
        
        # Сырые глобальные и временные признаки
        global_features = observations['global_features']
        time_features = observations['normalized_remaining_time']
        
        # Конкатенация всех признаков
        output = torch.cat([x, raw_node_features, global_features, time_features], dim=-1)
        # Проекция в embedding_size
        output = self.projection(output)

        return output