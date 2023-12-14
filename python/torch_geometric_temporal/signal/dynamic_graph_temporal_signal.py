import torch
import numpy as np
from typing import Sequence, Union
from torch_geometric.data import Data
# from adgnn.context.context import glContext
import torch.distributed as dist
import adgnn.context.context as context

Edges = Sequence[Union[np.ndarray, list, None]]
Edge_Weights = Sequence[Union[np.ndarray, list, None]]
Node_Features = Sequence[Union[np.ndarray, None]]
Targets = Sequence[Union[np.ndarray, None]]
Additional_Features = Sequence[np.ndarray]
Old2New_Maps = Sequence[Union[dict, None]]
Degs = Sequence[Union[np.ndarray, list, None]]
Target_Vertex = Sequence[Union[dict, list, None]]


class DynamicGraphTemporalSignal(object):
    r"""A data iterator object to contain a dynamic graph with a
    changing edge set and weights . The feature set and node labels
    (target) are also dynamic. The iterator returns a single discrete temporal
    snapshot for a time period (e.g. day or week). This single snapshot is a
    Pytorch Geometric Data object. Between two temporal snapshots the edges,
    edge weights, target matrices and optionally passed attributes might change.

    Args:
        edge_indices (Sequence of Numpy arrays): Sequence of edge index tensors.
        edge_weights (Sequence of Numpy arrays): Sequence of edge weight tensors.
        features (Sequence of Numpy arrays): Sequence of node feature tensors.
        targets (Sequence of Numpy arrays): Sequence of node label (target) tensors.
        **kwargs (optional Sequence of Numpy arrays): Sequence of additional attributes.
    """

    def __init__(
            self,
            edges: Edges,
            edge_weights: Edge_Weights,
            features: Node_Features,
            targets: Targets,
            target_vertex: Target_Vertex,
            degs: Degs,
            old2new_maps: Old2New_Maps,
            **kwargs: Additional_Features
    ):
        self.edges = edges
        self.edge_weights = edge_weights
        self.features = features
        self.targets = targets
        self.target_vertex = target_vertex
        self.degs = degs
        self.old2new_maps = old2new_maps
        self.additional_feature_keys = []

        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
        self._check_temporal_consistency()
        self._set_snapshot_count()

    def _check_temporal_consistency(self):
        assert len(self.features) == len(
            self.targets
        ), "Temporal dimension inconsistency."
        assert len(self.features) == len(
            self.edge_weights
        ), "Temporal dimension inconsistency."
        for key in self.additional_feature_keys:
            assert len(self.targets) == len(
                getattr(self, key)
            ), "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self.features)

    def _get_edge_weights(self, time_index: int):
        if self.edge_weights[time_index] is None or isinstance(self.edge_weights[time_index], list):
            return self.edge_weights[time_index]
        else:
            return torch.FloatTensor(self.edge_weights[time_index])

    def _get_edges(self, time_index: int):
        return self.edges[time_index]

    def _get_deg(self, time_index: int):
        return self.degs[time_index]

    def _get_features(self, time_index: int):
        if self.features[time_index] is None or isinstance(self.features[time_index], torch.Tensor):
            return self.features[time_index]
        else:
            return torch.FloatTensor(self.features[time_index])

    def _get_target(self, time_index: int):
        if self.targets[time_index] is None or isinstance(self.targets[time_index], torch.Tensor):
            return self.targets[time_index]
        else:
            if self.targets[time_index].dtype.kind == "i":
                return torch.LongTensor(self.targets[time_index])
            elif self.targets[time_index].dtype.kind == "f":
                return torch.FloatTensor(self.targets[time_index])

    def _get_additional_feature(self, time_index: int, feature_key: str):
        feature = getattr(self, feature_key)[time_index]
        if feature.dtype.kind == "i":
            return torch.LongTensor(feature)
        elif feature.dtype.kind == "f":
            return torch.FloatTensor(feature)

    def _get_additional_features(self, time_index: int):
        additional_features = {
            key: self._get_additional_feature(time_index, key)
            for key in self.additional_feature_keys
        }
        return additional_features

    def _get_old2new_maps(self, time_index: int):
        if self.old2new_maps is None:
            return None
        else:
            return self.old2new_maps[time_index]

    def _get_target_vertex(self, time_index: int):
        return self.target_vertex[time_index]

    def _get_nei_unique(self, edges, target_vertex):
        mask_nei = np.isin(edges[1], target_vertex)
        neis = np.unique(edges[0][mask_nei])
        mask_nei_not_in_target = ~np.isin(neis, target_vertex)
        nei_unique = neis[mask_nei_not_in_target]
        return nei_unique

    def _get_edge_and_edgeweight_by_target(self, target_vertex, edge, edge_weight):
        mask_agg = np.isin(edge[1], target_vertex)
        return edge[:, mask_agg], edge_weight[mask_agg]

    def _get_encode_edge(self, encode_vertex, edge):
        o2n_map = {j: i for i, j in enumerate(encode_vertex)}
        encode_edge = np.array(
            [[o2n_map.get(value, value) for value in row] for row in edge])
        return torch.LongTensor(encode_edge)

    def _get_v_of_each_layer(self, batch_size, window_id, window_size):
        layer_num = context.glContext.config['layer_num']
        target_vertex_batch = []
        target_vertex = [np.array([]) for i in range(layer_num + 1)]
        target_vertex[0] = np.random.choice(self.target_vertex[0], size=batch_size, replace=False)
        # target_vertex[0]=self.target_vertex[0]
        for j in range(layer_num):
            target_vertex[j + 1] = target_vertex[j]
            for i in range(window_id, window_id + window_size, 1):
                snapshot = self[i]
                nei_unique = self._get_nei_unique(snapshot.edge, target_vertex[j])
                target_vertex[j + 1] = np.concatenate((target_vertex[j + 1], nei_unique), axis=0)

        for i in range(window_id, window_id + window_size, 1):
            target_vertex_batch.append(target_vertex)

        return target_vertex_batch

    def construct_testable_data(self):
        layer_num = context.glContext.config['layer_num']
        edge_batch = []
        edge_weight_batch = []
        feature_batch = []
        target_batch = []
        target_vertex_batch = self._get_v_of_each_layer(len(self.target_vertex[0]), 0, self.snapshot_count)
        deg_batch = []

        for i in range(self.snapshot_count):
            snapshot = self[i]

            target_vertex_snap = target_vertex_batch[i]
            deg_layer = [None for i in range(layer_num + 1)]
            deg_layer[0] = snapshot.deg[target_vertex_snap[0]]
            edge_layer = [None for i in range(layer_num)]
            edge_weight_layer = [None for i in range(layer_num)]
            for j in range(layer_num):
                edge, edge_weight = self._get_edge_and_edgeweight_by_target(target_vertex_snap[j], snapshot.edge,
                                                                            snapshot.edge_weight)
                edge_encode = self._get_encode_edge(target_vertex_snap[j + 1], edge)
                edge_layer[j] = edge_encode
                edge_weight_layer[j] = edge_weight
                deg_layer[j + 1] = snapshot.deg[target_vertex_snap[j + 1]]

            edge_batch.append(edge_layer)
            edge_weight_batch.append(edge_weight_layer)
            feature_batch.append(snapshot.x[target_vertex_snap[layer_num]])
            target_batch.append(snapshot.y[target_vertex_snap[0]])
            deg_batch.append(deg_layer)

        data_batch = DynamicGraphTemporalSignal(edge_batch, edge_weight_batch, feature_batch, target_batch,
                                                target_vertex_batch, deg_batch, None)

        return data_batch

    def to_device(self, device):
        if device == 'cpu':
            return
        if device == 'cuda':
            for i in range(len(self.features)):
                self.features[i]=self.features[i].to(device)
                self.targets[i] = self.targets[i].to(device)
                for j in range(len(self.degs[i])):
                    self.degs[i][j]=self.degs[i][j].to(device)
                for j in range(len(self.edge_weights[i])):
                    self.edge_weights[i][j]=self.edge_weights[i][j].to(device)
                for j in range(len(self.edges[i])):
                    self.edges[i][j]=self.edges[i][j].to(device)

    def generate_batch(self, window_id):
        batch_size = context.glContext.config['batch_size']
        window_size = context.glContext.config['window_size']
        layer_num = context.glContext.config['layer_num']
        edge_batch = []
        edge_weight_batch = []
        feature_batch = []
        target_batch = []
        target_vertex_batch = self._get_v_of_each_layer(batch_size, window_id, window_size)
        deg_batch = []

        for i in range(window_id, window_id + window_size, 1):
            snapshot = self[i]

            target_vertex_snap = target_vertex_batch[i - window_id]
            deg_layer = [None for i in range(layer_num + 1)]
            deg_layer[0] = snapshot.deg[target_vertex_snap[0]]
            edge_layer = [None for i in range(layer_num)]
            edge_weight_layer = [None for i in range(layer_num)]
            for j in range(layer_num):
                edge, edge_weight = self._get_edge_and_edgeweight_by_target(target_vertex_snap[j], snapshot.edge,
                                                                            snapshot.edge_weight)
                edge_encode = self._get_encode_edge(target_vertex_snap[j + 1], edge)
                edge_layer[j] = edge_encode
                edge_weight_layer[j] = edge_weight
                deg_layer[j + 1] = snapshot.deg[target_vertex_snap[j + 1]]

            edge_batch.append(edge_layer)
            edge_weight_batch.append(edge_weight_layer)
            feature_batch.append(snapshot.x[target_vertex_snap[layer_num]])
            target_batch.append(snapshot.y[target_vertex_snap[0]])
            deg_batch.append(deg_layer)

        data_batch = DynamicGraphTemporalSignal(edge_batch, edge_weight_batch, feature_batch, target_batch,
                                                target_vertex_batch, deg_batch, None)

        return data_batch

    def __getitem__(self, time_index: Union[int, slice]):
        if isinstance(time_index, slice):
            snapshot = DynamicGraphTemporalSignal(
                self.edges[time_index],
                self.edge_weights[time_index],
                self.features[time_index],
                self.targets[time_index],
                self.target_vertex[time_index],
                self.degs[time_index],
                self.old2new_maps[time_index],
                **{key: getattr(self, key)[time_index] for key in self.additional_feature_keys}
            )
        else:

            edge = self._get_edges(time_index)
            edge_weight = self._get_edge_weights(time_index)
            x = self._get_features(time_index)
            y = self._get_target(time_index)
            target_vertex = self._get_target_vertex(time_index)
            deg = self._get_deg(time_index)
            old2new_map = self._get_old2new_maps(time_index)
            additional_features = self._get_additional_features(time_index)

            snapshot = Data(x=x, edge=edge, edge_weight=edge_weight,
                            y=y, target_vertex=target_vertex, deg=deg, old2new_map=old2new_map, **additional_features)
        return snapshot

    def __next__(self):
        if self.t < len(self.features):
            snapshot = self[self.t]
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self
