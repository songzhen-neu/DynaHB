import numpy as np
import torch
from torch_geometric.utils import scatter
from adgnn.context import context
import torch.distributed as dist


def _get_target_vertex(dataset):
    dataset.target_vertex = [
        {} for i in range(dataset._dataset["time_periods"] - dataset.lags)
    ]


def _calculate_deg(self):
    degs = []
    for i in range(self.snapshot_count):
        # num_nodes = int(self.edges[i].max()) + 1
        num_nodes=context.glContext.config['data_num']
        row, col = self.edges[i][0], self.edges[i][1]
        deg = scatter(torch.tensor(self.edge_weights[i]), torch.tensor(col), dim=0, dim_size=num_nodes,
                      reduce='sum')
        degs.append(deg.to(dtype=torch.float32))
    self.degs = degs


def _getPartition():
    with open(context.glContext.config['data_path'] + '/nodesPartition.' + context.glContext.config[
        'partitionMethod'] + str(context.glContext.config['worker_num']) + '.txt', 'r') as file:
        # 使用 readlines() 读取文件的所有行并存储在列表中
        partition_array = [np.array(line.strip().split('\t')) for line in file.readlines()]
        return partition_array


def _get_nei_unique(edges, target_vertex):
    mask_nei = np.isin(edges[1], target_vertex)
    neis = np.unique(edges[0][mask_nei])
    mask_nei_not_in_target = ~np.isin(neis, target_vertex)
    nei_unique = neis[mask_nei_not_in_target]
    return nei_unique


def _get_v_of_each_layer(self):
    partition = _getPartition()
    id = context.glContext.config['id']
    layer_num = context.glContext.config['layer_num']
    local_ids = partition[id].astype(int)
    target_vertex = [np.array([]) for i in range(layer_num + 1)]
    target_vertex[0] = local_ids

    for j in range(layer_num):
        target_vertex[j + 1] = target_vertex[j]
        for i in range(self.snapshot_count):
            nei_unique = _get_nei_unique(self.edges[i], target_vertex[j])
            target_vertex[j + 1] = np.union1d(target_vertex[j + 1], nei_unique)

    for i in range(self.snapshot_count):
        for j in range(layer_num + 1):
            self.target_vertex[i][j] = target_vertex[j]
    print('get target vertices of each layer end!!')


def _to_local_subgraph(self):
    layer_num = context.glContext.config['layer_num']
    for i in range(self.snapshot_count):
        ids = self.target_vertex[i][layer_num]
        mask_agg = np.isin(self.edges[i][1], ids)
        self.edges[i] = self.edges[i][:, mask_agg]
        self.edge_weights[i] = self.edge_weights[i][mask_agg]
        self.features[i] = self.features[i][ids, :]
        self.targets[i] = self.targets[i][ids]
        self.degs[i] = self.degs[i][ids]
    print("to local subgraph end")


def _encode_edge(self):
    for i in range(self.snapshot_count):
        self.edges[i] = np.array(
            [[self.old2new_maps[i].get(value, value) for value in row] for row in self.edges[i]])


def _encode(self):
    layer_num = context.glContext.config['layer_num']
    dist.barrier()
    old2new_maps = []

    for s in range(self.snapshot_count):
        old2new_tmp = {i: j for j, i in enumerate(self.target_vertex[s][layer_num])}
        old2new_maps.append(old2new_tmp)
        self.target_vertex[s] = np.array([old2new_tmp[i] for i in self.target_vertex[s][0]])
        dist.barrier()  # ensure all workers finished the i-th snapshot
    self.old2new_maps = old2new_maps


def start_cache(dataset):
    _get_target_vertex(dataset)
    _calculate_deg(dataset)
    _get_v_of_each_layer(dataset)
    _to_local_subgraph(dataset)
    _encode(dataset)
    _encode_edge(dataset)
