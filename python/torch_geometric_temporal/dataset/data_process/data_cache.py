import numpy as np
import torch
from torch_geometric.utils import scatter
from adgnn.context import context
import torch.distributed as dist

from adgnn.system_optimization.partition import workloadAwarePartition
from adgnn.util_python.timecounter import time_counter


def _get_target_vertex(dataset):
    # dataset.target_vertex = [
    #     {} for i in range(dataset._dataset["time_periods"] - dataset.lags)
    # ]
    dataset.target_vertex = [
        {} for i in range(dataset.snapshot_count)
    ]


def _calculate_deg(self):
    degs = []
    time_counter.start_single('_calculate_deg')
    for i in range(self.snapshot_count):
        # num_nodes = int(self.edges[i].max()) + 1
        num_nodes = context.glContext.config['data_num']
        row, col = self.edges[i][0], self.edges[i][1]
        deg = scatter(torch.tensor(self.edge_weights[i]), torch.tensor(col), dim=0, dim_size=num_nodes,
                      reduce='sum')
        deg_out = scatter(torch.tensor(self.edge_weights[i]), torch.tensor(row), dim=0, dim_size=num_nodes,
                      reduce='sum')
        degs.append([deg.to(dtype=torch.float32),deg_out.to(dtype=torch.float32)])
    self.degs = degs
    time_counter.end_single('_calculate_deg')


def _getPartition():
    with open(context.glContext.config['data_path'] + '/nodesPartition.' + context.glContext.config[
        'partitionMethod'] + str(context.glContext.config['worker_num']) + '.txt', 'r') as file:
        # 使用 readlines() 读取文件的所有行并存储在列表中
        partition_array = [np.array(line.strip().split('\t')).astype(int) for line in file.readlines()]

        return partition_array


def _get_nei_unique(edges, target_vertex):
    mask_nei = np.isin(edges[1], target_vertex)
    neis = np.unique(edges[0][mask_nei])
    mask_nei_not_in_target = ~np.isin(neis, target_vertex)
    nei_unique = neis[mask_nei_not_in_target]
    return nei_unique


def _show_cal_distribution(graph, partition):
    edge_degs = workloadAwarePartition.get_edge_degs(graph)
    for i in range(len(partition)):
        print(sum(edge_degs[partition[i]]))


def _get_v_of_each_layer(graph):
    time_counter.start_single('_get_v_of_each_layer')
    if context.glContext.config['partitionMethod'] == 'load_aware':
        time_counter.start_single('load_aware')
        partition = workloadAwarePartition.getPartition(graph)
        time_counter.end_single('load_aware')
    else:
        partition = _getPartition()

    _show_cal_distribution(graph, partition)

    id = context.glContext.config['id']
    layer_num = context.glContext.config['layer_num']
    local_ids = partition[id]
    target_vertex = [np.array([]) for i in range(layer_num + 1)]
    target_vertex[0] = local_ids

    for j in range(layer_num):
        target_vertex[j + 1] = target_vertex[j]
        for i in range(graph.snapshot_count):
            nei_unique = _get_nei_unique(graph.edges[i], target_vertex[j])
            target_vertex[j + 1] = np.union1d(target_vertex[j + 1], nei_unique)

    # for i in range(graph.snapshot_count):
    #     for j in range(layer_num + 1):
    #         graph.target_vertex[i][j] = target_vertex[j]
    graph.target_vertex=target_vertex
    time_counter.end_single('_get_v_of_each_layer')


def _to_local_subgraph(self):
    time_counter.start_single('_to_local_subgraph')
    layer_num = context.glContext.config['layer_num']
    for i in range(self.snapshot_count):
        ids = self.target_vertex[layer_num]
        mask_agg = np.isin(self.edges[i][1], ids)
        self.edges[i] = self.edges[i][:, mask_agg]
        self.edge_weights[i] = self.edge_weights[i][mask_agg]
        self.features[i] = self.features[i][ids, :]
        self.targets[i] = self.targets[i][ids]
        self.degs[i][0] = self.degs[i][0][ids]
        self.degs[i][1] = self.degs[i][1][ids]
    time_counter.end_single('_to_local_subgraph')


def _encode_edge(self):
    time_counter.start_single('_encode_edge')
    for i in range(self.snapshot_count):
        # self.edges[i] = np.array(
        #     [[self.old2new_maps[i].get(value, value) for value in row] for row in self.edges[i]])
        self.edges[i] = np.array(
            [[self.old2new_maps.get(value, value) for value in row] for row in self.edges[i]])
    time_counter.end_single('_encode_edge')


def _encode(self):
    time_counter.start_single('_encode')
    layer_num = context.glContext.config['layer_num']
    dist.barrier()


    old2new_map = {i: j for j, i in enumerate(self.target_vertex[layer_num])}
    # old2new_maps.append(old2new_tmp)
    self.target_vertex = np.vectorize(old2new_map.get)(self.target_vertex[0])
    dist.barrier()  # ensure all workers finished the i-th snapshot
    # for s in range(self.snapshot_count):
    #     old2new_tmp = {i: j for j, i in enumerate(self.target_vertex[s][layer_num])}
    #     old2new_maps.append(old2new_tmp)
    #     self.target_vertex[s] = np.vectorize(old2new_tmp.get)(self.target_vertex[s][0])
    #     dist.barrier()  # ensure all workers finished the i-th snapshot

    if context.glContext.config['partitionMethod']=='load_aware':
        for i in range(workloadAwarePartition.group_size):
            workloadAwarePartition.div_target_vertex[i] = np.array(
                [old2new_map[j] for j in workloadAwarePartition.div_target_vertex[i]])
    dist.barrier()
    # self.old2new_maps = old2new_maps
    self.old2new_maps = old2new_map
    time_counter.end_single('_encode')

# import pandas as pd
# def _edge_to_adj(dataset):
#     adjs=[]
#     for i in range(dataset.snapshot_count):
#         # 将起始节点和结束节点分别存储到两个数组中
#         df = pd.DataFrame(dataset.edges[i].T, columns=['src', 'dest'])
#         adjacency_list = df.groupby('dest')['src'].apply(list).to_dict()
#
#         adjs.append(adjacency_list)
#     return adjs

def start_cache(dataset):
    time_counter.start_single('start_cache')
    _get_target_vertex(dataset)
    _calculate_deg(dataset)
    _get_v_of_each_layer(dataset)
    _to_local_subgraph(dataset)
    _encode(dataset)
    _encode_edge(dataset) # TODO add self-loop at this position
    # _edge_to_adj(dataset)
    time_counter.end_single('start_cache')
