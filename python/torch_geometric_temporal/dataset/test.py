import json
import urllib
import numpy as np
from ..signal import DynamicGraphTemporalSignal

from adgnn.context import context
import torch.distributed as dist
import torch
from torch_geometric.utils import scatter


class TestDatasetLoader(object):
    """A dataset of mobility and history of reported cases of COVID-19
    in England NUTS3 regions, from 3 March to 12 of May. The dataset is
    segmented in days and the graph is directed and weighted. The graph
    indicates how many people moved from one region to the other each day,
    based on Facebook Data For Good disease prevention maps.
    The node features correspond to the number of COVID-19 cases
    in the region in the past **window** days. The task is to predict the
    number of cases in each node after 1 day. For details see this paper:
    `"Transfer Graph Neural Networks for Pandemic Forecasting." <https://arxiv.org/abs/2009.08388>`_
    """

    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):
        # url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/england_covid.json"
        # self._dataset = json.loads(urllib.request.urlopen(url).read())
        local_file_path = "/mnt/data/dataset/test/test.json"
        with open(local_file_path, "r") as file:
            self._dataset = json.load(file)

    def _get_edges(self):
        self.edges = []
        for time in range(self._dataset["time_periods"] - self.lags):
            self.edges.append(
                np.array(self._dataset["edge_mapping"]["edge_index"][str(time)]).T
            )

    def _get_edge_weights(self):
        self.edge_weights = []
        for time in range(self._dataset["time_periods"] - self.lags):
            self.edge_weights.append(
                np.array(self._dataset["edge_mapping"]["edge_weight"][str(time)])
            )
        print('get edge weight end!!')

    def _get_targets_and_features(self):

        stacked_target = np.array(self._dataset["y"])
        standardized_target = (stacked_target - np.mean(stacked_target, axis=0)) / (
                np.std(stacked_target, axis=0) + 10 ** -10
        )
        self.features = [
            standardized_target[i: i + self.lags, :].T
            for i in range(self._dataset["time_periods"] - self.lags)
        ]
        self.targets = [
            standardized_target[i + self.lags, :].T
            for i in range(self._dataset["time_periods"] - self.lags)
        ]
        print('_get_targets_and_features end')

    def _get_target_vertex(self):
        self.target_vertex = [
            {} for i in range(self._dataset["time_periods"] - self.lags)
        ]

    def _calculate_deg(self):
        degs = []
        for i in range(self.snapshot_count):
            num_nodes = int(self.edges[i].max()) + 1
            row, col = self.edges[i][0], self.edges[i][1]
            deg = scatter(torch.tensor(self.edge_weights[i]), torch.tensor(col), dim=0, dim_size=num_nodes,
                          reduce='sum')
            degs.append(deg.to(dtype=torch.float32))
        self.degs = degs

    def _getPartition(self):
        with open(context.glContext.config['data_path'] + '/nodesPartition.' + context.glContext.config[
            'partitionMethod'] +str(context.glContext.config['worker_num'])+ '.txt', 'r') as file:
            # 使用 readlines() 读取文件的所有行并存储在列表中
            partition_array = [np.array(line.strip().split('\t')) for line in file.readlines()]
            return partition_array


    def _get_nei_unique(self,edges,target_vertex):
        mask_nei = np.isin(edges[1], target_vertex)
        neis = np.unique(edges[0][mask_nei])
        mask_nei_not_in_target = ~np.isin(neis, target_vertex)
        nei_unique = neis[mask_nei_not_in_target]
        return nei_unique

    def _get_v_of_each_layer(self):
        partition = self._getPartition()
        id = context.glContext.config['id']
        layer_num = context.glContext.config['layer_num']
        local_ids = partition[id].astype(int)
        target_vertex=[np.array([]) for i in range(layer_num+1)]
        target_vertex[0]=local_ids

        for j in range(layer_num):
            target_vertex[j+1]=target_vertex[j]
            for i in range(self.snapshot_count):
                nei_unique=self._get_nei_unique(self.edges[i],target_vertex[j])
                target_vertex[j+1]=np.union1d(target_vertex[j+1],nei_unique)

        for i in range(self.snapshot_count):
            for j in range(layer_num+1):
                self.target_vertex[i][j]=target_vertex[j]
        print('get target vertices of each layer end!!')




    def _to_local_subgraph(self):
        layer_num = context.glContext.config['layer_num']
        for i in range(self.snapshot_count):
            ids = self.target_vertex[i][layer_num]
            mask_agg = np.isin(self.edges[i][1], ids)
            self.edges[i] = self.edges[i][:, mask_agg]
            self.edge_weights[i]=self.edge_weights[i][mask_agg]
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

    def get_dataset(self, lags: int = 8) -> DynamicGraphTemporalSignal:
        """Returning the England COVID19 data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The England Covid dataset.
        """
        self.lags = lags
        self.snapshot_count = self._dataset["time_periods"] - lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        self._get_target_vertex()
        self._calculate_deg()
        self._get_v_of_each_layer()
        self._to_local_subgraph()
        self._encode()
        self._encode_edge()

        dataset = DynamicGraphTemporalSignal(self.edges, self.edge_weights, self.features, self.targets,
                                             self.target_vertex, self.degs, self.old2new_maps)
        return dataset

