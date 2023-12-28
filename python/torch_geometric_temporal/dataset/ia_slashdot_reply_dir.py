import numpy as np

from adgnn.util_python.timecounter import time_counter
from ..signal import DynamicGraphTemporalSignal
from torch_geometric.utils import scatter
import torch
from torch_geometric.utils import add_remaining_self_loops
from python.torch_geometric_temporal.dataset.data_process.data_cache import start_cache
import pandas



path = '/mnt/data/dataset/ia-slashdot-reply-dir/ia-slashdot-reply-dir.edges'
window = 100
is_weighted = True
delimiter = '\s+'
skip_header = 2


# path='/mnt/data/dataset/soc-flickr-growth/soc-flickr-growth.edges'
# window=300
# is_weighted=True
# delimiter=' '
# skip_header=1

def get_src_tgt_wei_time(data):
    source_vertices = data[:, 0].astype(int)
    target_vertices = data[:, 1].astype(int)
    if data.shape[1] == 3:
        edge_weights = np.ones_like(source_vertices)
        timestamps = data[:, 2].astype(int)
    elif data.shape[1] == 4:
        edge_weights = data[:, 2].astype(int)
        timestamps = data[:, 3].astype(int)
    else:
        print('data dimensions should be 4 or 5')
        exit(-1)

    return source_vertices, target_vertices, edge_weights, timestamps


def get_encoded_src_tgt(source_vertices, target_vertices):
    # encode vertex
    unique_vertices = np.unique(np.concatenate((source_vertices, target_vertices)))
    vertex_num = unique_vertices.size
    vertex_mapping = {old_vertex: new_vertex for new_vertex, old_vertex in enumerate(unique_vertices)}
    source_vertices_mapped = np.vectorize(vertex_mapping.get)(source_vertices)
    target_vertices_mapped = np.vectorize(vertex_mapping.get)(target_vertices)
    return source_vertices_mapped, target_vertices_mapped, vertex_num


class IaSlashdotReplyDirDatasetLoader(object):
    def __init__(self):
        # self.N = N
        self.target_vertex = None
        self.degs = None
        self.old2new_maps = None

        self._read_web_data()

    def get_masked_snapshot(self):
        source_vertices, target_vertices, edge_weights, timestamps = get_src_tgt_wei_time(self._dataset)
        source_vertices_mapped, target_vertices_mapped, vertex_num = get_encoded_src_tgt(source_vertices,
                                                                                         target_vertices)
        # encode time
        unique_timestamps = np.unique(timestamps)
        unique_timestamps = np.sort(unique_timestamps)
        start_id = unique_timestamps[0]
        time_itv = (unique_timestamps[-1] - unique_timestamps[0]) / window

        # snap_mask = [None for i in range(window)]
        edge_snapshots = [None for i in range(window)]
        edge_weight_snapshots = [None for i in range(window)]
        self.N = vertex_num

        for i in range(window):
            time_counter.start_single('processed_window_' + str(i))
            mask = (start_id + i * time_itv <= timestamps) & (timestamps < start_id + (i + 1) * time_itv)
            edge_snapshots[i] = np.array([source_vertices_mapped[mask], target_vertices_mapped[mask]])

            edge_weight_snapshots[i] = np.array(edge_weights[mask])
            # edge_snapshots[i], edge_weight_snapshots[i] = add_remaining_self_loops(
            #     torch.tensor(edge_snapshots[i]), torch.tensor(edge_weight_snapshots[i]), 1., self.N)


            # edge_snapshots[i] = edge_snapshots[i].detach().numpy()
            # edge_weight_snapshots[i] = edge_weight_snapshots[i].detach().numpy()

            time_counter.end_single('processed_window_' + str(i))

        # edge_snapshots = [arr for arr in edge_snapshots if arr.size > 0]

        # edge_weight_snapshots = [arr for arr in edge_weight_snapshots if arr.size > 0]

        self.snapshot_count = len(edge_snapshots)
        self.edge_num = len(source_vertices)
        self.edge_weights = edge_weight_snapshots
        self.edges = edge_snapshots

    def _read_web_data(self):
        # self._dataset = np.loadtxt(path, skiprows=skip_header,dtype=int)
        time_counter.start_single('read_from_disk')
        self._dataset = pandas.read_csv(path, skiprows=skip_header, sep=delimiter).to_numpy()
        time_counter.end_single('read_from_disk')

        time_counter.start_single('get_snapshot_mask')
        self.get_masked_snapshot()
        time_counter.end_single('get_snapshot_mask')

        print('snapshots:{0}, edge_num:{1},vertex_num:{2}'.format(self.snapshot_count, self.edge_num, self.N))

    def _get_features(self):
        features = []
        for i in range(self.snapshot_count):
            num_nodes = self.N
            row, col = self.edges[i][0], self.edges[i][1]
            deg_in = scatter(torch.tensor(self.edge_weights[i]), torch.tensor(col), dim=0, dim_size=num_nodes,
                             reduce='sum')
            deg_out = scatter(torch.tensor(self.edge_weights[i]), torch.tensor(row), dim=0, dim_size=num_nodes,
                              reduce='sum')
            feat = torch.cat((deg_in, deg_out)).unsqueeze(dim=0).view(deg_in.shape[0], 2)
            features.append(feat.float())
        self.features = features

    def _get_targets(self):
        self.targets = []
        for time in range(self.snapshot_count):
            # predict node degrees in advance
            snapshot_id = min(time + 1, self.snapshot_count - 1)
            y = np.array(self.features[snapshot_id][:, 0])
            # logarithmic transformation for node degrees
            y = np.log(y+1)
            self.targets.append(y)

    def get_dataset(self, lags=0) -> DynamicGraphTemporalSignal:
        time_counter.start_single('get_dataset')
        self._get_features()
        self._get_targets()
        start_cache(self)

        dataset = DynamicGraphTemporalSignal(self.edges, self.edge_weights, self.features, self.targets,
                                             self.target_vertex, self.degs, self.old2new_maps)
        time_counter.end_single('get_dataset')
        return dataset
