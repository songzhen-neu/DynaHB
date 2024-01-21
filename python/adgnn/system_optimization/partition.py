from torch_geometric.utils import scatter
import adgnn.context.context as context
import torch
import numpy as np


class WorkloadAwarePartition:
    div_target_vertex=None
    group_size=10
    def get_edge_degs(self, graph):
        layer_num = context.glContext.config['layer_num']
        unweighted_degs = []
        for l in range(layer_num):
            unweighted_degs.append([])
            for i in range(graph.snapshot_count):
                unweighted_degs[l].append([])
                num_nodes = context.glContext.config['data_num']
                row, col = graph.edges[i][0], graph.edges[i][1]
                if l == 0:
                    edge_degree = torch.ones_like(torch.tensor(graph.edge_weights[i]))
                else:
                    edge_degree = unweighted_degs[l - 1][i]
                    edge_degree = edge_degree[row]
                edge_degree = scatter(edge_degree, torch.tensor(col), dim=0, dim_size=num_nodes,
                                      reduce='sum')
                unweighted_degs[l][i] = edge_degree.to(dtype=torch.float32)

        edge_degree_sum = []
        for l in range(layer_num):
            edge_degree_sum.append(torch.zeros_like(unweighted_degs[l][0]))
            for i in range(graph.snapshot_count):
                edge_degree_sum[l] += unweighted_degs[l][i]

        edge_degs = torch.zeros_like(edge_degree_sum[0])
        for l in range(layer_num):
            for i in range(l + 1):
                edge_degs += edge_degree_sum[i]

        edge_degs = edge_degs.detach().numpy()
        return edge_degs

    def div_array_by_mod(self, sorted_indices):
        worker_num = context.glContext.config['worker_num']
        indices_mod = [None for i in range(worker_num)]
        # for i in range(worker_num):
        #     indices_mod[i] = sorted_indices[i::worker_num]
        for i in range(worker_num):
            array_a=sorted_indices[i::worker_num*2]
            array_b=sorted_indices[2*worker_num-i-1::worker_num*2]
            min_length=min(len(array_a),len(array_b))
            if len(array_a)>min_length:
                last_elem_a=array_a[-1]
                array_a=array_a[:min_length]
                indices_mod[i]=np.column_stack((array_a,array_b)).reshape(-1)
                indices_mod[i]=np.append(indices_mod[i],last_elem_a)
            else:
                indices_mod[i] = np.column_stack((array_a, array_b)).reshape(-1)
        return indices_mod




    def getPartition(self, graph):
        div_method = 'num_avg'  # val_avg, num_avg
        with torch.no_grad():
            id = context.glContext.config['id']
            edge_degs = self.get_edge_degs(graph)
            sorted_indices = np.argsort(edge_degs)[::-1]
            sorted_deg = edge_degs[sorted_indices]
            indices_mod = self.div_array_by_mod(sorted_indices)

            indices_local = indices_mod[id]
            edge_degs_local = edge_degs[indices_local]

            div_target_v = [None for i in range(self.group_size)]
            if div_method == 'val_avg':
                val_itv = (sorted_deg[0] - sorted_deg[-1]) / self.group_size
                div_target_v[0] = indices_local[edge_degs_local < val_itv]
                div_target_v[-1] = indices_local[edge_degs_local >= 9 * val_itv]
                for i in range(1, self.group_size - 1):
                    div_target_v[i] = indices_local[
                        np.logical_and(edge_degs_local >= i * val_itv, edge_degs_local < (i + 1) * val_itv)]
            elif div_method == 'num_avg':
                num_itv = int(len(edge_degs_local)/self.group_size)
                div_target_v[-1]=indices_local[9*num_itv:]
                for i in range(self.group_size-1):
                    div_target_v[i]=indices_local[i*num_itv:(i+1)*num_itv]

        self.div_target_vertex=div_target_v

        print('get partition end!!')
        return indices_mod


workloadAwarePartition = WorkloadAwarePartition()
