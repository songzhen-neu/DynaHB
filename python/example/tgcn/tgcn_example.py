try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric_temporal.nn import TGCN
import adgnn.util_python.param_parser as pp
from adgnn.context import context
from adgnn.distributed.engine import Engine


from torch.nn.parallel import DistributedDataParallel

from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader

from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.dataset import EnglandCovidDatasetLoader
from torch_geometric_temporal.dataset import TestDatasetLoader

from adgnn.util_python.timecounter import time_counter

import torch.distributed as dist

torch.set_printoptions(4)
torch.manual_seed(42)
np.random.seed(42)




def calculate_dynamic_edge(edge_indice,edge_weight,snapshot_countt):

    egde_set = []
    egde_new = []
    for i in range(snapshot_countt):
        new_edges ={(x, y) for x, y in zip(edge_indice[i][0], edge_indice[i][1])}
        egde_set.append(new_edges)
    # print("egde_set",egde_set)

    for i in range(1):
    # for i in range(snapshot_countt):
        different_elements = list(egde_set[i].symmetric_difference(egde_set[i+1]))
        egde_new.append(different_elements)
    print("egde_new",egde_new)

    for i in range(snapshot_countt):
        keep_indices = np.ones(len(edge_indice[i][0]), dtype=bool)
        for j in range(len(edge_indice[i][0])):
            if (edge_indice[i][0][j],edge_indice[i][1][j]) in egde_new[0]:
                keep_indices[j] = False
        # print(keep_indices)
        edge_indice[i]=edge_indice[i][:, keep_indices]
        edge_weight[i]=edge_weight[i][keep_indices]



def buildInitGraph():
    loader = None

    if context.glContext.config['data_path'].__contains__('test'):
        loader = TestDatasetLoader()
    elif context.glContext.config['data_path'].__contains__('england_covid'):
        loader = EnglandCovidDatasetLoader()

    dataset = loader.get_dataset(lags=context.glContext.config['feature_dim'])

    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=context.glContext.config['train_ratio'])
    return train_dataset, test_dataset






# class getRmtFeatsFunction(torch.autograd.Function):
#     """Special function for only sparse region backpropataion layer."""
#
#     @staticmethod
#     def forward(ctx, time, input, status,vertex_push,rcv_nei_num):
#         ctx.time = time
#         ctx.status = status
#         ctx.vertex_push=vertex_push
#         ctx.rcv_nei_num=rcv_nei_num
#         context.glContext.dynamicGraphBuild.pushLocalFeats(input)
#         rmt_feats = torch.tensor(context.glContext.dgnnClientRouterForCpp.getRmtFeats(status, time,vertex_push))
#         allfeats = torch.cat([input, rmt_feats], 0)
#         allfeats = torch.FloatTensor(allfeats).to(context.glContext.config['device'])
#         return allfeats
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         time = ctx.time
#         status = ctx.status
#         vertex_push=ctx.vertex_push
#         rcv_nei_num=ctx.rcv_nei_num
#         # print(grad_output)
#         allfeats_grad = context.glContext.dgnnClientRouterForCpp.setAndSendGDynamic(
#             time, status,vertex_push,rcv_nei_num,
#             grad_output.cpu().detach().numpy()
#         )
#
#         # print(allfeats_grad)
#         return None, torch.FloatTensor(allfeats_grad).to(context.glContext.config['device']), None,None,None


# class GetRmtFeats(nn.Module):
#     def forward(self, time, input, status,vertex_push,rcv_nei_num):
#         return getRmtFeatsFunction.apply(time, input, status,vertex_push,rcv_nei_num)


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()


        self.layerNum = len(context.glContext.config['hidden'])
        self.tgcn = {}
        for i in range(1, self.layerNum + 1):
            if i == 1:
                self.tgcn[i] = TGCN(node_features, context.glContext.config['hidden'][0])
            else:
                self.tgcn[i] = TGCN(context.glContext.config['hidden'][i - 2],
                                    context.glContext.config['hidden'][i - 1])
            self.add_module("tgcn_{0}".format(i), self.tgcn[i])
        self.linear = torch.nn.Linear(context.glContext.config['hidden'][self.layerNum - 1], 1)

        # self.getRmtFeats = GetRmtFeats()


    def forward(self, x, edge, edge_weight, prev_hidden_state, time, status, deg):
        hidden = []
        y=x
        for i in range(1, self.layerNum + 1):
            target_num=len(deg[self.layerNum-i])
            h = self.tgcn[i](y, target_num, deg[self.layerNum-i+1], edge[self.layerNum-i], edge_weight[self.layerNum-i], prev_hidden_state[i - 1])
            hidden.append(h)
            y = F.relu(h)

        y = self.linear(y)
        return y, hidden


def setGradients(model):
    with torch.no_grad():
        for id in model.parameters_collection.keys():
            context.glContext.gradients[id] = model.parameters_collection[id].grad / context.glContext.config[
                'worker_num']


optimizer = None




class TGCN_Engine(Engine):
    def run_gnn(self):
        exit_all_loops=False
        min_cost=[100000,0]
        model = DistributedDataParallel(self.model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_dataset, test_dataset = buildInitGraph()
        test_dataset=test_dataset.construct_testable_data()
        data_num = context.glContext.config['data_num']
        model.train()
        epoch_count=0
        for epoch in range(context.glContext.config['iterNum']):
            for i in range(train_dataset.snapshot_count-context.glContext.config['window_size']+1):
                for j in range(int(len(train_dataset.target_vertex[0])/context.glContext.config['batch_size'])):
                    time_counter.start('batch_time')
                    time_counter.start('generate_batch')
                    data_batch=train_dataset.generate_batch(i,j)
                    time_counter.end('generate_batch')
                    cost_train = 0
                    hidden_state = [None for i in range(len(context.glContext.config['hidden']))]
                    for time, snapshot in enumerate(data_batch):
                        y_hat, hidden_state = model(snapshot.x, snapshot.edge, snapshot.edge_weight, hidden_state, time, 'train',snapshot.deg)
                        cost_train = cost_train + torch.mean((y_hat.view(-1) - snapshot.y) ** 2)


                    cost_train = cost_train / (time + 1)
                    cost_train.backward()

                    optimizer.step()
                    optimizer.zero_grad()
                    dist.barrier()
                    time_counter.end('batch_time')

                    epoch_count+=1
                    cost_test = 0
                    if epoch_count%context.glContext.config['print_result_interval']==0:
                        model.eval()
                        hidden_state = [None for i in range(len(context.glContext.config['hidden']))]
                        for time, snapshot in enumerate(test_dataset):
                            y_hat, hidden_state = model(snapshot.x, snapshot.edge, snapshot.edge_weight, hidden_state, time, 'test',snapshot.deg)
                            cost_test = cost_test + torch.mean((y_hat.view(-1) - snapshot.y) ** 2)
                        cost_test = cost_test / (time + 1)

                        # train_num = train_dataset.features[0].shape[0]
                        train_num = data_batch.target_vertex[0][0].shape[0]
                        test_num = test_dataset.target_vertex[0][0].shape[0]
                        acc_avrg = self.getAccAvrg([train_num, 0, test_num], [train_num*context.glContext.config['worker_num'], data_num, test_num*context.glContext.config['worker_num']], cost_train, 0,
                                                   cost_test)
                        print('Epoch: {:04d}'.format(epoch + 1),
                              'cost_train: {:.8f}'.format(acc_avrg['train']),
                              'cost_test: {:.8f}'.format(acc_avrg['test']))
                    else:
                        train_num = data_batch.target_vertex[0][0].shape[0]
                        test_num = test_dataset.target_vertex[0][0].shape[0]
                        acc_avrg = self.getAccAvrg([train_num, 0, test_num], [train_num*context.glContext.config['worker_num'], data_num, test_num*context.glContext.config['worker_num']], cost_train, 0,
                                                   0)
                        print('iteration {0}'.format(epoch_count),
                              'cost_train: {:.8f}'.format(acc_avrg['train']))

                    if cost_test<min_cost[0] and cost_test!=0:
                        min_cost[0]=cost_test
                        min_cost[1]=epoch_count
                    if epoch_count-min_cost[1]>=50:
                        exit_all_loops=True
                        break
                if exit_all_loops:
                    break
            if exit_all_loops:
                break
        print(min_cost)
        print("Training End")
        time_counter.printAvrgTime()


if __name__ == "__main__":
    pp.parserInit()
    model = RecurrentGCN(node_features=context.glContext.config['feature_dim'])
    gcn_engine = TGCN_Engine(model)
    gcn_engine()
    print("program end")
