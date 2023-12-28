try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

import sys, os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_PATH)
sys.path.insert(1, BASE_PATH + '/../')
sys.path.insert(2, BASE_PATH + '/../../')
sys.path.insert(3, BASE_PATH + '/../../../')
os.environ['GLOO_SOCKET_IFNAME'] = 'eno1'

import torch
import numpy as np
import torch.nn.functional as F
from adgnn.context import context
from torch_geometric_temporal.nn import EvolveGCNO
import adgnn.util_python.param_parser as pp

from adgnn.distributed.engine import Engine
import adgnn.util_python.adap_batch_size as adap

from torch.nn.parallel import DistributedDataParallel

from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.dataset import TwitterTennisDatasetLoader

from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.dataset import EnglandCovidDatasetLoader
from torch_geometric_temporal.dataset import TestDatasetLoader

from adgnn.util_python.timecounter import time_counter
from adgnn.system_optimization.batch_generate import batchGenerator

import torch.distributed as dist
import random
from adgnn.util_python.get_distributed_acc import getAccAvrg
import time as tm
from adgnn.system_optimization.synchronization.distributed_synchronization import SynchronousModel, AsynchronousModel

torch.set_printoptions(4)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print(torch.version.cuda)

num_threads = torch.get_num_threads()
print(f"current cores: {num_threads}")
# 设置新的线程数
torch.set_num_threads(int(num_threads / context.glContext.config['worker_num']))

# 打印 PyTorch 版本
print(f"PyTorch version: {torch.__version__}")


def buildInitGraph():
    loader = None

    if context.glContext.config['data_path'].__contains__('test'):
        loader = TestDatasetLoader()
    elif context.glContext.config['data_path'].__contains__('england_covid'):
        loader = EnglandCovidDatasetLoader()
    elif context.glContext.config['data_path'].__contains__('twitter_tennis'):
        loader = TwitterTennisDatasetLoader()

    dataset = loader.get_dataset(lags=context.glContext.config['feature_dim'])

    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=context.glContext.config['train_ratio'])
    context.glContext.config['snap_num_train'] = train_dataset.snapshot_count
    context.glContext.config['data_num_local'] = len(train_dataset.target_vertex[0])
    return train_dataset, test_dataset


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.layerNum = len(context.glContext.config['hidden'])
        self.recurrent = EvolveGCNO(node_features)
        self.linear = torch.nn.Linear(node_features, 1)

    def forward(self,  x, edge, edge_weight, prev_hidden_state, deg):
        target_num = len(deg[0])
        h = self.recurrent(x, target_num,deg[1],edge[0], edge_weight[0])
        #             h = self.tgcn[i](y, target_num, deg[self.layerNum - i + 1], edge[self.layerNum - i],
        #                              edge_weight[self.layerNum - i], prev_hidden_state[i - 1])
        h = F.relu(h)
        h = self.linear(h)
        return h,None





# class RecurrentGCN(torch.nn.Module):
#     def __init__(self, node_features):
#         super(RecurrentGCN, self).__init__()
#
#         self.layerNum = len(context.glContext.config['hidden'])
#         self.tgcn = {}
#         for i in range(1, self.layerNum + 1):
#             if i == 1:
#                 self.tgcn[i] = TGCN(node_features, context.glContext.config['hidden'][0])
#             else:
#                 self.tgcn[i] = TGCN(context.glContext.config['hidden'][i - 2],
#                                     context.glContext.config['hidden'][i - 1])
#             self.add_module("tgcn_{0}".format(i), self.tgcn[i])
#         self.linear = torch.nn.Linear(context.glContext.config['hidden'][self.layerNum - 1], 1)
#
#     def forward(self, x, edge, edge_weight, prev_hidden_state, deg):
#         hidden = []
#         y = x
#         for i in range(1, self.layerNum + 1):
#             target_num = len(deg[self.layerNum - i])
#             h = self.tgcn[i](y, target_num, deg[self.layerNum - i + 1], edge[self.layerNum - i],
#                              edge_weight[self.layerNum - i], prev_hidden_state[i - 1])
#             hidden.append(h)
#             y = F.relu(h)
#
#         y = self.linear(y)
#         return y, hidden


def test_model(model, test_dataset):
    cost_test = 0
    model.eval()
    # model.train()
    hidden_state = [None for i in range(len(context.glContext.config['hidden']))]
    for time, snapshot in enumerate(test_dataset):
        y_hat, hidden_state = model(snapshot.x, snapshot.edge, snapshot.edge_weight, hidden_state,
                                    snapshot.deg)
        cost_test = cost_test + torch.mean((y_hat.view(-1) - snapshot.y) ** 2)
    cost_test = cost_test / (time + 1)

    test_num = test_dataset.target_vertex[0][0].shape[0]
    return test_num, cost_test


def avrg_model(model):
    model_module = model.module if hasattr(model, "module") else model
    for param in model_module.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= dist.get_world_size()


optimizer = None


class TGCN_Engine(Engine):
    def run_gnn(self):
        global acc_avrg
        print(context.glContext.config)
        if torch.cuda.is_available():
            print("GPU available:")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            context.glContext.config['device'] = 'cpu'
            print("cannot find any GPU.")

        # buildInitGraph does some initial work, ensure this has been executed before using variables of context
        train_dataset, test_dataset = buildInitGraph()
        min_cost = [100000, 0, 0]

        adap_rl = adap.AdapRLTuner()
        # adap_rl = adap.AdapQLTuner()

        self.model = self.model.to(context.glContext.config['device'])
        if context.glContext.config['dist_mode'] == 'sync':
            model = SynchronousModel(self.model)
        elif context.glContext.config['dist_mode'] == 'asyn':
            model = AsynchronousModel(self.model)

        optimizer = torch.optim.Adam(model.parameters(), lr=context.glContext.config['lr'])

        test_dataset = test_dataset.construct_testable_data()
        if context.glContext.config['window_size'] == -1:
            train_dataset = train_dataset.construct_testable_data()
            train_dataset.to_device(context.glContext.config['device'])


        epoch_count = 0

        test_dataset.to_device(context.glContext.config['device'])
        # initial test loss
        if context.glContext.config['is_adap_batch']:
            adap_rl.init_adap(test_dataset, model)

        for epoch in range(context.glContext.config['iterNum']):
            model.train()
            time_counter.start('batch_time')
            data_batch = batchGenerator.generate_batch(adap_rl, train_dataset)
            cost_train = 0
            hidden_state = [None for i in range(len(context.glContext.config['hidden']))]
            time_counter.start('forward')
            for time, snapshot in enumerate(data_batch):
                y_hat, hidden_state = model(snapshot.x, snapshot.edge, snapshot.edge_weight, hidden_state, snapshot.deg)
                cost_train = cost_train + torch.mean((y_hat.view(-1) - snapshot.y) ** 2)
            cost_train = cost_train / (time + 1)
            time_counter.end('forward')

            time_counter.start('backward')
            cost_train.backward(retain_graph=True)
            time_counter.end('backward')

            time_counter.start('update_param')
            optimizer.step()
            optimizer.zero_grad()
            time_counter.end('update_param')
            time_counter.end('batch_time')

            if epoch_count == 0:
                AsynchronousModel.start_time = tm.time()

            if context.glContext.config['dist_mode'] == 'sync':
                dist.barrier()
            if (tm.time() - AsynchronousModel.start_time) / AsynchronousModel.time_update >= 1 or epoch_count == 0:

                if context.glContext.config['dist_mode'] == 'asyn':
                    avrg_model(model)
                test_num, cost_test = test_model(model, test_dataset)
                train_num = data_batch.target_vertex[0][0].shape[0]

                acc_avrg = getAccAvrg([train_num, test_num], [cost_train, cost_test])

                time_counter.start('rl_time')
                if context.glContext.config['is_adap_batch']:
                    adap_rl.train(acc_avrg['test'], epoch_count,
                                  time_counter.time_list['batch_time'][-1])
                    adap_rl.distributed_strategy_update()
                time_counter.end('rl_time')

                print('Epoch: {:04d}'.format(epoch + 1),
                      'cost_train: {:.8f}'.format(acc_avrg['train']),
                      'cost_test: {:.8f}'.format(acc_avrg['test']),
                      'time:{:.4f}'.format(time_counter.time_list['batch_time'][-1]))
                if acc_avrg['test'] < min_cost[0] and acc_avrg['test'] != 0:
                    min_cost[0] = acc_avrg['test']
                    min_cost[1] = epoch
                    min_cost[2] = epoch_count
                if epoch_count - min_cost[2] >= 10:
                    break
                AsynchronousModel.start_time = tm.time()
                epoch_count += 1

        print(min_cost)
        print('from pool:{0}, new generate:{1}'.format(batchGenerator.batch_choose_ratio[0],
                                                       batchGenerator.batch_choose_ratio[1]))
        print("Training End")
        time_counter.printAvrgTime(min_cost[1])


if __name__ == "__main__":
    pp.parserInit()
    model = RecurrentGCN(node_features=context.glContext.config['feature_dim'])
    gcn_engine = TGCN_Engine(model)
    gcn_engine()
    print("program end")
