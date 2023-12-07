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
import adgnn.util_python.adap_batch_size as adap


from torch.nn.parallel import DistributedDataParallel

from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader

from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.dataset import EnglandCovidDatasetLoader
from torch_geometric_temporal.dataset import TestDatasetLoader

from adgnn.util_python.timecounter import time_counter

import torch.distributed as dist
import random
from adgnn.util_python.get_distributed_acc import getAccAvrg

torch.set_printoptions(4)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)



def buildInitGraph():
    loader = None

    if context.glContext.config['data_path'].__contains__('test'):
        loader = TestDatasetLoader()
    elif context.glContext.config['data_path'].__contains__('england_covid'):
        loader = EnglandCovidDatasetLoader()

    dataset = loader.get_dataset(lags=context.glContext.config['feature_dim'])

    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=context.glContext.config['train_ratio'])
    context.glContext.config['snap_num_train']=train_dataset.snapshot_count
    context.glContext.config['data_num_local']=train_dataset.target_vertex[0].shape
    return train_dataset, test_dataset



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



    def forward(self, x, edge, edge_weight, prev_hidden_state, deg):
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
        # buildInitGraph does some initial work, ensure this has been executed before using variables of context
        train_dataset, test_dataset = buildInitGraph()
        min_cost=[100000,0]

        # adap_rl=adap.AdapRLTuner()
        adap_rl=adap.AdapQLTuner()

        model = DistributedDataParallel(self.model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


        test_dataset=test_dataset.construct_testable_data()
        model.train()
        epoch_count=0

        # initial test loss
        if context.glContext.config['is_adap_batch']:
            adap_rl.init_adap(test_dataset,model)

        for epoch in range(context.glContext.config['iterNum']):
            window_id=np.random.randint(0,train_dataset.snapshot_count-context.glContext.config['window_size']+1)
            time_counter.start('batch_time')
            time_counter.start('generate_batch')
            data_batch=train_dataset.generate_batch(window_id)
            time_counter.end('generate_batch')
            cost_train = 0
            hidden_state = [None for i in range(len(context.glContext.config['hidden']))]
            for time, snapshot in enumerate(data_batch):
                y_hat, hidden_state = model(snapshot.x, snapshot.edge, snapshot.edge_weight, hidden_state,snapshot.deg)
                cost_train = cost_train + torch.mean((y_hat.view(-1) - snapshot.y) ** 2)


            cost_train = cost_train / (time + 1)
            cost_train.backward()

            optimizer.step()
            optimizer.zero_grad()
            dist.barrier()
            time_counter.end('batch_time')
            # print('time: {:.4f}'.format(time_counter.time_list['batch_time'][-1]))


            cost_test = 0
            if epoch_count%context.glContext.config['print_result_interval']==0:
                model.eval()
                hidden_state = [None for i in range(len(context.glContext.config['hidden']))]
                for time, snapshot in enumerate(test_dataset):
                    y_hat, hidden_state = model(snapshot.x, snapshot.edge, snapshot.edge_weight, hidden_state,snapshot.deg)
                    cost_test = cost_test + torch.mean((y_hat.view(-1) - snapshot.y) ** 2)
                cost_test = cost_test / (time + 1)

                train_num = data_batch.target_vertex[0][0].shape[0]
                test_num = test_dataset.target_vertex[0][0].shape[0]
                acc_avrg = getAccAvrg([train_num, test_num], [cost_train,cost_test])
                # this function will change context: window_size and batch_size

                time_counter.start('rl_time')
                if context.glContext.config['is_adap_batch']:
                    adap_rl.train(acc_avrg['test'],epoch_count/context.glContext.config['print_result_interval'],time_counter.time_list['batch_time'][-1])
                    adap_rl.distributed_strategy_update()

                time_counter.end('rl_time')
                print('Epoch: {:04d}'.format(epoch + 1),
                      'cost_train: {:.8f}'.format(acc_avrg['train']),
                      'cost_test: {:.8f}'.format(acc_avrg['test']),
                      'time:{:.4f}'.format(time_counter.time_list['batch_time'][-1]))


            if acc_avrg['test']<min_cost[0] and acc_avrg['test']!=0:
                min_cost[0]=acc_avrg['test']
                min_cost[1]=epoch_count
            if epoch_count-min_cost[1]>=20:
                break
            epoch_count+=1
        print(min_cost)
        print("Training End")
        time_counter.printAvrgTime()





if __name__ == "__main__":
    pp.parserInit()
    model = RecurrentGCN(node_features=context.glContext.config['feature_dim'])
    gcn_engine = TGCN_Engine(model)
    gcn_engine()
    print("program end")
