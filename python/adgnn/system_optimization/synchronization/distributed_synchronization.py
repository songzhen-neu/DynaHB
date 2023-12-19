import time

from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch
import adgnn.context.context as context
from adgnn.util_python.timecounter import time_counter
from adgnn.util_python.get_distributed_acc import getAccAvrg


class DistributedSynchronization:
    model = None

    def __call__(self, model):
        self.set_sync_model(model)
        return self.model

    def set_sync_model(self, model):
        self.model = DistributedDataParallel(model)


SynchronousModel = DistributedSynchronization()


class DistributedAsynchronization:
    model = None
    start_time = None
    time_update = 1
    min_cost = [100000, 0]

    def __call__(self, model):
        self.model = model
        return self.model





    # def test_model(self,epoch,test_dataset, data_batch, cost_train, adap_rl):
    #     cost_test = 0
    #     model = self.model
    #     model.eval()
    #     hidden_state = [None for i in range(len(context.glContext.config['hidden']))]
    #     for time, snapshot in enumerate(test_dataset):
    #         y_hat, hidden_state = model(snapshot.x, snapshot.edge, snapshot.edge_weight, hidden_state,
    #                                     snapshot.deg)
    #         cost_test = cost_test + torch.mean((y_hat.view(-1) - snapshot.y) ** 2)
    #     cost_test = cost_test / (time + 1)
    #
    #     train_num = data_batch.target_vertex[0][0].shape[0]
    #     test_num = test_dataset.target_vertex[0][0].shape[0]
    #     acc_avrg = getAccAvrg([train_num, test_num], [cost_train, cost_test])
    #     # this function will change context: window_size and batch_size
    #
    #     time_counter.start('rl_time')
    #     if context.glContext.config['is_adap_batch']:
    #         adap_rl.train(acc_avrg['test'], epoch / context.glContext.config['print_result_interval'],
    #                       time_counter.time_list['batch_time'][-1])
    #         adap_rl.distributed_strategy_update()
    #
    #     time_counter.end('rl_time')
    #     print('Epoch: {:04d}'.format(epoch + 1),
    #           'cost_train: {:.8f}'.format(acc_avrg['train']),
    #           'cost_test: {:.8f}'.format(acc_avrg['test']),
    #           'time:{:.4f}'.format(time_counter.time_list['batch_time'][-1]))
    #     if acc_avrg['test'] < self.min_cost[0] and acc_avrg['test'] != 0:
    #         self.min_cost[0] = acc_avrg['test']
    #         self.min_cost[1] = epoch
    #     if epoch - self.min_cost[1] >= 10 * context.glContext.config['print_result_interval']:
    #         return acc_avrg,True
    #
    #     return acc_avrg,False




    # def update_model(self, epoch, test_dataset, data_batch, cost_train, adap_rl):
    #     if epoch == 0:
    #         self.start_time = time.time()
    #
    #     end_time = time.time()
    #     if (end_time - self.start_time) / self.time_update >= 1 or epoch==0:
    #         model_module = self.model.module if hasattr(self.model, "module") else self.model
    #         for param in model_module.parameters():
    #             dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
    #             param.data /= dist.get_world_size()
    #
    #         acc_avrg,finish_flag=self.test_model(epoch,test_dataset, data_batch, cost_train, adap_rl)
    #         self.start_time = time.time()
    #         return acc_avrg,finish_flag
    #     return 0,False


AsynchronousModel = DistributedAsynchronization()
