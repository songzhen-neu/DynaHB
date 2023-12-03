# import adgnn.context.context as context
from cmake.build.lib.pb11_ec import *
from adgnn.context import context
import torch.distributed as dist

class Engine:
    def __init__(self, model):
        self.model = model
        self.dgnnClient = None
        print("construct engine")

    def __call__(self):
        self.run()

    def run(self):
        context.glContext.initCluster()
        context.glContext.setWorkerContext()
        self.dgnnClient = context.glContext.dgnnClient

        init_method = "tcp://127.0.0.1:6213"
        dist.init_process_group(backend='gloo', init_method=init_method, rank=context.glContext.config['id'], world_size=context.glContext.config['worker_num'])

        self.run_gnn()



    def run_gnn(self):
        pass

    def train(self):
        context.glContext.is_train = True

    def eval(self):
        context.glContext.is_train = False

    def getAccAvrg(self, split_num_local,split_num_all, acc_train=0, acc_val=0, acc_test=0,is_avrg=False):
        train_num_all_worker = split_num_all[0]
        val_num_all_worker = split_num_all[1]
        test_num_all_worker = split_num_all[2]
        train_num=split_num_local[0]
        val_num=split_num_local[1]
        test_num=split_num_local[2]
        acc_avrg = {}

        if is_avrg:
            acc_entire = context.glContext.dgnnWorkerRouter[0].sendAccuracy(acc_val,
                                                                            acc_train,
                                                                            acc_test)
            acc_avrg['train']=acc_entire['train']/context.glContext.config['worker_num']
            acc_avrg['val']=acc_entire['val']/context.glContext.config['worker_num']
            acc_avrg['test']=acc_entire['test']/context.glContext.config['worker_num']
            return acc_avrg

        acc_entire = context.glContext.dgnnWorkerRouter[0].sendAccuracy(acc_val * val_num,
                                                                        acc_train * train_num,
                                                                        acc_test * test_num)
        context.glContext.dgnnWorkerRouter[0].barrier()

        # for key in acc_entire:
        acc_avrg['train'] = acc_entire['train'] / (float(train_num_all_worker))
        acc_avrg['val'] = acc_entire['val'] / (float(val_num_all_worker))
        acc_avrg['test'] = acc_entire['test'] / (float(test_num_all_worker))
        return acc_avrg
