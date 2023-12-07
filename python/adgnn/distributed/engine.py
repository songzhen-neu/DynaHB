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


