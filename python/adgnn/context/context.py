
from cmake.build.lib.pb11_ec import *

class Context(object):
    def foo(self):
        pass

    ifctx = True
    config = {
        'worker_address': {},
        'id': -1,
        # 'server_num': 1,
        'worker_num': 2,
        'partitionMethod': 'hash',  # hash,metis
        'layer_num': None,
        'emb_dims': [],
        'iterNum': 500,
        'lr': 0.01,
        'print_result_interval': 10,
        'device': 'cpu',

        # Dynamic Graphs
        'data_path':"/mnt/data/dataset/england_covid", #england_covid
        'feature_dim':8,
        'data_num':129,
        'hidden':[8,8],
        'class_num':1,
        'train_ratio':0.2,
        'batch_size':30,
        'window_size':10,

        # 'data_path':"/mnt/data/dataset/test", #england_covid
        # 'feature_dim':2,
        # 'data_num':6,
        # 'hidden':[4,4],
        # 'class_num':1,
        # 'train_ratio':0.5,
        # 'batch_size':1,
        # 'window_size':1,


    }


    dgnnClient = None
    dgnnClientRouterForCpp = None
    dgnnWorkerRouter = None

    # server 2001 worker 3001 master 4001
    def ipInit(self, workers):
        worker_num = glContext.config['worker_num']
        workers = str.split(workers, ',')
        for i in range(worker_num):
            self.config['worker_address'][i] = workers[i]


    def setWorkerContext(self):
        glContext.dynamicGraphBuild.transBasicDataToCpp(glContext.config['id'], glContext.config['worker_num'])


    def initCluster(self):
        self.dgnnWorkerRouter = []
        self.dgnnClient = DGNNClient()
        self.dgnnClientRouterForCpp = Router()
        self.dynamicGraphBuild=DynamicGraphBuild()
        self.worker_id = self.config['id']
        id = self.config['id']

        self.dgnnClient.serverAddress = self.config['worker_address'][id]
        self.dgnnClient.startClientServer()
        for i in range(self.config['worker_num']):
            self.dgnnWorkerRouter.insert(i, DGNNClient())
            self.dgnnWorkerRouter[i].init_by_address(self.config['worker_address'][i])
        self.dgnnClientRouterForCpp.initWorkerRouter(self.config['worker_address'])



glContext = Context()
