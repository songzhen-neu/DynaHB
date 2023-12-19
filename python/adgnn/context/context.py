
# from cmake.build.lib.pb11_ec import *
from python.pb11_ec import *
# import ctypes,os
# BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# pb11_dynahb = ctypes.CDLL(BASE_PATH+'/../../cmake/build/lib/pb11_ec.cpython-39-x86_64-linux-gnu.so')

class Context(object):

    ifctx = True
    config = {
        'worker_address': {},
        'id': -1,
        # 'ip':"127.0.0.1",
        # 'ip': "202.199.6.224",
        'ip':'192.168.1.1',
        'worker_num': 3,
        'partitionMethod': 'load_aware',  # hash,metis,load_aware
        'layer_num': None,
        'emb_dims': [],
        'iterNum': 2001,
        # 'lr': 0.01,
        'print_result_interval': 10,
        'device': 'cuda', # cpu,cuda

        # optimization switch
        'is_adap_batch':True,
        'is_batch_pool':True,
        'dist_mode':'asyn', # asyn, sync


        # Dynamic Graphs
        'data_path':"/mnt/data/dataset/twitter_tennis", #england_covid
        'feature_dim':16,
        'data_num':1000,
        'hidden':[16,16],
        'class_num':1,
        'train_ratio':0.4,
        'window_size':16, # 48,-1
        'batch_size':128, # 1000,-1
        'lr': 0.01
        # 'data_path':"/mnt/data/dataset/england_covid", #england_covid
        # 'feature_dim':8,
        # 'data_num':129,
        # 'hidden':[8,8],
        # 'class_num':1,
        # 'train_ratio':0.2,
        # 'window_size':10, # 4,10
        # 'batch_size':64, # 32,64
        # 'lr': 0.001,


        # 'data_path':"/mnt/data/dataset/test", #england_covid
        # 'feature_dim':2,
        # 'data_num':6,
        # 'hidden':[4,4],
        # 'class_num':1,
        # 'train_ratio':0.5,
        # 'window_size':1,
        # 'batch_size':1,
        # 'lr': 0.01,



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
