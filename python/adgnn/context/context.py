
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
        'ip': "202.199.6.224",
        # 'ip':'192.168.1.1',
        'worker_num': 5,

        'layer_num': None,
        'emb_dims': [],
        'device': 'cuda', # cpu,cuda

        # optimization switch

        'partitionMethod': 'hash',  # hash,metis,load_aware
        'is_adap_batch':True,
        'is_batch_pool':True,
        'dist_mode':'asyn', # asyn, sync

        # 'partitionMethod': 'hash',  # hash,metis,load_aware
        # 'is_adap_batch': False,
        # 'is_batch_pool': False,
        # 'dist_mode': 'sync',  # asyn, sync

        'print_itv':20,
        'time_update': 0.01,
        'iterNum': 3001,


        # Dynamic Graphs
        'data_path': "/mnt/data/dataset/ia-slashdot-reply-dir",  # 51k,140k
        'feature_dim': 2,
        'data_num': 51083,
        'hidden': [16],
        'class_num': 1,
        'train_ratio': 0.4,
        'window_size': 6,  # 6,31,78, batch/train/total
        'batch_size': 510,  # 510,10216,51083, batch/local/total
        'lr': 0.001,
        'preset_cost': 0.4278,

        # 'data_path': "/mnt/data/dataset/soc-bitcoin",  # edge 122948162, vertex_num 24575382
        # 'feature_dim': 2,
        # 'data_num': 24575382,
        # 'hidden': [2], #2
        # 'class_num': 1,
        # 'train_ratio': 0.4,
        # 'window_size': 7,  # 7,36,90
        # 'batch_size': 245735,  # 245735,4915076,24575382
        # 'lr': 0.001,
        # 'preset_cost': 0.1,



        # 'data_path': "/mnt/data/dataset/stackexch",  # edge 122948162, vertex_num 24575382
        # 'feature_dim': 2,
        # 'data_num': 545196,
        # 'hidden': [16],
        # 'class_num': 1,
        # 'train_ratio': 0.4,
        # 'window_size': 13,  # 13,68,170
        # 'batch_size': 5451,  # 5451,109039,545196
        # 'lr': 0.001,
        # 'preset_cost': 0.188,


        # 'data_path': "/mnt/data/dataset/soc-flickr-growth",  # england_covid
        # 'feature_dim': 2,
        # 'data_num': 2302925,
        # 'hidden': [8],
        # 'class_num': 1,
        # 'train_ratio': 0.4,
        # 'window_size': 6,  # 6,28,70
        # 'batch_size': 23029,  # 23029,460585,2302925
        # 'lr': 0.001,
        # 'preset_cost': 0.650,


        # 'data_path': "/mnt/data/dataset/rec-amazon-ratings",  # snapshots:100, edge_num:5838038,vertex_num:2146057
        # 'feature_dim': 2,
        # 'data_num': 2146057,
        # 'hidden': [16],
        # 'class_num': 1,
        # 'train_ratio': 0.4,
        # 'window_size': 5,  # 5,28,70
        # 'batch_size': 21460,  # 21460,429211,2302925
        # 'lr': 0.001,
        # 'preset_cost': 1.190,

        # 'data_path': "/mnt/data/dataset/soc-youtube-growth",  # snapshots:100, edge_num:12223773,vertex_num:3223589
        # 'feature_dim': 2,
        # 'data_num': 3223589,
        # 'hidden': [8],
        # 'class_num': 1,
        # 'train_ratio': 0.4,
        # 'window_size': 6,  # 6,32,80
        # 'batch_size': 32235,  # 32235,644717,3223589
        # 'lr': 0.001,
        # 'preset_cost': 0.229,


        # 'data_path': "/mnt/data/dataset/rec-amz-Books",  # snapshots:100, edge_num:22507154,vertex_num:10356390
        # 'feature_dim': 2,
        # 'data_num': 10356390,
        # 'hidden': [8],
        # 'class_num': 1,
        # 'train_ratio': 0.4,
        # 'window_size': 30,  # 132
        # 'batch_size': 17027,  # 2302925
        # 'lr': 0.001
        # 'iterNum': 301,


        # 'data_path':"/mnt/data/dataset/twitter_tennis", #england_covid
        # 'feature_dim':16,
        # 'data_num':1000,
        # 'hidden':[16],
        # 'class_num':1,
        # 'train_ratio':0.4,
        # 'window_size':16, # 48,-1
        # 'batch_size':128, # 1000,-1
        # 'lr': 0.01
        # 'iterNum': 301,



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
