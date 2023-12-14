# import adgnn.context.context as context
# from cmake.build.lib.pb11_ec import *
# from adgnn.context import pb11_dynahb
import threading

from adgnn.context import context
import torch.distributed as dist
import socket, torch
import netifaces as ni


def is_local_ip(ip_address):
    # 判断是否为局域网IP
    return ip_address.startswith(("192.168.", "10.", "172."))


def get_local_ip():
    try:
        # 获取所有网络接口名
        interfaces = ni.interfaces()
        for interface in interfaces:
            # 获取IPv4地址信息
            addresses = ni.ifaddresses(interface).get(ni.AF_INET, [])
            for address_info in addresses:
                ip_address = address_info.get('addr')
                if ip_address and not ip_address.startswith("127."):
                    # 如果是局域网IP，返回
                    if is_local_ip(ip_address):
                        return ip_address

        # 如果没有找到局域网IP，则获取第一个非回环的IPv4地址
        for interface in interfaces:
            addresses = ni.ifaddresses(interface).get(ni.AF_INET, [])
            for address_info in addresses:
                ip_address = address_info.get('addr')
                if ip_address and not ip_address.startswith("127."):
                    return ip_address

        # 如果仍然没有找到，则返回 None
        return None

    except Exception as e:
        print(f"Error getting local IP: {e}")
        return None


class Engine:
    def __init__(self, model):
        self.model = model
        self.dgnnClient = None
        print("construct engine")

    def __call__(self):
        self.run()


    def _set_context_ip_and_port(self):
        id = context.glContext.config['id']
        worker_num = context.glContext.config['worker_num']
        ip_full_template = torch.tensor(list('000.000.000.000:00000'.encode('utf-8')), dtype=torch.uint8)
        local_ip=get_local_ip()
        if id + 1 < 10:
            ip = local_ip + ":" + '200' + str(id + 1)
        else:
            ip = local_ip + ":" + '20' + str(id + 1)
        print('local_id:{0}'.format(local_ip))
        byte_data = ip.encode('utf-8')  # 将字符串转换为字节流
        byte_data = torch.tensor(list(byte_data), dtype=torch.uint8)
        tensor_to_send = torch.zeros(ip_full_template.shape[0], dtype=torch.uint8)
        tensor_to_send[:byte_data.shape[0]] = byte_data
        ip_all_worker = torch.zeros((worker_num, ip_full_template.shape[0]), dtype=torch.uint8)
        if id == 0:
            ip_all_worker[0] = tensor_to_send
        if id != 0:
            dist.send(tensor_to_send, dst=0)
        else:
            for i in range(1, worker_num):
                received_tensor = torch.zeros_like(ip_full_template)
                dist.recv(received_tensor, src=i)
                # ip_all_worker[i]=received_tensor
                ip_all_worker[i] = received_tensor

        dist.barrier()
        if id == 0:
            for i in range(1, worker_num):
                dist.send(ip_all_worker, dst=i)
        else:
            dist.recv(ip_all_worker, src=0)

        dist.barrier()

        for i in range(worker_num):
            received_byte_data = bytes(ip_all_worker[i].tolist())
            received_string = received_byte_data.decode('utf-8').rstrip('\x00')
            context.glContext.config['worker_address'][i] = received_string

        print("get ip end!!")

    def run(self):
        init_method = "tcp://" + context.glContext.config['ip'] + ":7889"
        # init_method = "file:///mnt/data/nfs/sharedfile"
        print(init_method,context.glContext.config['id'],context.glContext.config['worker_num'])

        dist.init_process_group(backend='gloo', init_method=init_method, rank=context.glContext.config['id'],
                                world_size=context.glContext.config['worker_num']) #gloo,mpi,nccl
        # dist.init_process_group(backend='gloo', init_method=init_method, rank=context.glContext.config['id'],
        #                         world_size=context.glContext.config['worker_num']) #gloo
        self._set_context_ip_and_port()
        context.glContext.initCluster()
        context.glContext.setWorkerContext()
        self.dgnnClient = context.glContext.dgnnClient

        self.run_gnn()

    def run_gnn(self):
        pass

    def train(self):
        context.glContext.is_train = True

    def eval(self):
        context.glContext.is_train = False
