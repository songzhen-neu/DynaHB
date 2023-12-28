import time
import numpy as np
from adgnn.context import context
import psutil,os

class TimeCounter():

    def __init__(self):
        self.time_list = {}
        self.start_time = {}
        self.single_time={}


    # def clear_time(self):
    #     for id in context.glContext.time_epoch.keys():
    #         context.glContext.time_epoch[id] = 0

    def start(self, key):
        self.start_time[key] = time.time()

    def end(self, key):
        end = time.time()
        if not self.time_list.__contains__(key):
            self.time_list[key] = []
        self.time_list[key].append(end - self.start_time[key])

    def show(self, key):
        print((key + ' time: {:.4f}').format(self.time_list[key][-1]))

    def printAvrgTime(self,end_epoch):
        for id in self.time_list.keys():
            print('average ' + str(id) + ' time: {:.4f}s'.format(np.array(self.time_list[id][:end_epoch]).mean()))

    def printTotalTime(self):
        for id in self.time_list.keys():
            print('total ' + str(id) + ' time: {:.4f}s'.format(np.array(self.time_list[id]).sum()))

    def getMemory(self):
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return '{:.2f}MB'.format(memory_info.rss / (1024 * 1024))

    def start_single(self,key):
        self.single_time[key]=time.time()
        print(key+' start: {0}'.format(self.getMemory()))

    def end_single(self,key):
        print(key + ' end:{0}, {1:.4f}s'.format(self.getMemory(),time.time()-self.single_time[key]))

time_counter = TimeCounter()
