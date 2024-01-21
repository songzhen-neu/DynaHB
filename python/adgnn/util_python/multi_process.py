# import multiprocessing

import adgnn.context.context as context
import numpy as np
import adgnn.system_optimization.partition as partition
import time

def generate_batch_multiple_process(train_dataset, i, adap_rl,share_list):

    # adap_rl=context.glContext.adap_rl
    actionTrans_and_actionID = adap_rl.get_action_and_trans_all()
    full_vertex_num = context.glContext.config['data_num_local'] * context.glContext.config['snap_num_train']
    action_trans = actionTrans_and_actionID[0][i]
    action_size=adap_rl.get_action_size()
    window_size = action_trans[0]
    vertex_size = action_trans[1]
    action_vertex_num = window_size * vertex_size
    vertex_ratio = max(int(full_vertex_num / action_vertex_num/action_size),1)


    for _ in range(vertex_ratio):
        context.glContext.lock.acquire()
        window_id = np.random.randint(0, train_dataset.snapshot_count - window_size + 1)
        data_batch = train_dataset.get_batch(window_id, window_size, vertex_size)
        # action_id = actionTrans_and_actionID[1][i]
        # adap_rl.batch_pool[action_id].append(data_batch)
        # context.glContext.lock.release()
        share_list.append(data_batch)
        context.glContext.lock.release()


        # print([len(adap_rl.batch_pool[i]) for i in range(len(adap_rl.batch_pool))])
        # print(adap_rl.batch_pool)
        # batch_pool[action_id].append(data_batch)
        # batch_pool_global[action_id].append(data_batch)
    print('sub-process{0},<window_size:{1},vertex_size:{2}> generating end!'.format(i,window_size,vertex_size))
