import adgnn.context.context as context
from adgnn.util_python.timecounter import time_counter
import numpy as np



class BatchGenerator:
    batch_choose_ratio = [0, 0]

    _instance = None

    def __new__(self):
        if not self._instance:
            self._instance = super(BatchGenerator, self).__new__(self)
        return self._instance

    def generate_batch(self, adap_rl, train_dataset):
        window_id = np.random.randint(0, train_dataset.snapshot_count - context.glContext.config['window_size'] + 1)
        if context.glContext.config['window_size'] != -1:
            time_counter.start('generate_batch')
            action = adap_rl.get_action()
            action_trans = adap_rl.get_action_trans()
            action_num_in_pool = len(adap_rl.batch_pool[action])

            full_vertex_num = context.glContext.config['data_num_local'] * context.glContext.config[
                'snap_num_train']
            action_vertex_num = action_trans[0] * action_trans[1]
            vertex_ratio = max(int(full_vertex_num / action_vertex_num/adap_rl.get_action_size()),1)
            if context.glContext.config['is_batch_pool']:
                if np.random.rand() < action_num_in_pool / vertex_ratio:
                    index = np.random.randint(0, action_num_in_pool)
                    data_batch = adap_rl.batch_pool[action][index]
                    self.batch_choose_ratio[0] += 1
                else:
                    context.glContext.config['batch_size']=-1
                    data_batch = context.glContext.train_dataset_full.get_batch(window_id)
                    # adap_rl.batch_pool[action].append(data_batch)
                    self.batch_choose_ratio[1] += 1
            else:
                data_batch = train_dataset.get_batch(window_id)
            time_counter.end('generate_batch')
            time_counter.start('batch_to_device')
            data_batch.to_device(context.glContext.config['device'])
            time_counter.end('batch_to_device')
        else:
            data_batch = train_dataset

        return data_batch


batchGenerator = BatchGenerator()
