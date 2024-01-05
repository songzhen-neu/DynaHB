import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import adgnn.context.context as context
from adgnn.util_python.get_distributed_acc import getAccAvrg
import torch.distributed as dist


# # 定义简单的 Q 网络
# class _QNetwork(nn.Module):
#     def __init__(self, state_size, action_size):
#         super(_QNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_size, 16)
#         # self.fc2 = nn.Linear(16, 16)
#         self.fc2 = nn.Linear(16, action_size)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         # x = torch.relu(self.fc2(x))
#         x = self.fc2(x)
#         return x
#
#
# # 定义经验回放缓冲区
# class _ReplayBuffer:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.buffer = []
#         self.position = 0
#
#     def push(self, transition):
#         if len(self.buffer) < self.capacity:
#             self.buffer.append(None)
#         self.buffer[self.position] = transition
#         self.position = (self.position + 1) % self.capacity
#
#     def sample(self, batch_size):
#         return random.sample(self.buffer, batch_size)
#
#
# # 定义 DQN 算法
# class _DQN:
#     def __init__(self, state_size, action_size, gamma=0.99, epsilon=1, epsilon_decay=0.98, min_epsilon=0.1):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_decay = epsilon_decay
#         self.min_epsilon = min_epsilon
#
#         # 初始化 Q 网络和目标网络
#         self.q_network = _QNetwork(state_size, action_size)
#         self.target_network = _QNetwork(state_size, action_size)
#         self.target_network.load_state_dict(self.q_network.state_dict())
#         self.target_network.eval()
#
#         # 定义优化器和损失函数
#         self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
#         self.criterion = nn.MSELoss()
#
#         # 初始化经验回放缓冲区
#         self.replay_buffer = _ReplayBuffer(capacity=1000)
#
#     def select_action(self, state):
#         if np.random.rand() < self.epsilon:
#             return np.random.randint(0, self.action_size)
#         else:
#             with torch.no_grad():
#                 q_values = self.q_network(torch.tensor(state).float())
#                 return torch.argmax(q_values).item()
#
#     def train(self, batch_size):
#         if len(self.replay_buffer.buffer) < batch_size:
#             return
#
#         transitions = self.replay_buffer.sample(batch_size)
#         batch = list(zip(*transitions))
#
#         state_batch = torch.tensor(np.array(batch[0])).float()
#         action_batch = torch.tensor(np.array(batch[1])).long()
#         reward_batch = torch.tensor(np.array(batch[2])).float()
#         next_state_batch = torch.tensor(np.array(batch[3])).float()
#         done_batch = torch.tensor(np.array(batch[4])).float()
#
#         # 计算目标 Q 值
#         with torch.no_grad():
#             target_q_values_next = self.target_network(next_state_batch)
#             target_q_values = reward_batch + (1 - done_batch) * self.gamma * target_q_values_next.max(1).values
#
#         # 计算估计 Q 值
#         q_values = self.q_network(state_batch)
#         q_values = q_values.gather(1, action_batch.unsqueeze(1))
#
#         # 计算损失
#         loss = self.criterion(q_values, target_q_values.unsqueeze(1))
#
#         # 反向传播和优化
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#         # 更新目标网络参数
#         self.soft_update_target_network()
#
#         # 衰减探索率
#         self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
#
#     def soft_update_target_network(self, tau=0.01):
#         for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
#             target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
#
#
# class AdapRLTuner:
#     def __init__(self):
#         # self.__window_size_set = [i for i in
#         #                           range(1, context.glContext.config['snap_num_train'])]  # [1,2,3,4,5,6,7,8,9,10]
#         self.__window_size_set = [pow(2, i) for i in range(0, int(
#             np.log2(context.glContext.config['snap_num_train'])) + 1)]  # [1,2,4,8]
#         self.__vertex_size_set = [pow(2, i) for i in range(0, int(
#             np.log2(context.glContext.config['data_num_local'])) + 1)]  # [1,2,4,8,16,32,64,128,256,512,1024,2048]
#         self.__window_action_set = [1 / 2, 1, 2]
#         self.__vertex_action_set = [1 / 2, 1, 2]
#         self.__state_size = 3  # 状态空间的维度
#         # self.__action_size = len(self.__window_size_set)*len(self.__vertex_size_set)  # 动作空间的维度
#         self.__action_size = 9  # 动作空间的维度
#         self.__dqn_agent = _DQN(self.__state_size, self.__action_size)
#         self.__total_reward = 0
#         self.__state = np.array([0])
#         self.__loss_lst = 0
#         self.__delta_loss_lst = 0
#         self.__action_trans_lst = [0, 0]
#         self.__action_trans = None
#         self.__action_lst = 0
#         self.__avg_loss = 0
#         self.__time_lst = 0
#         self.__action = 0
#         self.__action_description = None
#         self.__base_loss_time = [0, 0]
#
#     def init_adap(self, test_dataset, model):
#         model.eval()
#         hidden_state = [None for i in range(len(context.glContext.config['hidden']))]
#         cost_test = 0
#         for time, snapshot in enumerate(test_dataset):
#             y_hat, hidden_state = model(snapshot.x, snapshot.edge, snapshot.edge_weight, hidden_state,
#                                         snapshot.deg)
#             cost_test = cost_test + torch.mean((y_hat.view(-1) - snapshot.y) ** 2)
#         cost_test = cost_test / (time + 1)
#
#         test_num = test_dataset.target_vertex[0][0].shape[0]
#         acc_avrg = getAccAvrg([1, test_num], [1, cost_test])
#         self.__loss_lst = acc_avrg['test']
#
#     def trans_to_size(self, action_div):
#         window_size = int(context.glContext.config['window_size'] * action_div[0])
#         batch_size = int(context.glContext.config['batch_size'] * action_div[1])
#         if not self.__window_size_set.__contains__(window_size):
#             window_size = context.glContext.config['window_size']
#         if not self.__vertex_size_set.__contains__(batch_size):
#             batch_size = context.glContext.config['batch_size']
#         return [window_size, batch_size]
#
#     def trans_to_description(self, action_div):
#         action_one = None
#         action_two = None
#         if action_div[0] == self.__window_action_set[0]:
#             action_one = 'descent'
#         elif action_div[0] == self.__window_action_set[1]:
#             action_one = 'keep'
#         else:
#             action_one = 'increase'
#
#         if action_div[1] == self.__vertex_action_set[0]:
#             action_two = 'descent'
#         elif action_div[1] == self.__vertex_action_set[1]:
#             action_two = 'keep'
#         else:
#             action_two = 'increase'
#         return [action_one, action_two]
#
#     def distributed_strategy_update(self):
#         # 发送参数到其他设备
#         with torch.no_grad():
#             if dist.get_rank() == 0:
#                 # 发送参数到rank为1的设备
#                 data_to_send = torch.tensor(
#                     np.array([context.glContext.config['window_size'], context.glContext.config['batch_size']]))
#                 for i in range(1, context.glContext.config['worker_num']):
#                     dist.send(tensor=data_to_send, dst=i)
#             else:
#                 # 接收参数
#                 received_data = torch.zeros(2, dtype=torch.int64)
#                 dist.recv(tensor=received_data, src=0)
#                 received_data = received_data.detach().numpy()
#                 context.glContext.config['window_size'] = received_data[0]
#                 context.glContext.config['batch_size'] = received_data[1]
#
#     def clean_time_reward(self, time_reward):
#         # if self.__action_trans_lst[0]<= self.__action_trans[0] and self.__action_trans_lst[1]<= self.__action_trans[1] and time_reward>0:
#         #     return 0
#         if self.__action_trans_lst[0] >= self.__action_trans[0] and self.__action_trans_lst[1] >= self.__action_trans[
#             1] and time_reward < 0:
#             return 0
#         return time_reward
#
#
#
#     def train(self, loss, step, time):
#         if context.glContext.config['id'] != 0:
#             return
#         if step == 0:
#             self.__delta_loss_lst = self.__loss_lst - loss
#             self.__loss_lst = loss
#             self.__time_lst = time
#             self.__action_trans_lst = [context.glContext.config['window_size'], context.glContext.config['batch_size']]
#             self.__action_lst = 0
#             self.__state = np.array([self.__delta_loss_lst,self.__action_trans_lst[0],self.__action_trans_lst[1]])
#             self.__action_trans = [context.glContext.config['window_size'], context.glContext.config['batch_size']]
#             self.__action_description = ['keep', 'keep']
#             self.__base_loss_time[0] = self.__delta_loss_lst
#             self.__base_loss_time[1] = time
#             return
#
#         delta_loss = self.__loss_lst - loss
#         # loss_reward=delta_loss/self.__loss_lst
#         loss_reward= (delta_loss-self.__delta_loss_lst)/delta_loss
#         # loss_reward = (delta_loss - self.__base_loss_time[0]) / delta_loss
#
#         time_reward_tmp=(self.__time_lst-time)/self.__time_lst
#         # time_reward_tmp = (self.__base_loss_time[1] - time) / self.__base_loss_time[1]
#         time_reward = self.clean_time_reward(time_reward_tmp)
#
#
#         # if loss_reward<=0:
#         #     time_reward=0
#         #     loss_reward*=100
#         # loss_reward*=alpha
#
#         reward = loss_reward + time_reward
#         next_state = np.array([delta_loss,self.__action_trans[0],self.__action_trans[1]])
#
#         done = False
#         # 存储经验到经验回放缓冲区
#         self.__dqn_agent.replay_buffer.push((self.__state, self.__action, reward, next_state, done))
#
#         # 进行训练
#         self.__dqn_agent.train(batch_size=5)
#         self.__total_reward += reward
#
#         print(
#             f"Episode: {step + 1}, Reward: {reward}, Action_Description: {self.__action_description}, Last Action:{self.__action_trans_lst},Action_Trans:{self.__action_trans}, Action: {self.__action}")
#
#         # save needed old data
#         self.__state = next_state
#         self.__delta_loss_lst = delta_loss
#         self.__loss_lst = loss
#         self.__time_lst = time
#         self.__action_lst = self.__action
#         self.__action_trans_lst = self.__action_trans
#
#         # select a new action
#         self.__action = self.__dqn_agent.select_action(self.__state)
#         action_div = [self.__window_action_set[int(self.__action / len(self.__vertex_action_set))],
#                       self.__vertex_action_set[self.__action % len(self.__vertex_action_set)]]
#         self.__action_trans = self.trans_to_size(action_div)
#         context.glContext.config['window_size'] = self.__action_trans[0]
#         context.glContext.config['batch_size'] = self.__action_trans[1]
#         self.__action_description = self.trans_to_description(action_div)

##############################################################################


# 定义简单的 Q 网络
class _QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(_QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        # self.fc2 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = self.fc2(x)
        return x


# 定义经验回放缓冲区
class _ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


# 定义 DQN 算法
class _DQN:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1, epsilon_decay=0.95, min_epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # 初始化 Q 网络和目标网络
        self.q_network = _QNetwork(state_size, action_size)
        self.target_network = _QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # 定义优化器和损失函数
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.005)
        self.criterion = nn.MSELoss()

        # 初始化经验回放缓冲区
        self.replay_buffer = _ReplayBuffer(capacity=1000)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.tensor(state).float())
                return torch.argmax(q_values).item()

    def train(self, batch_size):
        if len(self.replay_buffer.buffer) < batch_size:
            return

        transitions = self.replay_buffer.sample(batch_size)
        batch = list(zip(*transitions))

        state_batch = torch.tensor(np.array(batch[0])).float()
        action_batch = torch.tensor(np.array(batch[1])).long()
        reward_batch = torch.tensor(np.array(batch[2])).float()
        next_state_batch = torch.tensor(np.array(batch[3])).float()
        done_batch = torch.tensor(np.array(batch[4])).float()

        # 计算目标 Q 值
        with torch.no_grad():
            target_q_values_next = self.target_network(next_state_batch)
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * target_q_values_next.max(1).values

        # 计算估计 Q 值
        q_values = self.q_network(state_batch)
        q_values = q_values.gather(1, action_batch.unsqueeze(1))

        # 计算损失
        loss = self.criterion(q_values, target_q_values.unsqueeze(1))

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络参数
        self.soft_update_target_network()

        # 衰减探索率
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def soft_update_target_network(self, tau=0.01):
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class AdapRLTuner:
    def __init__(self):
        # self.__window_size_set = [i for i in
        #                           range(1, context.glContext.config['snap_num_train'])]  # [1,2,3,4,5,6,7,8,9,10]
        # self.__window_size_set = [pow(2, i) for i in range(0, int(
        #     np.log2(context.glContext.config['snap_num_train'])) + 1)]  # [1,2,4,8]
        # self.__window_size_set = [2, 8, 16, 32]
        # self.__vertex_size_set = [32, 256, 1024]
        data_num=context.glContext.config['data_num']
        snap_num=context.glContext.config['snap_num_train']
        self.__window_size_set = [int(0.1*snap_num),int(0.2*snap_num), int(0.3*snap_num)]
        self.__vertex_size_set = [int(0.005*data_num),int(0.01*data_num), int(0.05*data_num), int(0.1*data_num)]
        # self.__vertex_size_set = [pow(2, i) for i in range(0, int(
        #     np.log2(context.glContext.config['data_num_local'])) + 1)]  # [1,2,4,8,16,32,64,128,256,512,1024,2048]
        # self.__vertex_size_set.append(int(context.glContext.config['data_num']/context.glContext.config['worker_num']))
        self.__state_size = 1  # 状态空间的维度
        self.__action_size = len(self.__window_size_set) * len(self.__vertex_size_set)  # 动作空间的维度
        self.__dqn_agent = _DQN(self.__state_size, self.__action_size)
        self.__total_reward = 0
        self.__state = np.array([0])
        self.__loss_lst = 0
        self.__delta_loss_lst = 0
        self.__action_trans_lst = [0, 0]
        self.__action_trans = [context.glContext.config['window_size'], context.glContext.config['batch_size']]
        self.__action_lst = 0
        self.__avg_loss = 0
        self.__time_lst = 0
        self.__action = 0
        self.__action_description = None
        self.__base_loss_time = [0, 0]
        self.batch_pool = {i: [] for i in range(self.__action_size)}

    def get_action(self):
        return self.__action

    def get_action_trans(self):
        return self.__action_trans

    def init_adap(self, test_dataset, model):
        with torch.no_grad():
            model.to('cpu')
            hidden_state = [None for i in range(len(context.glContext.config['hidden']))]
            cost_test = 0
            for time, snapshot in enumerate(test_dataset):
                y_hat, hidden_state = model(snapshot.x, snapshot.edge, snapshot.edge_weight, hidden_state,
                                            snapshot.deg)
                cost_test = cost_test + torch.mean((y_hat.view(-1) - snapshot.y) ** 2)
            cost_test = cost_test / (time + 1)

            test_num = test_dataset.target_vertex[0][0].shape[0]
            acc_avrg = getAccAvrg([1, test_num], [1, cost_test])
            self.__loss_lst = acc_avrg['test']
            model.to(context.glContext.config['device'])


    # def trans_to_size(self, action_div):
    #     window_size = int(context.glContext.config['window_size'] * action_div[0])
    #     batch_size = int(context.glContext.config['batch_size'] * action_div[1])
    #     if not self.__window_size_set.__contains__(window_size):
    #         window_size = context.glContext.config['window_size']
    #     if not self.__vertex_size_set.__contains__(batch_size):
    #         batch_size = context.glContext.config['batch_size']
    #     return [window_size, batch_size]
    #
    # def trans_to_description(self, action_div):
    #     action_one = None
    #     action_two = None
    #     if action_div[0] == self.__window_action_set[0]:
    #         action_one = 'descent'
    #     elif action_div[0] == self.__window_action_set[1]:
    #         action_one = 'keep'
    #     else:
    #         action_one = 'increase'
    #
    #     if action_div[1] == self.__vertex_action_set[0]:
    #         action_two = 'descent'
    #     elif action_div[1] == self.__vertex_action_set[1]:
    #         action_two = 'keep'
    #     else:
    #         action_two = 'increase'
    #     return [action_one, action_two]

    def distributed_strategy_update(self):
        # 发送参数到其他设备
        with torch.no_grad():
            if dist.get_rank() == 0:
                # 发送参数到rank为1的设备
                data_to_send = torch.tensor(
                    np.array([context.glContext.config['window_size'], context.glContext.config['batch_size']]))
                for i in range(1, context.glContext.config['worker_num']):
                    dist.send(tensor=data_to_send, dst=i)
            else:
                # 接收参数
                received_data = torch.zeros(2, dtype=torch.int64)
                dist.recv(tensor=received_data, src=0)
                received_data = received_data.detach().numpy()
                context.glContext.config['window_size'] = received_data[0]
                context.glContext.config['batch_size'] = received_data[1]

    def __clean_time_reward(self, time_reward):
        # if self.__action_trans_lst[0]<= self.__action_trans[0] and self.__action_trans_lst[1]<= self.__action_trans[1] and time_reward>0:
        #     return 0
        if self.__action_trans_lst[0] >= self.__action_trans[0] and self.__action_trans_lst[1] >= self.__action_trans[
            1] and time_reward < 0:
            return 0
        return time_reward

    def __clean_loss_reward(self, loss_reward):
        if self.__action_trans_lst[0] <= self.__action_trans[0] and self.__action_trans_lst[1] <= self.__action_trans[
            1] and loss_reward < 0:
            return 0
        return loss_reward

    def train(self, loss, step, time):
        if context.glContext.config['id'] != 0:
            return
        if step == 0:
            self.__delta_loss_lst = self.__loss_lst - loss
            self.__loss_lst = loss
            self.__time_lst = time
            self.__action_trans_lst = [context.glContext.config['window_size'], context.glContext.config['batch_size']]
            self.__action_lst = 0
            self.__state = np.array([self.__delta_loss_lst])
            self.__action_trans = [context.glContext.config['window_size'], context.glContext.config['batch_size']]
            self.__base_loss_time[0] = self.__delta_loss_lst
            self.__base_loss_time[1] = time
            return

        delta_loss = self.__loss_lst - loss
        loss_reward = (delta_loss - self.__delta_loss_lst) / delta_loss
        # loss_reward = delta_loss / self.__loss_lst
        loss_reward = self.__clean_loss_reward(loss_reward)

        time_reward_tmp = (self.__time_lst - time) / self.__time_lst
        time_reward = self.__clean_time_reward(time_reward_tmp)

        reward = loss_reward + time_reward
        # reward=loss_reward
        if delta_loss <= 0:
            reward = -np.abs(loss_reward) - np.abs(time_reward)
        next_state = np.array([delta_loss])

        done = False
        # 存储经验到经验回放缓冲区
        self.__dqn_agent.replay_buffer.push((self.__state, self.__action, reward, next_state, done))

        # 进行训练
        self.__dqn_agent.train(batch_size=5)
        self.__total_reward += reward

        print(
            f"Episode: {step + 1}, Reward: {reward}, Last Action:{self.__action_trans_lst},Action_Trans:{self.__action_trans}, Action: {self.__action}")

        # save needed old data
        self.__state = next_state
        self.__delta_loss_lst = delta_loss
        self.__loss_lst = loss
        self.__time_lst = time
        self.__action_lst = self.__action
        self.__action_trans_lst = self.__action_trans

        # select a new action
        self.__action = self.__dqn_agent.select_action(self.__state)
        self.__action_trans = [self.__window_size_set[int(self.__action / len(self.__vertex_size_set))],
                               self.__vertex_size_set[self.__action % len(self.__vertex_size_set)]]
        context.glContext.config['window_size'] = self.__action_trans[0]
        context.glContext.config['batch_size'] = self.__action_trans[1]


##############################################################################

class _QLearningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = torch.zeros((state_space, action_space))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_space))
        else:
            return torch.argmax(self.q_table[state]).item()

    def update_q_table(self, state, action, reward, next_state):
        self.q_table[state][action] += self.learning_rate * (
                reward + self.discount_factor * torch.max(self.q_table[next_state]) - self.q_table[state][action]
        )


class AdapQLTuner:
    def __init__(self):
        self.__window_size_set = [pow(2, i) for i in range(0, int(
            np.log2(context.glContext.config['snap_num_train'])) + 1)]  # [1,2,4,8]
        self.__vertex_size_set = [pow(2, i) for i in range(0, int(
            np.log2(context.glContext.config['data_num_local'])) + 1)]  # [1,2,4,8,16,32,64,128,256,512,1024,2048]
        self.__window_action_set = [1 / 2, 1, 2]
        self.__vertex_action_set = [1 / 2, 1, 2]
        self.__state_size = 3  # 状态空间的维度
        self.__action_size = 9  # 动作空间的维度
        self.__ql_agent = _QLearningAgent(self.__state_size, self.__action_size)
        self.__total_reward = 0
        self.__state = np.array([0])
        self.__loss_lst = 0
        self.__delta_loss_lst = 0
        self.__action_trans_lst = [0, 0]
        self.__action_trans = None
        self.__action_lst = 0
        self.__avg_loss = 0
        self.__time_lst = 0
        self.__action = 0
        self.__action_description = None
        self.__base_loss_time = [0, 0]
        self.__state_bound = None

    def init_adap(self, test_dataset, model):
        model.eval()
        hidden_state = [None for i in range(len(context.glContext.config['hidden']))]
        cost_test = 0
        for time, snapshot in enumerate(test_dataset):
            y_hat, hidden_state = model(snapshot.x, snapshot.edge, snapshot.edge_weight, hidden_state,
                                        snapshot.deg)
            cost_test = cost_test + torch.mean((y_hat.view(-1) - snapshot.y) ** 2)
        cost_test = cost_test / (time + 1)

        test_num = test_dataset.target_vertex[0][0].shape[0]
        acc_avrg = getAccAvrg([1, test_num], [1, cost_test])
        self.__loss_lst = acc_avrg['test']

    def trans_to_size(self, action_div):
        window_size = int(context.glContext.config['window_size'] * action_div[0])
        batch_size = int(context.glContext.config['batch_size'] * action_div[1])
        if not self.__window_size_set.__contains__(window_size):
            window_size = context.glContext.config['window_size']
        if not self.__vertex_size_set.__contains__(batch_size):
            batch_size = context.glContext.config['batch_size']
        return [window_size, batch_size]

    def trans_to_description(self, action_div):
        action_one = None
        action_two = None
        if action_div[0] == self.__window_action_set[0]:
            action_one = 'descent'
        elif action_div[0] == self.__window_action_set[1]:
            action_one = 'keep'
        else:
            action_one = 'increase'

        if action_div[1] == self.__vertex_action_set[0]:
            action_two = 'descent'
        elif action_div[1] == self.__vertex_action_set[1]:
            action_two = 'keep'
        else:
            action_two = 'increase'
        return [action_one, action_two]

    def distributed_strategy_update(self):
        # 发送参数到其他设备
        with torch.no_grad():
            if dist.get_rank() == 0:
                # 发送参数到rank为1的设备
                data_to_send = torch.tensor(
                    np.array([context.glContext.config['window_size'], context.glContext.config['batch_size']]))
                for i in range(1, context.glContext.config['worker_num']):
                    dist.send(tensor=data_to_send, dst=i)
            else:
                # 接收参数
                received_data = torch.zeros(2, dtype=torch.int64)
                dist.recv(tensor=received_data, src=0)
                received_data = received_data.detach().numpy()
                context.glContext.config['window_size'] = received_data[0]
                context.glContext.config['batch_size'] = received_data[1]

    def get_state(self, delta_loss):
        if delta_loss >= 2 * self.__state_bound:
            return 0
        elif self.__state_bound <= delta_loss and delta_loss < 2 * self.__state_bound:
            return 1
        else:
            return 2

    def train(self, loss, step, time):
        if context.glContext.config['id'] != 0:
            return
        if step == 0:
            n = context.glContext.config['print_result_interval']
            self.__delta_loss_lst = (pow(0.95, n) - 1) / (0.95 - 1) * (self.__loss_lst - loss)
            # for i in range(context.glContext.config['print_result_interval']):
            #     self.__delta_loss_lst+=(self.__loss_lst-loss)*pow(0.95,i)
            self.__loss_lst = loss
            self.__time_lst = time
            self.__action_trans_lst = [context.glContext.config['window_size'], context.glContext.config['batch_size']]
            self.__action_lst = 0
            self.__action_trans = [context.glContext.config['window_size'], context.glContext.config['batch_size']]
            self.__action_description = ['keep', 'keep']
            self.__base_loss_time[0] = self.__delta_loss_lst
            self.__base_loss_time[1] = time
            self.__state_bound = self.__delta_loss_lst / 3
            self.__state = self.get_state(self.__delta_loss_lst)
            return

        delta_loss = self.__loss_lst - loss
        # loss_reward=(delta_loss-pow(0.95,step)*self.__base_loss_time[0])/delta_loss
        # time_reward=(self.__base_loss_time[1]-time)/self.__base_loss_time[1]
        n = context.glContext.config['print_result_interval']
        # loss_reward = (delta_loss - pow(0.995, n) * self.__delta_loss_lst) / delta_loss
        loss_reward = (delta_loss - self.__delta_loss_lst) / delta_loss
        # loss_reward=delta_loss/self.__loss_lst
        time_reward = (self.__time_lst - time) / self.__time_lst
        if loss_reward < 0 or delta_loss < 0:
            time_reward = 0
            loss_reward *= 2

        reward = loss_reward + time_reward

        self.__total_reward += reward
        next_state = self.get_state(delta_loss)
        self.__ql_agent.update_q_table(self.__state, self.__action, reward, next_state)

        print(
            f"Episode: {step + 1}, Reward: {reward}, Action_Description: {self.__action_description}, Last Action:{self.__action_trans_lst},Action_Trans:{self.__action_trans}, Action: {self.__action}")

        # save needed old data

        self.__state = next_state
        self.__delta_loss_lst = delta_loss
        self.__loss_lst = loss
        self.__time_lst = time
        self.__action_lst = self.__action
        self.__action_trans_lst = self.__action_trans

        # select a new action
        self.__action = self.__ql_agent.choose_action(self.__state)
        action_div = [self.__window_action_set[int(self.__action / len(self.__vertex_action_set))],
                      self.__vertex_action_set[self.__action % len(self.__vertex_action_set)]]
        self.__action_trans = self.trans_to_size(action_div)
        context.glContext.config['window_size'] = self.__action_trans[0]
        context.glContext.config['batch_size'] = self.__action_trans[1]
        self.__action_description = self.trans_to_description(action_div)
