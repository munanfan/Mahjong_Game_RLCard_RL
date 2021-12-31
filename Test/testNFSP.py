import os
import rlcard
import torch
import numpy as np
from Agent.DuelDQNAgent import DuelDQNAgent
from rlcard.agents import DQNAgent, NFSPAgent
from rlcard.utils import get_device

device = get_device()
env = rlcard.make("mahjong")
# 提前加载好模型
nfsp_agent = agent = NFSPAgent(num_actions=env.num_actions,
                               state_shape=env.state_shape[0],
                               hidden_layers_sizes=[64, 64],
                               q_mlp_layers=[64, 64],
                               device=device)

model_path = os.path.join("..", "TrainedModel", "Others", '1500_-0-0085_nfspagent.pth')
# print(torch.load(model_path)['net'])
nfsp_agent.policy_network.load_state_dict(torch.load(model_path)['net'])
# 加载DQN

agents = [nfsp_agent]
for _ in range(env.num_players - 1):
    agents.append(nfsp_agent)
env.set_agents(agents)
win_counter = 0
lose_counter = 0
he_counter = 0
pay_result = np.array([0, 0, 0, 0])
for i in range(1000):
    tra, pay = env.run()
    pay_result = pay_result + pay
    if pay[0] == 0:
        he_counter += 1
    if pay[0] < 0:
        lose_counter += 1
    if pay[0] > 0:
        win_counter += 1
print("胜利次数：", win_counter)
print("失败次数：", lose_counter)
print("和牌次数：", he_counter)
print(pay_result)
