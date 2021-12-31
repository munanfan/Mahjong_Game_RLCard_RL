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
duel_agent = DuelDQNAgent(replay_memory_size=20000,
                          replay_memory_init_size=8000,
                          update_target_estimator_every=100,
                          discount_factor=0.9,
                          epsilon_start=1.0,
                          epsilon_end=0.1,
                          epsilon_decay_steps=100000,
                          batch_size=512,
                          num_actions=env.num_actions,
                          state_shape=env.state_shape[0],
                          train_every=1,
                          learning_rate=0.00005,
                          device=device)

model_path = os.path.join("..", "TrainedModel", "Dueling DQN-1", '16000_0-205_DuelingDQN.pth')
duel_agent.q_estimator.qnet.load_state_dict(torch.load(model_path)['net'])
# 加载DQN

agents = [duel_agent]
for _ in range(env.num_players - 1):
    agents.append(duel_agent)
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
