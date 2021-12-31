import rlcard
import numpy as np
from Agent.HelperTwo import HelperAgent
from rlcard.agents import RandomAgent

env = rlcard.make("mahjong")
agents = [HelperAgent(num_actions=env.num_actions)]
for _ in range(env.num_players-1):
    agents.append(RandomAgent(num_actions=env.num_actions))
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
# with Logger(os.path.join(os.getcwd(), 'Logger')) as logger:
#     logger.log_performance(env.timestep, tournament(env, 1000)[0])
