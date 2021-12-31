import rlcard
import argparse
import os
import torch
from Agent.DuelDQNAgent import DuelDQNAgent
from rlcard.agents import RandomAgent
from rlcard.utils import get_device, tournament, Logger
from Utils.Utils import reshape_own_reward, reshape_reward
from Agent.HelperTwo import HelperAgent


is_training = False
action_map = {}
for i in range(38):
    if (i >= 0) & (i <= 8):
        action_map["bamboo-" + str(i + 1)] = i
    if (i >= 9) & (i <= 17):
        action_map["characters-" + str(i % 9 + 1)] = i
    if (i >= 18) & (i <= 26):
        action_map["dots-" + str(i % 18 + 1)] = i
    if i == 27:
        action_map['dragons-green'] = i
    if i == 28:
        action_map['dragons-red'] = i
    if i == 29:
        action_map['dragons-white'] = i
    if i == 30:
        action_map['winds-east'] = i
    if i == 31:
        action_map['winds-west'] = i
    if i == 32:
        action_map['winds-north'] = i
    if i == 33:
        action_map['winds-south'] = i
    if i == 34:
        action_map['pong'] = i
    if i == 35:
        action_map['chow'] = i
    if i == 36:
        action_map['gong'] = i
    if i == 37:
        action_map['stand'] = i


def lose_env(env):
    payoffs = [0]
    while payoffs[0] < 1:
        trajectories, payoffs = env.run(is_training=False)
    return trajectories, payoffs


def win_env(env):
    flag = True
    index = 0
    while flag:
        trajectories, payoffs = env.run(is_training=False)
        for i in range(len(payoffs)):
            if payoffs[i] == 1:
                index = i
                flag = False
                break
    return trajectories[index], payoffs[index]


def save_model(dueling_agent, epoch, score):
    save = {
        'net': dueling_agent.q_estimator.qnet.state_dict(),
        'optimizer': dueling_agent.q_estimator.optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(save, os.path.join(os.getcwd(), 'Model3', str(epoch) + '_' + str(score).replace(".", "-") + '_' + 'DuelingDQN.pth'))


def run(args):
    device = get_device()
    # 设置第一阶段的麻将环境
    env_learn = rlcard.make(args.env, config={'seed': 2021})
    agents = [HelperAgent(num_actions=env_learn.num_actions)]
    for _ in range(env_learn.num_players-1):
        agents.append(RandomAgent(num_actions=env_learn.num_actions))
    env_learn.set_agents(agents)
    # 设置第二阶段的麻将环境，不用Helper
    env = rlcard.make(args.env, config={'seed': 2021})
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
    agents = [duel_agent]
    for _ in range(env.num_players-1):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)
    cut_episode = int((args.num_episodes + 1) / 2)
    with Logger(os.path.join(os.getcwd(), 'Logger')) as logger:
        for episode in range(1, cut_episode):
            print("epoch: {} / {}, env: {}".format(episode, cut_episode, "env_learn"))
            # 前1/2训练过程学习HelperAgent的操作
            trajectories, payoffs = env_learn.run()
            trajectories = reshape_reward(trajectories[0], payoffs[0])
            for ts in trajectories[0]:
                duel_agent.feed(ts)
            if episode % args.evaluate_steps == 0:
                score = tournament(env, 200)[0]
                save_model(duel_agent, episode, score)
                logger.log_performance(env.timestep, score)
        for episode in range(cut_episode, args.num_episodes + 1):
            print("epoch: {} / {}, env: {}".format(episode, args.num_episodes, "env_battle"))
            # 后1/2训练过程学习自己的操作
            if episode % 2 == 0:
                trajectories, payoffs = win_env(env)
            else:
                trajectories, payoffs = lose_env(env)
            trajectories = reshape_own_reward(trajectories[0], payoffs[0])
            for ts in trajectories[0]:
                duel_agent.feed(ts)
            if episode % args.evaluate_steps == 0:
                logger.log_performance(env.timestep, tournament(env, 500)[0])
    save_model(dueling_agent=duel_agent, epoch=episode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dueling DQN model")
    parser.add_argument('--env', type=str, default='mahjong')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--num_episodes', type=int, default=100000)
    parser.add_argument('--evaluate_steps', type=int, default=300)

    args = parser.parse_args()
    run(args)
