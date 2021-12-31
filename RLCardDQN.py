''' An example of training a reinforcement learning agent on the environments in RLCard
'''
import os
import argparse
import torch
import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import get_device, set_seed, tournament, Logger, plot_curve
from Agent.HelperTwo import HelperAgent
from Utils import get_first, get_single, get_dead_actions, get_none_1_none, get_two_continue, get_1_gap, get_two_same, get_three_same, get_three_continue



def reshape_reward(trajectory, payoff):
    ''' Reorganize the trajectory to make it RL friendly

    Args:
        trajectory (list): A list of trajectories
        payoffs (list): A list of payoffs for the players. Each entry corresponds to one player

    Returns:
        (list): A new trajectories that can be fed into RL algorithms.

    '''
    new_trajectories = [[]]
    for i in range(0, len(trajectory)-2, 2):
        reward = 0
        # 设置是否结束
        if i ==len(trajectory)-3:
            done = True
        else:
            done = False
        # 获取当前状态
        current_state = trajectory[i]
        good_0_actions = get_first(current_state['obs'][0])
        good_1_actions = get_single(current_state['obs'][0]) + get_dead_actions(current_state)  # 最好的行为：单张+死牌+听碰杠吃
        good_2_actions = get_none_1_none(current_state['obs'][0])  # 次的行为：空有空
        good_3_actions = get_two_continue(current_state['obs'][0]) + get_1_gap(current_state['obs'][0])  # 有空有+两个连续的
        good_4_actions = get_two_same(current_state['obs'][0]) + get_three_same(
            current_state['obs'][0]) + get_three_continue(
            current_state['obs'][0])  # 次次行为，拆掉连续的
        legal_actions = current_state['legal_actions']
        legal_actions = sorted(legal_actions)
        c_good_0 = []
        c_good_1 = []
        c_good_2 = []
        c_good_3 = []
        c_good_4 = []
        c_others = []
        c_legal_actions = []
        # 计算不同的出牌等级
        for j in legal_actions:
            c_legal_actions.append(j)
            if j in good_0_actions:
                c_good_0.append(j)
            elif j in good_1_actions:
                c_good_1.append(j)
            elif j in good_2_actions:
                c_good_2.append(j)
            elif j in good_3_actions:
                c_good_3.append(j)
            elif j in good_4_actions:
                c_good_4.append(j)
            else:
                c_others.append(j)
        action_make = trajectory[i + 1]
        # 查看正确的行为在哪一层次
        if action_make in c_good_0:
            reward = 1
        elif action_make in c_good_1:
            reward = 0.8
        elif action_make in c_good_2:
            reward = 0.6
        elif action_make in c_others:
            reward = 0.4
        elif action_make in c_good_3:
            reward = 0.2
        elif action_make in c_good_4:
            reward = 0
        transition = trajectory[i:i+3].copy()
        transition.insert(2, reward)
        transition.append(done)
        new_trajectories[0].append(transition)
    return new_trajectories

def save_model(dueling_agent, epoch, score):
    save = {
        'net': dueling_agent.q_estimator.qnet.state_dict(),
        'optimizer': dueling_agent.q_estimator.optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(save, os.path.join(os.getcwd(), 'TrainedModel/DQNwithrewardhelper', str(epoch) + '_' + str(score).replace(".", "-") + '_' + 'DQNwithrewardhelper.pth'))



def train(args):
    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed})

    # Initialize the agent and use random agents as opponents
    if args.algorithm == 'dqn':
        from rlcard.agents import DQNAgent
        agent = DQNAgent(num_actions=env.num_actions,
                         state_shape=env.state_shape[0],
                         mlp_layers=[64, 64],
                         learning_rate=0.00001,
                         device=device)
    elif args.algorithm == 'nfsp':
        from rlcard.agents import NFSPAgent
        agent = NFSPAgent(num_actions=env.num_actions,
                          state_shape=env.state_shape[0],
                          hidden_layers_sizes=[64, 64],
                          q_mlp_layers=[64, 64],
                          device=device)
    agents = [agent]
    for _ in range(env.num_players - 1):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)

    env_learn = rlcard.make(args.env, config={'seed': args.seed})
    agents_learn = [HelperAgent(num_actions=env_learn.num_actions)]
    for _ in range(env.num_players - 1):
        agents_learn.append(RandomAgent(num_actions=env.num_actions))
    env_learn.set_agents(agents_learn)

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            if args.algorithm == 'nfsp':
                agents[0].sample_episode_policy()

            # Generate data from the environment
            trajectories, payoffs = env_learn.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reshape_reward(trajectories[0], payoffs[0])

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                score = tournament(env, args.num_eval_games)[0]
                save_model(agents[0], episode, score)
                logger.log_performance(env.timestep, score)

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, args.algorithm)

    # Save model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent, save_path)
    print('TrainedModel saved in', save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
    parser.add_argument('--env', type=str, default='mahjong',
                        choices=['blackjack', 'leduc-holdem', 'limit-holdem', 'doudizhu', 'mahjong', 'no-limit-holdem',
                                 'uno', 'gin-rummy'])
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['dqn', 'nfsp'])
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=100000)
    parser.add_argument('--num_eval_games', type=int, default=2000)
    parser.add_argument('--evaluate_every', type=int, default=300)
    parser.add_argument('--log_dir', type=str, default='experiments/mahjong_nfsp_result/')

    args = parser.parse_args(args=[])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)

