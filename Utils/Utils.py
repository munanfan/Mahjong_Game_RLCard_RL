import numpy as np


def double_same_actions(state):
    # 两个相同的牌的组合
    same_actions = []
    continue_actions = []
    for i in range(state.shape[0]):
        if state[i].sum() == 2:
            same_actions.append(i)
    # 两个连续的牌
    i = 0
    while i <= 7:
        if state[i:i + 2].sum() == 2:
            continue_actions.append(i)
            continue_actions.append(i + 1)
            i += 2
            continue
        i += 1
    i = 9
    while i <= 16:
        if state[i:i + 2].sum() == 2:
            continue_actions.append(i)
            continue_actions.append(i + 1)
            i += 2
            continue
        i += 1
    i = 18
    while i <= 25:
        if state[i:i + 2].sum() == 2:
            continue_actions.append(i)
            continue_actions.append(i + 1)
            i += 2
            continue
        i += 1
    bad_actions = list(set(same_actions + continue_actions))
    return bad_actions


def triple_same_actions(state):
    # 三个相同的牌的组合或者三个连续的牌的组合
    same_actions = []
    continue_actions = []
    # 首先处理三个相同的牌的情况
    for i in range(state.shape[0]):
        if state[i].sum() == 3:
            same_actions.append(i)
    i = 0
    while i <= 6:
        if state[i:i + 3].sum() == 3:
            continue_actions.append(i)
            continue_actions.append(i + 1)
            continue_actions.append(i + 2)
            i += 3
            continue
        i += 1
    i = 9
    while i <= 15:
        if state[i:i + 3].sum() == 3:
            continue_actions.append(i)
            continue_actions.append(i + 1)
            continue_actions.append(i + 2)
            i += 3
            continue
        i += 1
    i = 18
    while i <= 24:
        if state[i:i + 3].sum() == 3:
            continue_actions.append(i)
            continue_actions.append(i + 1)
            continue_actions.append(i + 2)
            i += 3
            continue
        i += 1
    bad_actions = list(set(same_actions + continue_actions))
    return bad_actions


def one_same_5_actions(state):
    good_actions = []
    # 最好的出牌选择
    if state[0:3].sum() == 1:
        good_actions.append(0)
    if state[1:4].sum() == 1:
        good_actions.append(1)

    if state[9:12].sum() == 1:
        good_actions.append(9)
    if state[10:13].sum() == 1:
        good_actions.append(10)

    if state[18:21].sum() == 1:
        good_actions.append(18)
    if state[19:22].sum() == 1:
        good_actions.append(19)

    if state[6:9].sum() == 1:
        good_actions.append(8)
    if state[5:8].sum() == 1:
        good_actions.append(7)

    if state[15:18].sum() == 1:
        good_actions.append(17)
    if state[14:17].sum() == 1:
        good_actions.append(16)

    if state[24:27].sum() == 1:
        good_actions.append(26)
    if state[23:26].sum() == 1:
        good_actions.append(25)

    i = 0
    while i <= 4:
        if state[i:i + 5].sum() == 1:
            good_actions.append(i + 2)
            i += 1
            continue
        i += 1
    i = 9
    while i <= 13:
        if state[i:i + 5].sum() == 1:
            good_actions.append(i + 2)
            i += 1
            continue
        i += 1
    i = 18
    while i <= 22:
        if state[i:i + 5].sum() == 1:
            good_actions.append(i + 2)
            i += 1
            continue
        i += 1
    for i in range(27, 34):
        if state[i].sum() == 1:
            good_actions.append(i)
    good_actions.append(34)
    good_actions.append(35)
    good_actions.append(36)
    return good_actions


def one_same_3_actions(state):
    good_actions = []
    # 最好的出牌选择
    if state[0:2].sum() == 1:
        good_actions.append(0)
    if state[9:11].sum() == 1:
        good_actions.append(9)
    if state[18:20].sum() == 1:
        good_actions.append(18)

    if state[7:9].sum() == 1:
        good_actions.append(8)
    if state[16:18].sum() == 1:
        good_actions.append(17)
    if state[25:27].sum() == 1:
        good_actions.append(26)

    i = 0
    while i <= 6:
        if state[i:i + 3].sum() == 1:
            good_actions.append(i + 1)
            i += 1
            continue
        i += 1
    i = 9
    while i <= 15:
        if state[i:i + 3].sum() == 1:
            good_actions.append(i + 1)
            i += 1
            continue
        i += 1
    i = 18
    while i <= 24:
        if state[i:i + 3].sum() == 1:
            good_actions.append(i + 1)
            i += 1
            continue
        i += 1
    return good_actions


def get_two_same(state):
    # 两个相同的牌的组合
    same_actions = []
    for i in range(state.shape[0]):
        if state[i].sum() == 2:
            same_actions.append(i)
    return same_actions


def get_three_same(state):
    # 三个相同的牌的组合或者三个连续的牌的组合
    same_actions = []
    # 首先处理三个相同的牌的情况
    for i in range(state.shape[0]):
        if state[i].sum() == 3:
            same_actions.append(i)
    return same_actions


def get_two_continue(state):
    continue_actions = []
    # 两个连续的牌
    i = 1
    while i <= 6:
        if (state[i-1].sum() == 0) & (state[i].sum() == 1) & (state[i+1].sum() == 1) & (state[i+2].sum() == 0):
            continue_actions.append(i)
            continue_actions.append(i+1)
            i += 3
            continue
        i += 1
    i = 10
    while i <= 15:
        if (state[i-1].sum() == 0) & (state[i].sum() == 1) & (state[i+1].sum() == 1) & (state[i+2].sum() == 0):
            continue_actions.append(i)
            continue_actions.append(i+1)
            i += 3
            continue
        i += 1
    i = 19
    while i <= 24:
        if (state[i-1].sum() == 0) & (state[i].sum() == 1) & (state[i+1].sum() == 1) & (state[i+2].sum() == 0):
            continue_actions.append(i)
            continue_actions.append(i+1)
            i += 3
            continue
        i += 1
    return continue_actions


def get_three_continue(state):
    continue_actions = []
    i = 0
    while i <= 6:
        if state[i:i + 3].sum() == 3:
            continue_actions.append(i)
            continue_actions.append(i + 1)
            continue_actions.append(i + 2)
            i += 3
            continue
        i += 1
    i = 9
    while i <= 15:
        if state[i:i + 3].sum() == 3:
            continue_actions.append(i)
            continue_actions.append(i + 1)
            continue_actions.append(i + 2)
            i += 3
            continue
        i += 1
    i = 18
    while i <= 24:
        if state[i:i + 3].sum() == 3:
            continue_actions.append(i)
            continue_actions.append(i + 1)
            continue_actions.append(i + 2)
            i += 3
            continue
        i += 1
    return continue_actions


def get_first(state):
    single_actions = []
    for i in range(27, 34):
        if state[i].sum() == 1:
            single_actions.append(i)
    return single_actions


def get_single(state):
    single_actions = []
    for i in [0, 9, 18]:
        if (state[i].sum() == 1) & (state[i+1].sum() == 0) & (state[i+2].sum() == 0):
            single_actions.append(i)
    for i in [1, 7, 10, 16, 19, 25]:
        if (state[i-1].sum() == 0) & (state[i].sum() == 1) & (state[i+1].sum() == 0):
            single_actions.append(i)
    for i in [8, 17, 26]:
        if (state[i-2].sum() == 0) & (state[i-1].sum() == 0) & (state[i].sum() == 1):
            single_actions.append(i)
    # 判断后面是否有牌
    i = 2
    while i <= 6:
        if (state[i].sum() == 1) & (state[i-2:i].sum() == 0) & (state[i+1:i+3].sum() == 0):
            single_actions.append(i)
            # 如果加入成功，说明后面两个位置都没有牌
            i += 3
            continue
        # 如果没有加入成功，前进一位
        i += 1
    i = 11
    while i <= 15:
        if (state[i].sum() == 1) & (state[i-2:i].sum() == 0) & (state[i+1:i+3].sum() == 0):
            single_actions.append(i)
            # 如果加入成功，说明后面两个位置都没有牌
            i += 3
            continue
        # 如果没有加入成功，前进一位
        i += 1
    i = 20
    while i <= 24:
        if (state[i].sum() == 1) & (state[i-2:i].sum() == 0) & (state[i+1:i+3].sum() == 0):
            single_actions.append(i)
            # 如果加入成功，说明后面两个位置都没有牌
            i += 3
            continue
        # 如果没有加入成功，前进一位
        i += 1
    # 顺便在加入听、杠、吃、碰
    # single_actions.append(37)
    single_actions.append(34)
    single_actions.append(35)
    single_actions.append(36)
    return single_actions


def get_none_1_none(state):
    actions = []
    # 处理开头和结尾
    for i in [8, 17, 26]:
        if (state[i-2].sum() == 0) & (state[i-1].sum() == 0) & (state[i].sum() == 1):
            actions.append(i)
    for i in [0, 9, 18]:
        if (state[i].sum() == 1) & (state[i+1].sum() == 0) & (state[i+2].sum() == 0):
            actions.append(i)
    # 两个连续的牌
    i = 1
    while i <= 7:
        if (state[i-1].sum() == 0) & (state[i].sum() == 1) & (state[i + 1].sum() == 0):
            actions.append(i)
            i += 2
            continue
        i += 1
    i = 10
    while i <= 15:
        if (state[i-1].sum() == 0) & (state[i].sum() == 1) & (state[i + 1].sum() == 0):
            actions.append(i)
            i += 2
            continue
        i += 1
    i = 19
    while i <= 24:
        if (state[i-1].sum() == 0) & (state[i].sum() == 1) & (state[i + 1].sum() == 0):
            actions.append(i)
            i += 2
            continue
        i += 1
    return actions


def get_1_gap(state):
    gap_actions = []
    # 两个连续的牌
    i = 0
    while i <= 6:
        if (state[i].sum() == 1) & (state[i + 1].sum() == 0) & (state[i + 2].sum() == 1):
            gap_actions.append(i)
            gap_actions.append(i + 2)
            i += 3
            continue
        i += 1
    i = 9
    while i <= 15:
        if (state[i].sum() == 1) & (state[i + 1].sum() == 0) & (state[i + 2].sum() == 1):
            gap_actions.append(i)
            gap_actions.append(i + 2)
            i += 3
            continue
        i += 1
    i = 18
    while i <= 24:
        if (state[i].sum() == 1) & (state[i + 1].sum() == 0) & (state[i + 2].sum() == 1):
            gap_actions.append(i)
            gap_actions.append(i + 2)
            i += 3
            continue
        i += 1
    return gap_actions


def get_my_pai(state):
    pais = []
    for i in range(34):
        if state[i].sum() > 0:
            pais.append(i)
    return pais


def get_dead_actions(states):
    dead_actions = []
    # 先看自己手上的牌
    pais = get_my_pai(states['obs'][0])
    # 如果场上每张牌明牌的数量 = 3，说明是死牌
    for i in pais:
        if states['obs'][1:, i, :].sum() == 3:
            dead_actions.append(i)
    # 获取手上的相同的牌
    two_same_actions = get_two_same(states['obs'][0])
    # 手上2张相同的牌数量大于2，因为只需要一对，所以只用保留一对
    if len(two_same_actions) > 1:
        wait_actions = []
        for i in two_same_actions:
            if states['obs'][1:, i, :].sum() == 2:
                wait_actions.append(i)
        if len(two_same_actions) == len(wait_actions):
            # 如果都没牌了，保留一对
            dead_actions = dead_actions + wait_actions[:-1]
    return dead_actions


def reshape_label_reward(trajectory, payoff):
    ''' Reorganize the trajectory to make it RL friendly

    Args:
        trajectory (list): A list of trajectories
        payoffs (list): A list of payoffs for the players. Each entry corresponds to one player

    Returns:
        (list): A new trajectories that can be fed into RL algorithms.

    '''
    new_trajectories = [[]]
    for i in range(0, len(trajectory)-2, 2):
        reward = np.zeros(38)
        # 设置是否结束
        if i ==len(trajectory)-3:
            done =True
        else:
            done = False
        reward[trajectory[i + 1]] = 1
        transition = trajectory[i:i+3].copy()
        transition.insert(2, reward)
        transition.append(done)
        new_trajectories[0].append(transition)
    return new_trajectories


def reshape_own_reward(trajectory, payoff):
    ''' Reorganize the trajectory to make it RL friendly

    Args:
        trajectory (list): A list of trajectories
        payoffs (list): A list of payoffs for the players. Each entry corresponds to one player

    Returns:
        (list): A new trajectories that can be fed into RL algorithms.

    '''
    new_trajectories = [[]]
    for i in range(0, len(trajectory)-2, 2):
        reward = 100
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
        if len(c_good_0) > 0:
            if action_make in c_good_0:
                reward += 3
        elif len(c_good_1) > 0:
            if action_make in c_good_1:
                reward += 2
        elif len(c_good_2) > 0:
            if action_make in c_good_2:
                reward += 1
        elif len(c_others) > 0:
            if action_make in c_others:
                reward += 0
        elif len(c_good_3) > 0:
            if action_make in c_good_3:
                reward -= 1
        elif len(c_good_4) > 0:
            if action_make in c_good_4:
                reward -= 2
        transition = trajectory[i:i+3].copy()
        transition.insert(2, reward)
        transition.append(done)
        new_trajectories[0].append(transition)
    return new_trajectories


def reshape_own_reward(trajectory, payoff):
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
        if payoff > 0:
            reward = 1
        else:
            reward = 0
        transition = trajectory[i:i+3].copy()
        transition.insert(2, reward)
        transition.append(done)
        new_trajectories[0].append(transition)
    return new_trajectories


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
            reward = 10
        elif action_make in c_good_1:
            reward = 8
        elif action_make in c_good_2:
            reward = 6
        elif action_make in c_others:
            reward = 4
        elif action_make in c_good_3:
            reward = 2
        elif action_make in c_good_4:
            reward = 0
        transition = trajectory[i:i+3].copy()
        transition.insert(2, reward)
        transition.append(done)
        new_trajectories[0].append(transition)
    return new_trajectories