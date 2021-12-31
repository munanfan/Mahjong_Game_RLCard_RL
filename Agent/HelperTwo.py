from Utils.Utils import get_first, get_single, get_dead_actions, get_none_1_none, get_two_continue, get_1_gap, get_two_same, get_three_same, get_three_continue


class HelperAgent(object):
    ''' A random agent. Random agents is for running toy examples on the card games
    '''

    def __init__(self, num_actions):
        ''' Initilize the random agent

        Args:
            num_actions (int): The size of the ouput action space
        '''
        self.use_raw = False
        self.num_actions = num_actions

    @staticmethod
    def step(state):
        ''' Predict the action given the curent state in gerenerating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
        '''
        good_0_actions = get_first(state['obs'][0])
        good_1_actions = get_single(state['obs'][0]) + get_dead_actions(state)  # 最好的行为：单张+死牌+听碰杠吃
        good_2_actions = get_none_1_none(state['obs'][0])  # 次的行为：空有空
        good_3_actions = get_two_continue(state['obs'][0]) + get_1_gap(state['obs'][0])  # 有空有+两个连续的
        good_4_actions = get_two_same(state['obs'][0]) + get_three_same(state['obs'][0]) + get_three_continue(
            state['obs'][0])  # 次次行为，拆掉连续的
        legal_actions = state['legal_actions']
        legal_actions = sorted(legal_actions)
        c_good_0 = []
        c_good_1 = []
        c_good_2 = []
        c_good_3 = []
        c_good_4 = []
        c_others = []
        c_legal_actions = []
        for i in legal_actions:
            c_legal_actions.append(i)
            if i in good_1_actions:
                c_good_0.append(i)
            elif i in good_1_actions:
                c_good_1.append(i)
            elif i in good_2_actions:
                c_good_2.append(i)
            elif i in good_3_actions:
                c_good_3.append(i)
            elif i in good_4_actions:
                c_good_4.append(i)
            else:
                c_others.append(i)
        if len(c_good_0) > 0:
            action = c_good_0[-1]
        elif len(c_good_1) > 0:
            action = c_good_1[-1]
        elif len(c_good_2) > 0:
            action = c_good_2[-1]
        elif len(c_others) > 0:
            action = c_others[-1]
        elif len(c_good_3) > 0:
            action = c_good_3[-1]
        elif len(c_good_4) > 0:
            action = c_good_4[-1]
        return action

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
        info = None
        return self.step(state), info
