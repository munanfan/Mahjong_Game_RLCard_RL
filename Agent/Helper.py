from Utils import double_same_actions, triple_same_actions, one_same_3_actions, one_same_5_actions


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
        bad_actions_1 = triple_same_actions(state['obs'][0])
        bad_actions_2 = double_same_actions(state['obs'][0])
        good_actions_1 = one_same_5_actions(state['obs'][0])
        good_actions_2 = one_same_3_actions(state['obs'][0])
        legal_actions = state['legal_actions']
        legal_actions = sorted(legal_actions)
        c_bad_1 = []
        c_bad_2 = []
        c_good_1 = []
        c_good_2 = []
        c_legal_actions = []
        for i in legal_actions:
            c_legal_actions.append(i)
            if i in bad_actions_1:
                c_bad_1.append(i)
            if i in bad_actions_2:
                c_bad_2.append(i)
            if i in good_actions_1:
                c_good_1.append(i)
            if i in good_actions_2:
                c_good_2.append(i)
        if len(c_good_1) > 0:
            action = c_good_1[-1]
        elif len(c_good_2) > 0:
            action = c_good_2[-1]
        elif len(c_bad_2) > 0:
            action = c_bad_2[-1]
        elif len(c_bad_1) > 0:
            action = c_bad_1[-1]
        else:
            action = c_legal_actions[-1]
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
