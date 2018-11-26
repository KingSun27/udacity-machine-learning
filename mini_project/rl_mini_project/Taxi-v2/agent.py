import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))


    def get_pros_greedy_epsilon(self,state,epsilon = 0.005):
        policy_s = np.ones(self.nA) * epsilon /self.nA
        best_a = np.argmax(self.Q[state])
        policy_s[best_a] = 1 - epsilon + epsilon /self.nA
        return policy_s



    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        epsilon = 0.05

        action = np.random.choice(self.nA,p= self.get_pros_greedy_epsilon(state))

        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # learning_rate
        alpha = 0.08
        # discount
        gamma = 0.95
        #next_action = self.select_action(next_state)

        next_state_Q_expectation = np.dot(self.Q[next_state],self.get_pros_greedy_epsilon(next_state))
        self.Q[state][action] += alpha * (reward + gamma * next_state_Q_expectation - self.Q[state][action])