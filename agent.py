import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha=.75, gamma=1, get_epsilon=lambda i: .8*i**.999,
                 get_alpha=None, get_gamma=None, beta=.8, get_beta=None,
                 c1=0, get_c1=None, c2=0, get_c2=None):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        - alpha: default learning rate
        - gamma: defuult discount rate
        - get_epsilon: function to choose epsilon (action stocahsticity) given episode number
        - get_alpha: function to choose alpha (learning rate) given episode number
        - get_gamma: function to choose gamma (discount rate) given episode number
        
        """
        self.alpha_init = alpha
        self.gamma_init = gamma
        self.beta_init = beta
        self.c1_init = c1
        self.c2_init = c2
        self.get_epsilon = get_epsilon
        self.get_alpha = (lambda i:self.alpha_init) if get_alpha is None else get_alpha
        self.get_gamma = (lambda i:self.gamma_init) if get_gamma is None else get_gamma
        self.get_beta = (lambda i:self.beta_init) if get_beta is None else get_beta
        self.get_c1 = (lambda i:self.c1_init) if get_c1 is None else get_c1
        self.get_c2 = (lambda i:self.c2_init) if get_c2 is None else get_c2
        
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.recent = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = self.get_epsilon(0)
        self.alpha = self.get_alpha(0)
        self.gamma = self.get_gamma(0)
        self.beta = self.get_beta(0)
        self.c1 = self.get_c1(0)
        self.c2 = self.get_c2(0)
        self.i_episode = 0
        self.temp_p = None
        self.temp_r = None
        self.temp_q = None

    def select_action(self, state):
        """ Given the state, select an action
            using epsilon-greedy selection method

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """


        if state in self.Q:
            q = np.asarray(self.Q[state])
            r = np.asarray(self.recent[state])
            p = self.softmax(q*self.c1 - r*self.c2)
            greedy_action = np.asarray(self.Q[state]).argmax()
            random_action = np.random.choice(self.nA, p=p) 
            action = np.random.choice([random_action, greedy_action], 
                                      p=[self.epsilon, 1-self.epsilon])
            self.temp_p = p
            self.temp_q = q
            self.temp_r = r
        else:
            action = np.random.choice(self.nA) 
        return action
 
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple,
            according to the standard Q-learning procedure
            
            Also pdate learning parameters for next episode if current one is done

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        greedy_action = np.asarray(self.Q[next_state]).argmax()
        self.Q[state][action] += self.alpha*(reward + 
                                             self.gamma*self.Q[next_state][greedy_action] - 
                                             self.Q[state][action])
        self.recent[state][action] += 1
        
        if done:
            self.i_episode += 1
            for state in self.recent:
                self.recent[state] = [count*self.beta for count in self.recent[state]]
            self.epsilon = self.get_epsilon(self.i_episode)
            self.alpha = self.get_alpha(self.i_episode)
            self.gamma = self.get_gamma(self.i_episode)
            self.beta = self.get_beta(self.i_episode)
            self.c1 = self.get_c1(self.i_episode)
            self.c2 = self.get_c2(self.i_episode)
#            if not self.i_episode % 1000:
#                print(self.i_episode, self.epsilon,
#                      self.softmax(self.temp_q), self.softmax(self.temp_r), self.temp_p)

    @staticmethod
    def softmax(a):
        e = np.exp(a)
        return e/e.sum()
