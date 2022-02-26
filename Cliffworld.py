#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import matplotlib.pyplot as plt 


# In[2]:


actions = np.array([[-1,0], [1,0], [0,-1], [0,1]]) # {U,D,L,R} which can also be represented as {0,1,2,3} indices
actions_to_take = ["up", "down", "left", "right"]
discount_factor = 1 #discount factor
alpha = 0.5
epsilon = 0.1
grid_world = np.array([4,12])
start_state = np.array([3,0])
goal = np.array([3,11])
states = [[grid_world[0]-1-i, j] for j in range(grid_world[1]) for i in range(grid_world[0])]
cliff = [[grid_world[0]-1, j] for j in range(1, grid_world[1]-1)]
q_values = np.zeros((len(states), len(actions)))


# In[3]:


class cliff_world_environment:
    def __init__(self):
        self.position = None
        
    def reset(self):
        self.position = start_state
        return(self.position)
        
    def actionRewardFunction(self, action):
        self.action = action
        # Possible actions
        if self.action == 0 and self.position[0] > 0:
            self.position = self.position + np.array(actions[0])
        if self.action == 1 and self.position[0] < 3:
            self.position = self.position + np.array(actions[1])
        if self.action == 2 and self.position[1] > 0:
            self.position = self.position + np.array(actions[2])
        if self.action == 3 and self.position[1] < 11:
            self.position = self.position + np.array(actions[3])

        if any(np.array_equal(self.position, x) for x in cliff):
            reward = -100 
            end_of_episode = True
        elif (np.array_equal(self.position, goal)):
            reward = -1
            end_of_episode = True
        else:
            reward = -1
            end_of_episode = False

        return(self.position, reward, end_of_episode)


# In[4]:


def epsilon_greedy(state_action_values, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(len(actions))
    else:
        state_idx = (state[0]*grid_world[1]+state[1])
        return np.argmax(state_action_values[state_idx])


# In[5]:


cliff_world = cliff_world_environment()


# # SARSA

# In[6]:


def sarsa(cliff_world_environment, alpha, epsilon, discount_factor, episodes):
    Q_sarsa = np.zeros((len(states), len(actions)))
    rewards = []
    avg_rewards = []
    
    for episode in range(episodes):
        #Resetting the agent after the end of each episode
        state = cliff_world.reset()
        end_of_episode = False
        sum_of_rewards = 0
        n = 0
        
        # Choose action        
        action = epsilon_greedy(Q_sarsa, state, epsilon)

        while not end_of_episode:        
            next_state, reward, end_of_episode = cliff_world_environment.actionRewardFunction(action)
            sum_of_rewards += reward
            n += 1
            
            # Choose next action
            next_action = epsilon_greedy(Q_sarsa, next_state, epsilon)
            
            state_idx = state[0]*grid_world[1] + state[1]
            next_state_idx = next_state[0]*grid_world[1] + next_state[1]
            
            # Update Q-value
            Q_sarsa[state_idx][action] += alpha * ((reward + discount_factor * Q_sarsa[next_state_idx][next_action]) - Q_sarsa[state_idx][action])

            # Update state and action        
            state = next_state
            action = next_action
                
        rewards.append(sum_of_rewards)
        avg_rewards.append(sum_of_rewards/n)
        
    return(rewards, avg_rewards, Q_sarsa)


# In[7]:


sarsa_rewards, sarsa_avg_rewards, q_values_sarsa = sarsa(cliff_world, alpha, epsilon, discount_factor, 500)


# In[8]:


plt.plot(sarsa_rewards[1:], color='blue', label='Sarsa')
plt.xlabel('Epsiodes')
plt.ylabel('Sum of rewards during episode')
plt.title('For $\\alpha$ = {} , $\epsilon$ = {} , $\delta$ = {}'.format(alpha,epsilon,discount_factor))
plt.legend(loc='best')
plt.show()


# In[9]:


plt.plot(sarsa_avg_rewards[1:], color='blue', label='Sarsa')
plt.xlabel('Epsiodes')
plt.ylabel('Average of rewards during episode')
plt.title('For $\\alpha$ = {} , $\epsilon$ = {} , $\delta$ = {}'.format(alpha,epsilon,discount_factor))
plt.legend(loc='best')
plt.show()


# In[10]:


sarsa_difference_in_rewards = []
for i in range(1,len(sarsa_rewards)-1):
    sarsa_difference_in_rewards.append(sarsa_rewards[i] - sarsa_rewards[i-1])

plt.plot(sarsa_difference_in_rewards[1:], color='blue', label='Sarsa')
plt.xlabel('Epsiodes')
plt.ylabel('Difference of rewards between episodes')
plt.title('For $\\alpha$ = {} , $\epsilon$ = {} , $\delta$ = {}'.format(alpha,epsilon,discount_factor))
plt.legend(loc='best')
plt.show()


# # Q-learning

# In[11]:


def qLearning(cliff_world_environment, alpha, epsilon, discount_factor, episodes):
    Q_qLearning = np.zeros((len(states), len(actions)))
    rewards = []
    avg_rewards = []
    
    for episode in range(episodes):
        #Resetting the agent after the end of each episode
        state = cliff_world.reset()
        end_of_episode = False
        sum_of_rewards = 0
        n = 0

        while not end_of_episode:            
            # Choose action        
            action = epsilon_greedy(Q_qLearning, state, epsilon)

            next_state, reward, end_of_episode = cliff_world_environment.actionRewardFunction(action)
            sum_of_rewards += reward
            n += 1
            
            state_idx = state[0]*grid_world[1] + state[1]
            next_state_idx = next_state[0]*grid_world[1] + next_state[1]

            # Update q_values       
            Q_qLearning[state_idx][action] += alpha * (reward + discount_factor * np.max(Q_qLearning[next_state_idx]) - Q_qLearning[state_idx][action])

            # Update state
            state = next_state

        rewards.append(sum_of_rewards)
        avg_rewards.append(sum_of_rewards/n)
    
    return(rewards, avg_rewards, Q_qLearning)


# In[12]:


qLearning_rewards, qLearning_avg_rewards, q_values_qLearning = qLearning(cliff_world, alpha, epsilon, discount_factor, 500)


# In[13]:


plt.plot(qLearning_rewards[1:], color='red', label='Q-Learning')
plt.xlabel('Epsiodes')
plt.ylabel('Sum of rewards during episode')
plt.title('For $\\alpha$ = {} , $\epsilon$ = {} , $\delta$ = {}'.format(alpha,epsilon,discount_factor))
plt.legend(loc='best')
plt.show()


# In[14]:


plt.plot(qLearning_avg_rewards[1:], color='red', label='Q-Learning')
plt.xlabel('Epsiodes')
plt.ylabel('Average of rewards during episode')
plt.title('For $\\alpha$ = {} , $\epsilon$ = {} , $\delta$ = {}'.format(alpha,epsilon,discount_factor))
plt.legend(loc='best')
plt.show()


# In[15]:


qLearning_difference_in_rewards = []
for i in range(1,len(qLearning_rewards)-1):
    qLearning_difference_in_rewards.append(qLearning_rewards[i] - qLearning_rewards[i-1])

plt.plot(qLearning_difference_in_rewards[1:], color='red', label='Q-Learning')
plt.xlabel('Epsiodes')
plt.ylabel('Difference of rewards between episodes')
plt.title('For $\\alpha$ = {} , $\epsilon$ = {} , $\delta$ = {}'.format(alpha,epsilon,discount_factor))
#plt.title('For alpha = {}, epsilon = {} and discount factor = {}'.format(0.5,0.4,discount_factor))
plt.legend(loc='best')
plt.show()


# # Compare the cumulative rewards obtained by SARSA vs Q-learning, vs the number of episodes.

# In[16]:


plt.plot(sarsa_rewards[1:], color='blue', label='Sarsa')
plt.plot(qLearning_rewards[1:], color='red', label='Q-Learning')
plt.xlabel('Epsiodes')
plt.ylabel('Sum of rewards during episode')
plt.title('For $\\alpha$ = {} , $\epsilon$ = {} , $\delta$ = {}'.format(alpha,epsilon,discount_factor))
plt.legend(loc='best')
plt.show()


# In[17]:


plt.plot(sarsa_difference_in_rewards[1:], color='blue', label='Sarsa')
plt.plot(qLearning_difference_in_rewards[1:], color='red', label='Q-Learning')
plt.xlabel('Epsiodes')
plt.ylabel('Difference of rewards between episodes')
plt.title('For $\\alpha$ = {} , $\epsilon$ = {} , $\delta$ = {}'.format(alpha,epsilon,discount_factor))
plt.legend(loc='best')
plt.show()


# In[18]:


plt.plot(sarsa_avg_rewards[1:], color='blue', label='Sarsa')
plt.plot(qLearning_avg_rewards[1:], color='red', label='Q-Learning')
plt.xlabel('Epsiodes')
plt.ylabel('Average of rewards during episode')
plt.title('For $\\alpha$ = {} , $\epsilon$ = {} , $\delta$ = {}'.format(alpha,epsilon,discount_factor))
plt.legend(loc='best')
plt.show()


# # Results for different combination of parameters

# In[19]:


for alpha1 in np.arange(0.1,0.5,0.1):
    for epsilon1 in np.arange(0,0.5,0.1):
        alpha1 = round(alpha1,1)
        epsilon1 = round(epsilon1,1)
        sarsa_rewards1, sarsa_avg_rewards1, q_values_sarsa1 = sarsa(cliff_world, alpha1, epsilon1, discount_factor, 500)
        qLearning_rewards1, qLearning_avg_rewards1, q_values_qLearning1 = qLearning(cliff_world, alpha1, epsilon1, discount_factor, 500)
        
        plt.plot(sarsa_rewards1[1:], color='blue', label='Sarsa')
        plt.plot(qLearning_rewards1[1:], color='red', label='Q-Learning')
        plt.xlabel('Epsiodes')
        plt.ylabel('Sum of rewards during episode')
        plt.title('For $\\alpha$ = {} , $\epsilon$ = {} , $\delta$ = {}'.format(alpha1,epsilon1,discount_factor))
        plt.legend(loc='best')
        plt.show()


# In[20]:


def OptimalPolicy_and_Path(q_value):
    q = np.copy(q_value)
    path = np.zeros((grid_world))-1
    state = start_state
    end_of_episode = False
    
    while not end_of_episode:
        state_idx = state[0]*grid_world[1]+state[1]
        a = np.argmax(q[state_idx])
        path[state[0],state[1]] = a

        state = state + np.array(actions[a])
        
        if np.array_equal(state, goal):
            end_of_episode = True
        
    for i in range(grid_world[0]):
        for j in range(grid_world[1]):
            state = [i,j]
            state_idx = state[0]*grid_world[1]+state[1]
            
            if np.array_equal(state,start_state):
                print(" S ",end = "")
                continue
            if np.array_equal(state,goal):
                print(" G ",end = "")
                return path
            if (all(q[state_idx]) == 0):
                print(" - ",end = "")
                continue
        
            a = path[i,j]
            
            if a == -1:
                print(" - ",end = "")
            elif a == 0:
                print(" U ",end = "")
            elif a == 1:
                print(" D ",end = "")
            elif a == 2:
                print(" L ",end = "")
            elif a == 3:
                print(" R ",end = "")
        print("")


# In[21]:


print("Sarsa Optimal Policy and Path\n")
sarsa_path = OptimalPolicy_and_Path(q_values_sarsa)
print("\n\nQ-learning Optimal Policy and Path\n")
qLearning_path = OptimalPolicy_and_Path(q_values_qLearning)


# In[22]:


def sarsa_decaying_epsilon(cliff_world_environment, alpha, epsilon, discount_factor, episodes):
    Q_sarsa = np.zeros((len(states), len(actions)))
    rewards = []
    avg_rewards = []
    
    epsilon_int = epsilon
    epsilon_end = 0.1
    
    for episode in range(episodes):    
        r = max((episodes-episode)/episodes, 0)
        epsilon = (epsilon_int-epsilon_end)*r + epsilon_end
        
        #Resetting the agent after the end of each episode
        state = cliff_world.reset()
        end_of_episode = False
        sum_of_rewards = 0
        n = 0
        
        # Choose action        
        action = epsilon_greedy(Q_sarsa, state, epsilon)

        while not end_of_episode:        
            next_state, reward, end_of_episode = cliff_world_environment.actionRewardFunction(action)
            sum_of_rewards += reward
            n += 1
            
            # Choose next action
            next_action = epsilon_greedy(Q_sarsa, next_state, epsilon)
            
            state_idx = state[0]*grid_world[1] + state[1]
            next_state_idx = next_state[0]*grid_world[1] + next_state[1]
            
            # Update Q-value
            Q_sarsa[state_idx][action] += alpha * ((reward + discount_factor * Q_sarsa[next_state_idx][next_action]) - Q_sarsa[state_idx][action])

            # Update state and action        
            state = next_state
            action = next_action
                
        rewards.append(sum_of_rewards)
        avg_rewards.append(sum_of_rewards/n)
        
    return(rewards, avg_rewards, Q_sarsa)


# In[23]:


sarsa_d_rewards, sarsa_d_avg_rewards, q_values_d_sarsa = sarsa_decaying_epsilon(cliff_world, alpha, 0.5, discount_factor, 500)


# In[24]:


def qLearning_decaying_epsilon(cliff_world_environment, alpha, epsilon, discount_factor, episodes):
    Q_qLearning = np.zeros((len(states), len(actions)))
    rewards = []
    avg_rewards = []
    
    epsilon_int = epsilon
    epsilon_end = 0.1
    
    for episode in range(episodes):
        r = max((episodes-episode)/episodes, 0)
        epsilon = (epsilon_int-epsilon_end)*r + epsilon_end
        
        #Resetting the agent after the end of each episode
        state = cliff_world.reset()
        end_of_episode = False
        sum_of_rewards = 0
        n = 0

        while not end_of_episode:            
            # Choose action        
            action = epsilon_greedy(Q_qLearning, state, epsilon)

            next_state, reward, end_of_episode = cliff_world_environment.actionRewardFunction(action)
            sum_of_rewards += reward
            n += 1
            
            state_idx = state[0]*grid_world[1] + state[1]
            next_state_idx = next_state[0]*grid_world[1] + next_state[1]

            # Update q_values       
            Q_qLearning[state_idx][action] += alpha * (reward + discount_factor * np.max(Q_qLearning[next_state_idx]) - Q_qLearning[state_idx][action])

            # Update state
            state = next_state

        rewards.append(sum_of_rewards)
        avg_rewards.append(sum_of_rewards/n)
    
    return(rewards, avg_rewards, Q_qLearning)


# In[25]:


qLearning_d_rewards, qLearning_d_avg_rewards, q_values_d_qLearning = qLearning_decaying_epsilon(cliff_world, alpha, 0.5, discount_factor, 500)


# In[26]:


plt.plot(sarsa_d_rewards[1:], color='blue', label='Sarsa')
plt.plot(qLearning_d_rewards[1:], color='red', label='Q-Learning')
plt.xlabel('Epsiodes')
plt.ylabel('Sum of rewards during episode')
plt.title('For $\\alpha$ = {} , $\epsilon$ = decaying from 0.5 to 0.1 , $\delta$ = {}'.format(alpha,discount_factor))
plt.legend(loc='best')
plt.show()


# In[27]:


print("Sarsa Optimal Policy and Path\n")
sarsa_path = OptimalPolicy_and_Path(q_values_d_sarsa)
print("\n\nQ-learning Optimal Policy and Path\n")
qLearning_path = OptimalPolicy_and_Path(q_values_d_qLearning)

