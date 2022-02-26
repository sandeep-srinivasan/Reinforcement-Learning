#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import matplotlib.pyplot as plt
import time


# # Part (a)

# In[2]:


actions = np.array([[-1,0], [1,0], [0,-1], [0,1]]) # {U,D,L,R} which can also be represented as {0,1,2,3} indices
discount_factor = 1 #discount factor
grid_world = np.array([3,4])
start_state = np.array([2,0])
goal = np.array([0,3])
trap = np.array([1,3])
block = np.array([1,1])
states = [[grid_world[0]-1-i, j] for j in range(grid_world[1]) for i in range(grid_world[0])]
iterations = 1000
random_policy = np.ones([len(states), len(actions)]) / len(actions) #probabilities of taking the action
probs = 0.25 #state transition probabilities


# In[3]:


def actionRewardFunction(initialPosition, action):
        
    if(np.array_equal(initialPosition, goal)):
        finalPosition = 'terminal_state' #Representation like a terminal state
        reward = +10
        return(finalPosition, reward)
    
    elif(np.array_equal(initialPosition, trap)):
        finalPosition = 'terminal_state' #Representation like a terminal state
        reward = -10
        return(finalPosition, reward)
    
    elif(np.array_equal(initialPosition, block)):
        reward = 0
        finalPosition = initialPosition
        return(finalPosition, reward)
    
    else:
        finalPosition = np.array(initialPosition) + np.array(action)
        reward = -1
        
        if (finalPosition[0] >= 0) and (finalPosition[0] <= 2):
            if (finalPosition[1] >= 0) and (finalPosition[1] <= 3):
                if (np.array_equal(finalPosition, block)):
                    finalPosition = initialPosition
                else:
                    return(finalPosition, reward)
      
        finalPosition = initialPosition

        return(finalPosition, reward)


# In[4]:


def iterative_policy_evaluation(grid_world_environment, policy, theta):

    total_time = 0
    state_values = np.zeros((grid_world_environment))
    delta = theta + 1
    count = 0
    values = []

    while (delta > theta):
        start = time.process_time()
        delta = 0

        for s in range(len(states)):
            value = 0

            for a in range(len(actions)):
                finalPosition, reward = actionRewardFunction(states[s], actions[a])
                if (np.array_equal(finalPosition,'terminal_state')):
                    value += (policy[s][a]*1)*(reward+(discount_factor*0)) # Since the agent loops in the terminal state and stays in place
                    
                else:
                    value += (policy[s][a]*probs)*(reward+(discount_factor*state_values[finalPosition[0], finalPosition[1]])) 

            delta = max(delta, np.abs(value - state_values[states[s][0], states[s][1]]))
            state_values[states[s][0], states[s][1]] = value

        values.append(state_values)
        count += 1 

        end = time.process_time()
        total_time += end-start

    time_taken = total_time/count
    
    print("\nThe number of iterations are {}".format(count))
    print('\nThe time taken to converge: {}s'.format(total_time))
    
    return(state_values, values)


# In[5]:


theta = 1e-6

V, total_values = iterative_policy_evaluation(grid_world, random_policy, theta)

print('\nThe Values of this policy are:\n',V)


# # Part (b)

# In[6]:


random.shuffle(states)
print(states)


# In[7]:


theta = 1e-6

V, total_values = iterative_policy_evaluation(grid_world, random_policy, theta)

print('\nThe Values of this policy are:\n',V)


# In[8]:


states = [[grid_world[0]-1-i, j] for i in range(grid_world[0]) for j in range(grid_world[1])]


# In[9]:


theta = 1e-6

V, total_values = iterative_policy_evaluation(grid_world, random_policy, theta)

print('\nThe Values of this policy are:\n',V)


# # Part (c)

# In[10]:


states = [[grid_world[0]-1-i, j] for j in range(grid_world[1]) for i in range(grid_world[0])]


# In[11]:


def policy_improvement(grid_world_environment, value_at_k):
    # Initiallize a policy arbitarily
    policy = np.ones([len(states), len(actions)]) / len(actions)
    
    while (True):
        # Compute the Value Function for the current policy
        V = value_at_k
        
        # Will be set to false if we update the policy
        policy_stable = True
        
        # Improve the policy at each state
        for s in range(len(states)):
            # The best action we would take under the currect policy
            chosen_a = np.argmax(policy[s])
            # Find the best action by one-step lookahead
            action_values = np.zeros(len(actions))
            for a in range(len(actions)):
                finalPosition, reward = actionRewardFunction(states[s], actions[a])
                if (np.array_equal(finalPosition,'terminal_state')):
                    action_values[a] += (1)*(reward+(discount_factor*0)) 
                else:
                    action_values[a] += (probs)*(reward+(discount_factor*V[finalPosition[0], finalPosition[1]])) 
                    
            best_a = np.argmax(action_values)
            
            # Greedily (max in the above line) update the policy
            if chosen_a != best_a:
                policy_stable = False
            else:
                policy_stable = True
                
            policy[s] = np.eye(len(actions))[best_a]
        
        # Until we've found an optimal policy. Return it
        if policy_stable:
            return policy, V


# In[12]:


print('For the states:\n', states)

for k in [1,3,5,7,len(total_values)-1]:
    policy, V = policy_improvement(grid_world, total_values[k])
    
    #Mapping of states is {U,D,L,R}
    print('\nThe greedy policy at iteration {} is \n{}'.format(k+1, policy))

