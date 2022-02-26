#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import matplotlib.pyplot as plt
import time


# In[2]:


gambler_capital = np.arange(101) #Which represents the states of the MDP
discount_factor = 1 #Undiscounted task
ph = [0.25, 0.55] #probability that the coin flips heads
theta = 1e-6


# # Value-Iteration

# In[3]:


def value_iteration(states, theta, probability_of_heads):

    reward = np.zeros(len(states)) #The rewards for all the states
    reward[100] = +1 #Reward for goal state is +1 and rewards for all other states are zero
    state_values = np.zeros((len(states))) #Initializing the value functions for all the states
    policy = np.zeros((len(states))) #Initializing the policy for all the states
    
    count = 1
    total_time = 0
    
    delta = theta + 1

    while (delta > theta):
        start = time.process_time()
        delta = 0

        for s in range(1,len(states)-1): #Since state 0 (Capital = 0) and state 100 (Capital = 100) are the terminal states
            
            optimal_value = state_values[s]
            old_value = state_values[s]
            
            A = np.zeros(len(states))
            
            #Need to define the stakes or the actions taken for the corresponding capital. The minimum bet is 1 and maximum bet is min(s,100-s) 
            for a in range(1, min(s, 100-s)+1):
                
                #The state transition or the capital left after winning or losing respectively is
                gain = s + a
                loss = s - a
                
                #Since the chances are that she wins with probability of ph and loses with probability (1-ph),
                #the value sum of both the occurrences is
                 
                #Calculating the action value function since it is easier to calculate the best action
                A[a] = ((probability_of_heads)*(reward[gain]+(discount_factor*state_values[gain])) + (1-probability_of_heads)*(reward[loss]+(discount_factor*state_values[loss])))
                
                #If the improvement in the action value is more than that of theta, we need to update the actions and the values
                if((A[a] - optimal_value) > theta):
                    
                    optimal_value = A[a]
                    state_values[s] = A[a]
                    policy[s] = a

            delta = max(delta, abs(old_value - state_values[s]))
            state_values[s] = optimal_value
            
        count += 1 

        end = time.process_time()
        total_time += end-start

    time_taken = total_time/count
    
    print("\nThe number of iterations are {}".format(count))
    print('\nThe time taken to converge: {}s'.format(total_time))
        
    plt.figure(figsize = (7,7))
    plt.plot(policy)
    plt.xlabel('Current Capital', fontsize = 10)
    plt.ylabel('Stakes Chosen (Final Policy)', fontsize = 10)
    plt.show()
    
    return(state_values, policy)


# In[4]:


v1, policy1 = value_iteration(gambler_capital, theta, ph[0])


# In[5]:


plt.figure(figsize = (7,7))
x = np.arange(1,100)
y = policy1[1:100]
 
plt.bar(x, y, align='center', alpha=0.5)
 
plt.xlabel('Current Capital', fontsize = 10)
plt.ylabel('Stakes Chosen (Final Policy)', fontsize = 10)
plt.title('Current Capital vs Final Policy for ph = {}'.format(ph[0]))
plt.show()


# In[6]:


plt.figure(figsize = (7,7))
x = np.arange(1,100)
y = v1[1:100]
 
plt.plot(x, y)
 

plt.xlabel('Current Capital', fontsize = 10)
plt.ylabel('Value Estimates', fontsize = 10)
plt.title('Current Capital vs Value Estimates for ph = {}'.format(ph[0]))
plt.show()


# In[7]:


v2, policy2 = value_iteration(gambler_capital, theta, ph[1])


# In[8]:


plt.figure(figsize = (7,7))
x = np.arange(1,100)
y = policy2[1:100]
 
plt.bar(x, y, align='center', alpha=0.5)
 
plt.xlabel('Current Capital', fontsize = 10)
plt.ylabel('Stakes Chosen (Final Policy)', fontsize = 10)
plt.title('Current Capital vs Final Policy for ph = {}'.format(ph[1]))
plt.show()


# In[9]:


plt.figure(figsize = (7,7))
x = np.arange(1,100)
y = v2[1:100]
 
plt.plot(x, y)
 
plt.xlabel('Current Capital', fontsize = 10)
plt.ylabel('Value Estimates', fontsize = 10)
plt.title('Current Capital vs Value Estimates for ph = {}'.format(ph[1]))
plt.show()


# # Every Visit Monte Carlo Prediction

# In[10]:


def generate_episodes(states, probability_of_heads, pi):
    reward = np.zeros(len(states))
    reward[100] = +1
    termination_states = [states[0], states[100]]
    
    episode = []
    
    s = np.random.choice(states[1:-1])
    while True:
        a = int(pi[s])

        if(np.random.binomial(1,probability_of_heads) == 1):
            next_state = np.array(s) + np.array(a)

        else:
            next_state = np.array(s) - np.array(a)

        episode.append([s, a, reward[next_state]])

        s = next_state

        if((next_state) in termination_states):
            return(episode)            


# In[11]:


def every_visit_monte_carlo(states, pi, probability_of_heads, num_episodes):
    #pi = policy to be evaluated
    
    state_values = np.zeros((len(states)))
    
    num_visits = np.zeros(len(states)) #Initializing the number of visits to a state
    returns = {new_list: [] for new_list in range(len(states))} #Initializing the returns of a state
    
    theta = 1e-6
    delta = theta + 1
    
    for iterations in range(1, num_episodes):
        
        episode = generate_episodes(states, probability_of_heads, pi)
        G = 0

        for output in episode[::-1]:

            G = discount_factor*G + output[2]
            s = (output[0])
            returns[s].append(G)
            num_visits[s] += 1
            value = np.sum(returns[s])/num_visits[s]
        
            state_values[s] = value

    return(state_values)


# In[12]:


v_mc = every_visit_monte_carlo(gambler_capital, policy2, ph[1], 1000)


# In[13]:


plt.figure(figsize = (7,7))
x = np.arange(1,100)
y = v_mc[1:100]
 
plt.plot(x, y)
 
plt.xlabel('Current Capital', fontsize = 10)
plt.ylabel('Value Estimates', fontsize = 10)
plt.title('Current Capital vs Value Estimates for ph = {}'.format(ph[1]))
plt.show()

