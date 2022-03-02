# Reinforcement-Learning

1. Grid-World Environment:
In this problem, implemented an iterative policy evaluation and improvement on the running Grid-World example. Consider the gridworld environment shown below (with almost everything the same as the previous homework): the agent can move in one of four directions fU;D; L;Rg. The agent has reward of -1 each time it moves, gets a reward +10 after reaching the goal, and a reward -10 after entering the trap. Entering either the goal or the trap state ends the episode. The agent stays in place when hitting the
walls or the block in the middle of the grid. The discount factor is set to 𝛿 = 1.

![image](https://user-images.githubusercontent.com/42225976/156284379-989361b7-833d-46be-8cec-9c40e6a27d7a.png)

Consider the random policy: at each state, the agent moves in one of four directions, each with equal probability.

![image](https://user-images.githubusercontent.com/42225976/156287980-b1735009-cc74-44c3-b1f7-287a474b506f.png)

The greedy policy for the iterations 2, 4, 6, 8, 10 iterations, after finding the respective value functions with the aid of the policy improvement theorem are the same. The greedy policies are the same for all the iterations. Considering the mapping of {U, D, L, R} i.e., the Up, Down, Left and Right actions respectively, the results of the implementations of part (c) translated to the respective actions of the state are:
State 0 - U
State 1 - U
State 2 - R
State 3 - L
State 4 - R
State 5 - L
State 6 - U
State 7 - R
State 8 - L

Since the goal and the trap state always transitions to terminal state, S∞.

2. Gambler's problem
Problem statement: a gambler repeatedly places bets on the outcomes of a sequence of coin
flips. If the flip is heads, she wins as many dollars as she has bet on that flip; if it is tails, she loses all the bet. The game ends when the gambler reaches her goal of making $100, or when she runs out of all her money. At each time, the gambler chooses how much of her money to bet on the next flip.
This situation can be formulated as an undiscounted, episodic, finite MDP. The state is the
gambler's capital, s ∈ {0, 1, 2, . . . , 99, 100}, and her actions are stakes (i.e., how much to bet) a ∈ {0, 1, . . . , min{s, 100 - s}}. The rewards are zero for all state transitions, except when the transition leads to the gambler reaching her $100 goal, in which case the reward is +1. Let ph be the probability that the coin flips heads.
The results of the implementation using different methodologies are,

![image](https://user-images.githubusercontent.com/42225976/156296421-6389bd51-3db8-4d29-b841-dc5aad637234.png)

![image](https://user-images.githubusercontent.com/42225976/156296991-cc5271ab-45be-45f0-9b33-cdca7f3f2dc1.png)

![image](https://user-images.githubusercontent.com/42225976/156297037-d6e032a5-c478-435f-ad42-9f3e9942e227.png)

![image](https://user-images.githubusercontent.com/42225976/156297071-58d95109-c29a-402c-962a-1f48d6edf2fe.png)

![image](https://user-images.githubusercontent.com/42225976/156297170-c057c0ee-0a42-4e78-b7ea-6407f5b4ad0c.png)

3. Implemented Q-learning and SARSA on the Cliff Walking problem. This gridworld example compares Sarsa and Q-learning, highlighting the di↵erence between on-policy (Sarsa) and o↵-policy (Q-learning) methods.
Problem statement: Consider the gridworld shown below. This is a standard undiscounted, episodic task, with start and goal states, and the usual actions causing movement up, down, right, and left. Reward is −1 on all transitions except those into the region marked “The Cliff.” Stepping into this region incurs a reward of −100 and sends the agent instantly back to the start.

![image](https://user-images.githubusercontent.com/42225976/156297787-cfe3ed9d-a11a-4069-a247-d4c89ee62f8a.png)

The graph to the right shows the performance of the Sarsa and Qlearning methods with ϵ-greedy action selection, ϵ = 0.1. After an initial transient, Q-learning learns values for the optimal policy, that which travels right along the edge of the cliff. Unfortunately, this results in its occasionally falling off of the cliff because of the ϵ-greedy action
selection. Sarsa, on the other hand, takes the action selection into account and learns the longer but safer path through the upper part of the grid. Although Q-learning actually learns the values of the optimal policy, its online performance is worse than that of Sarsa, which learns the roundabout policy. Of course, if " were gradually reduced, then both methods would asymptotically converge to the optimal policy.

![image](https://user-images.githubusercontent.com/42225976/156298036-c903f44d-1877-42f9-8ab4-230a1435d042.png)

