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
