---
layout: default
title:  Status
---
Video: https://youtu.be/rdak83dwi1w

# Project Summary
The purpose of our AI is to act as a body guard for a specific target. The target may be a user, character, or structure. Our scenario consists of having a villager NPC guarded by our AI from zombies. The AI will be trained through reinforcement learning, being awarded positively for the duration it is able to keep the target alive and negatively when the target perishes. The eventual goal is to be able to have the AI act as a sentry and shield a point of interest from harm for a user, allowing said user to tend to other matters while either preventing or at least delaying the target's demise.

# Approach
For our project we are using Reinforcement Learning, particularly Q-Learning and implementing Q-SARSA as our updating function.
In the begining of our project we began with an approach of using a q_table to learn off of, however with such as large state space
we have now chosen to use function approximation instead, using a Stochastic Gradient Descent Regressor. Our update function is again of a
generic type and is as follows:
```
q_values_current = estimator.predict(s_prime)
td_target = current_r + self.gamma * np.argmax(q_values_current)
```
We predict what the greatest reward would be from the previous state to get the q_values_current. Then we multiply the max q_values_current by gamma, our discount factor, with this
we then add current_r, our current_reward. This gives us the current reward for our state space.

Our Markov Decision Process consists of our state, player world, as the nodes and our actions are the edges, that connect in betwen different states. Currently we have four features in our state space as follows:
1. Absolute distance between Villager x-coordinate and Protector x-coordinate.
2. Absolute distance between Villager x-coordinate and Zombie x-coordinate.
3. Absolute distance between Villager z-coordinate and Protector z-coordinate.
4. Absolute distance between Villager z-coordinate and Zombie z-coordinate.

This is a fairly large state space and is harder to esitmate in size as each feature is held as a float. This is smaller however than the previous feature set that we had created.
This old state space consisted of 7 different features, is_villager_alive, villager(x,y,z), and zombie(x,y,z). We realized that about 3 of these features were superfluous and deciced
to shrink this to only have 4 state space features. We currently have 5 actions to choose from also consisting of up, down, left, right, and swing, swing the sword.

Our reward function has a couple of considerations:
1. Is the Protectee dead, if so large negative reward, AI failed
2. Is the Zombie dead, if so large positive reward, AI did well
3. Is the Protector getting closer to the Zombie, if so give slightly negative reward
4. Is the Protector getting closer to the Protectee, if so give a slightly negative reward
5. Is the health of the Protectee lowering, if so give negative reward

With this reward function we have had our AI begin to run towards the rescue of the Protectee, we are now begining to get it in range of attacking the zombie. The attacks are sporadic at this point,
however we are looking into ways to have better directed attacks.

As mentioned earlier in the evaluation section we are now using an value approximation function, through the use of a Stochastic Gradient Descent Regressor built into sklearn.
We are using this as it is faster to train on a larger state space as it does not take into account each state and hold each state but come up with an approximation of what an
optimal state would be. This has resulted so far in slightly better training. There is still more work to be done though through feature opimization and a possible improvement in
our reward function.

# Evaluation
Our project is still a work in progress. While not ideal, we have been able to get our AI to find its way towards villager and zombie. Initially, the AI moves around in what seems like random directions, often away from the villager as it is being attack by the zombie. However, it has been able to learn to improve its score by decreasing the distance between itself and the other entities. Although it does not consistently attack the zombie, our AI is at least able to move to the location of the target. After over a hundred episodes, the AI has shown much improvement over its initial movement patterns, consistently heading towards the target over several consecutive episodes. 

![Distance Plot](https://github.com/IanSchweer/CS175/blob/master/docs/distance.png)
The image above is a plot of the average change in distance (between player and zombie) between actions over 300 episodes. Towards the earlier episodes, the AI has a much higher positive average, indicating that the AI is moving farther from the zombie as well as the villager it is attacking. However, we can see that there is improvement after each episode. The averages eventually become negative, which shows that the AI is able to decrease the distance between it and the zombie with each action over time.

![Time Plot](https://github.com/IanSchweer/CS175/blob/master/docs/time.png)


# Remaining Goals and Challenges
