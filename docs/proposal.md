---
layout: default
title: Proposal
---

## Summary
### Proposal Idea:
The purpose of our AI is to act as a body guard for a specified target. This target may be a user, character, or structure, but for our purposes we will use a sheep. The AI will protect the sheep from all surrounding enemies. We will be using reinforcement learning to train our AI with a reward being the time that our protected target is alive and the negative reward being the destruction of said target. The purposes of this AI can be protecting an asset(house/mine/village), protecting a being while harvesting, or acting as a century to protect a general point of interest.

### AI/ML Algorithms:
We will be using reinforcement learning for training our AI with a starting point of using Monte Carlo Policy Evaluation.

### Evaluation Plan:
For our evaluation we will be using a reward system consisting of the status of our intended target(Dead/Alive/Not Destroyed), the status of our character(how much health), and the spacing between our character, target and the possible threats. Our sanity case will be to keep a zombie from killing the target sheep and our character. Our secondary case will be to keep a small group of say 3 zombies from killing the target, and the moonshot case will be to have the character fight off a mob of zombies from a zombie spawner.

Our first metric that we will be analyzing for our AI is how long it is able to keep the target sheep alive and how long it can keep our character alive with a heavier weight on keeping the sheep alive. The second metric once we can keep both alive while killing the enemy is how many enemies can our AI fight off. If these do not hold fruitful our backup metric for success will be how far can we keep the zombie away from the sheep. For all of these we will make a small visualizing log to see if these metrics are growing or receding with each iteration of our learning phase.

