import random
import numpy as np

items=['pumpkin', 'sugar', 'egg', 'egg', 'red_mushroom', 'planks', 'planks']

food_recipes = {'pumpkin_pie': ['pumpkin', 'egg', 'sugar'],
                'pumpkin_seeds': ['pumpkin'],
                'bowl': ['planks', 'planks'],
                'mushroom_stew': ['bowl', 'red_mushroom']}


rewards_map = {'pumpkin': -5, 'egg': -25, 'sugar': -10,
               'pumpkin_pie': 100, 'pumpkin_seeds': -50,
               'red_mushroom': 5, 'planks': -5, 'bowl': -1,
               'mushroom_stew': 100}

def is_solution(reward):
    # return reward == 100 for part one
    return reward >= 200

def get_curr_state(items):
    state = ""

    # to enforce order not mattering, we need to sort
    items.sort()
    for item in items:
        for i in range(item[1]):
            state = state + "," + item[0]
    return state

def choose_action(curr_state, possible_actions, eps, q_table):
    rnd = random.random()
    if (rnd <= eps):
        return possible_actions[random.randint(0, len(possible_actions) - 1)]
    else:
        # we will get in here with prob 1 - eps
        max = []
        list = q_table[curr_state].items()
        argmax = np.argmax([x[1] for x in list])
        argmax = list[argmax] if argmax < len(list) else list[0]
        for i in range(len(list)):
            if list[i][1] == argmax[1]:
                max.append(i)
        index = max[random.randint(0, len(max) - 1)]
        return list[index][0]