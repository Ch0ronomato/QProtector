import MalmoPython
import os
import sys
import time
import json
import random
import math
import Tkinter as tk
import numpy as np
from collections import deque

MOB_TYPE = "Endermite"  # Change for fun, but note that spawning conditions have to be correct - eg spiders will require darker conditions.
ARENA_WIDTH = 60
ARENA_BREADTH = 60

PROTECTEE = 'Villager'
ENEMY = 'Zombie'
def getCorner(index,top,left,expand=0,y=206):
    ''' Return part of the XML string that defines the requested corner'''
    x = str(-(expand+ARENA_WIDTH/2)) if left else str(expand+ARENA_WIDTH/2)
    z = str(-(expand+ARENA_BREADTH/2)) if top else str(expand+ARENA_BREADTH/2)
    return 'x'+index+'="'+x+'" y'+index+'="' +str(y)+'" z'+index+'="'+z+'"'

class Protector(object):
    def __init__(self, alpha=0.3, gamma=1, n=1):
        """Constructing an RL agent.

        Args
            alpha:  <float>  learning rate      (default = 0.3)
            gamma:  <float>  value decay rate   (default = 1)
            n:      <int>    number of back steps to update (default = 1)
        """
        self.epsilon = 0.2  # chance of taking a random action instead of the best
        self.q_table = {}
        self.n, self.gamma, self.alpha = n, gamma, alpha

    def position_states(self, agent_host):
        """Returns the states of all the entities [px, pz, pyaw, vx, vz, vyaw, (zx, zz, zyaw)*]"""
        state = []
        order = ['The Hunted', 'Villager', 'Zombie']
        while True:
            world_state = agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                msg = world_state.observations[-1].text
                ob = json.loads(msg)
                for ent in ob['entities']:
                    name = ent['name']
                    for next in order:
                        if name == next:
                            state.extend([str(x) for x in [ent['x'], ent['z'], ent['yaw']]])
                return ','.join(state)

    def is_solution(reward):
        """If the reward equals to the maximum reward possible returns True, False otherwise. """
        return reward == 10000

    def get_possible_actions(self, agent_host, is_first_action=False):
        """Returns all possible actions that can be done at the current state. """
        return ['up', 'down', 'left', 'right', 'swing']

    def get_curr_state(self, agent_host):
        """Creates a unique identifier for a state.

        The state is defined as the items in the agent inventory. Notice that the state has to be sorted -- otherwise
        differnt order in the inventory will be different states.
        """
        return self.position_states(agent_host)

    def choose_action(self, curr_state, possible_actions, eps):
        """Chooses an action according to eps-greedy policy. """
        if curr_state not in self.q_table:
            self.q_table[curr_state] = {}
        for action in possible_actions:
            if action not in self.q_table[curr_state]:
                self.q_table[curr_state][action] = 0

        rnd = random.random()
        if rnd <= eps:
            return possible_actions[random.randint(0, len(possible_actions) - 1)]
        else:
            # we will get in here with prob 1 - eps
            max = []
            list = self.q_table[curr_state].items()
            argmax = np.argmax([x[1] for x in list])
            argmax = list[argmax] if argmax < len(list) else list[0]
            for i in range(len(list)):
                if list[i][1] == argmax[1]:
                    max.append(i)
            index = max[random.randint(0, len(max) - 1)]
            return list[index][0]

    def act(self, agent_host, action):
        print action + ",",
        # if action == 'present_gift':
        #     return self.present_gift(agent_host)
        # elif action.startswith('c_'):
        #     self.craft_item(agent_host, action[2:])
        # else:
        #     self.fetch_item(agent_host, action)
        if not action == 'swing':
            self.move(action, agent_host)
        else:
            self.attack(agent_host)

        return self.score(agent_host)

    def move(self, action, agent_host):
        if action == 'up':
            agent_host.sendCommand('move 1')
        elif action == 'down':
            agent_host.sendCommand('move -1')
        elif action == 'left':
            agent_host.sendCommand('strafe 1')
        elif action == 'right':
            agent_host.sendCommand('strafe -1')

    def attack(self, agent_host):
        agent_host.sendCommand('attack 1')

    def score(self, agent_host):
        '''
        A dummy reward function. High if no zombies, but villager is alive, very negative if the opposite
        :param agent_host: 
        :return: number score (see above)
        '''
        if not self.is_villager_alive(agent_host):
            return -10000
        if not self.has_zombies(agent_host):
            return 10000
        return -1

    def update_q_table(self, tau, S, A, R, T):
        """Performs relevant updates for state tau.

        Args
            tau: <int>  state index to update
            S:   <dequqe>   states queue
            A:   <dequqe>   actions queue
            R:   <dequqe>   rewards queue
            T:   <int>      terminating state index
        """
        curr_s, curr_a, curr_r = S.popleft(), A.popleft(), R.popleft()
        G = sum([self.gamma ** i * R[i] for i in range(len(S))])
        if tau + self.n < T:
            G += self.gamma ** self.n * self.q_table[S[-1]][A[-1]]

        old_q = self.q_table[curr_s][curr_a]
        self.q_table[curr_s][curr_a] = old_q + self.alpha * (G - old_q)

    def run(self, agent_host):
        S, A, R = deque(), deque(), deque()
        present_reward = 0
        done_update = False
        agent_host.sendCommand("hotbar.9 1")  # Press the hotbar key
        agent_host.sendCommand("hotbar.9 0")  # Release hotbar key - agent should now be holding diamond_pickaxe
        while not done_update:
            s0 = self.get_curr_state(agent_host)
            possible_actions = self.get_possible_actions(agent_host, True)
            a0 = self.choose_action(s0, possible_actions, self.epsilon)
            S.append(s0)
            A.append(a0)
            R.append(0)

            T = sys.maxint
            for t in xrange(sys.maxint):
                time.sleep(0.1)
                if t < T:
                    current_r = self.act(agent_host, A[-1])
                    R.append(current_r)

                    if current_r == -10000:
                        # Terminating state
                        T = t + 1
                        S.append('Term State')
                        present_reward = current_r
                        print "Reward:", present_reward
                        agent_host.sendCommand("quit")
                    else:
                        s = self.get_curr_state(agent_host)
                        S.append(s)
                        possible_actions = self.get_possible_actions(agent_host)
                        next_a = self.choose_action(s, possible_actions, self.epsilon)
                        A.append(next_a)

                tau = t - self.n + 1
                if tau >= 0:
                    self.update_q_table(tau, S, A, R, T)

                if tau == T - 1:
                    while len(S) > 1:
                        tau = tau + 1
                        self.update_q_table(tau, S, A, R, T)
                    done_update = True
                    break

    def has_zombies(self, agent_host):
        while True:
            world_state = agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                msg = world_state.observations[-1].text
                ob = json.loads(msg)
                for entity in ob['entities']:
                    if entity['name'] == ENEMY:
                        return True
                return False

    def is_villager_alive(self, agent_host):
        while True:
            world_state = agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                msg = world_state.observations[-1].text
                ob = json.loads(msg)
                for entity in ob['entities']:
                    if entity['name'] == PROTECTEE:
                        return True
                return False

# exp_num_steps = float('inf')
# converged = False
# while not converged:
#     S, A = deque(), deque()
#     S0 = random.randint(2, 9)  # Sampling the starting point
#     S.append(S0)
#     A0 = choose_action_idx(actions, q_table[S0])
#     A.append(A0)
#     T = sys.maxint
#     for t in xrange(sys.maxint):  # What will happen if you'll use range instead of xrange
#         if t < T:
#             # Taking action At
#             next_s = S[-1] + actions[A[-1]]
#             S.append(next_s)
#             next_r = Rs[next_s]
#
#             # Checking if it's in terminate states
#             if next_s in Ts:
#                 T = t + 1
#             else:
#                 next_a = choose_action_idx(actions, q_table[next_s])
#                 A.append(next_a)
#
#         tau = t - n + 1
#         if tau >= 0:
#             curr_s, curr_a = S.popleft(), A.popleft()
#             G = sum([gamma ** i * Rs[S[i]] for i in range(len(S))])
#             if tau + n < T:
#                 G += gamma ** n * q_table[S[-1]][A[-1]]
#
#             old_q = q_table[curr_s][curr_a]
#             q_table[curr_s][curr_a] = old_q + alpha * (G - old_q)
#
#         if tau == T - 1:
#             break
#
#     # Checking for convergance.
#     tmp = expected_num_steps(2, q_table, actions)
#     if np.abs(exp_num_steps - tmp) < 0.0001:
#         converged = True


def GetMissionXML(summary):
    ''' Build an XML mission string.'''
    return '''<?xml version="1.0" encoding="UTF-8" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <About>
            <Summary>''' + summary + '''</Summary>
        </About>

        <ModSettings>
            <MsPerTick>20</MsPerTick>
        </ModSettings>
        <ServerSection>
            <ServerInitialConditions>
                <Time>
                    <StartTime>17000</StartTime>
                    <AllowPassageOfTime>true</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
            </ServerInitialConditions>
            <ServerHandlers>
                <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1" />
                <DrawingDecorator>
                    <DrawCuboid ''' + getCorner("1",True,True,expand=1) + " " + getCorner("2",False,False,y=226,expand=1) + ''' type="stone"/>
                    <DrawCuboid ''' + getCorner("1",True,True,y=207) + " " + getCorner("2",False,False,y=226) + ''' type="air"/>

                    <DrawEntity x="20.5" y="207.0" z="20.5" type="Zombie" />
                </DrawingDecorator>
                <ServerQuitWhenAnyAgentFinishes />
                <ServerQuitFromTimeUp timeLimitMs="30000"/>
            </ServerHandlers>
        </ServerSection>

        <AgentSection mode="Survival">
            <Name>The Hunted</Name>
            <AgentStart>
                <Placement x="12.5" y="207.0" z="10.5" yaw="-90"/>
                <Inventory>
                        <InventoryItem slot="8" type="diamond_pickaxe"/>
                </Inventory>                
            </AgentStart>
            <AgentHandlers>
                <ChatCommands/>
                <ContinuousMovementCommands turnSpeedDegs="360"/>
                <MissionQuitCommands/>
                <InventoryCommands/>
                <AbsoluteMovementCommands/>
                <ObservationFromNearbyEntities>
                    <Range name="entities" xrange="'''+str(ARENA_WIDTH)+'''" yrange="2" zrange="'''+str(ARENA_BREADTH)+'''" />
                </ObservationFromNearbyEntities>
                <ObservationFromFullStats/>
            </AgentHandlers>
        </AgentSection>

    </Mission>'''


if __name__ == '__main__':
    random.seed(0)
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately

    expected_reward = 3390
    my_client_pool = MalmoPython.ClientPool()
    my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))

    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse(sys.argv)
    except RuntimeError as e:
        print 'ERROR:', e
        print agent_host.getUsage()
        exit(1)
    if agent_host.receivedArgument("help"):
        print agent_host.getUsage()
        exit(0)

    num_reps = 30000
    n,gamma,alpha = 10, 5,.7
    ai = Protector(alpha, gamma, n)
    for iRepeat in range(num_reps):
        my_mission = MalmoPython.MissionSpec(GetMissionXML("Defend #" + str(iRepeat)), True)
        my_mission_record = MalmoPython.MissionRecordSpec()  # Records nothing by default
        my_mission.requestVideo(800, 500)
        my_mission.setViewpoint(0)
        max_retries = 3
        for retry in range(max_retries):
            try:
                # Attempt to start the mission:
                agent_host.startMission(my_mission, my_client_pool, my_mission_record, 0, "Defense")
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print "Error starting mission", e
                    print "Is the game running?"
                    exit(1)
                else:
                    time.sleep(2)

        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
        print "started"
        agent_host.sendCommand("chat /summon Villager ~10 ~ ~ {NoAI:1}")
        agent_host.sendCommand("hotbar.9 1")  # Press the hotbar key
        agent_host.sendCommand("hotbar.9 0")  # Release hotbar key - agent should now be holding diamond_pickaxe
        time.sleep(.1)
        while world_state.is_mission_running:
            ai.run(agent_host)
            time.sleep(0.1)
            world_state = agent_host.getWorldState()

        # Every few iteration Chester will show us the best policy that he learned.
        # if (iRepeat + 1) % 5 == 0:
        #     print (iRepeat+1), 'Showing best policy:',
        #     found_solution = chester.best_policy(agent_host)
        #     if found_solution:
        #         print 'Found solution'
        #         print 'Done'
        #         break
        # else:
        #     print (iRepeat+1), 'Learning Q-Table:',
        #     chester.run(agent_host)
        #
        # chester.clear_inventory()
        time.sleep(1)
