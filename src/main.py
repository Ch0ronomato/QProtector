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
import matplotlib
import numpy as np
import sklearn
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler


MOB_TYPE = "Endermite"  # Change for fun, but note that spawning conditions have to be correct - eg spiders will require darker conditions.
ARENA_WIDTH = 60
ARENA_BREADTH = 60

PROTECTEE = 'Villager'
ENEMY = 'Zombie'

ACTIONS = ['up', 'down', 'turn-left', 'turn-right', 'swing']

def getCorner(index,top,left,expand=0,y=206):
    ''' Return part of the XML string that defines the requested corner'''
    x = str(-(expand+ARENA_WIDTH/2)) if left else str(expand+ARENA_WIDTH/2)
    z = str(-(expand+ARENA_BREADTH/2)) if top else str(expand+ARENA_BREADTH/2)
    return 'x'+index+'="'+x+'" y'+index+'="' +str(y)+'" z'+index+'="'+z+'"'



scaler = sklearn.preprocessing.StandardScaler()
intitial_state_space = np.zeros(6)
scaler.fit(intitial_state_space)

featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(intitial_state_space))

# estimator for SDGRegressor
class Estimator():
    """
    Value Function approximator.
    """

    def __init__(self):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.models = []
        for _ in range(5):
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            start_space = np.zeros(6)
            model.partial_fit([self.featurize_state(start_space)], [0])
            self.models.append(model)

    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = scaler.transform([state])
        featurized = featurizer.transform(scaled)
        return featurized[0]

    def predict(self, s, a=None):
        """
        Makes value function predictions.

        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for

        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.

        """
        features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]

    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])




class Protector(object):
    def __init__(self, alpha=0.3, gamma=1, n=1, estimator=Estimator()):
        """Constructing an RL agent.

        Args
            alpha:  <float>  learning rate      (default = 0.3)
            gamma:  <float>  value decay rate   (default = 1)
            n:      <int>    number of back steps to update (default = 1)
        """
        self.epsilon = 0.2  # chance of taking a random action instead of the best
        self.q_table = {}
        self.n, self.gamma, self.alpha, self.estimator = n, gamma, alpha, estimator
        self.reward_file = file("reward.txt", 'a+')
        self.time_file = file("time.txt", 'a+')
        self.distance_file = file("distance.txt", 'a+')
        self.distances = []

    def get_entities(self, agent_host):
        running = True
        while running:
            world_state = agent_host.getWorldState()
            running = world_state.is_mission_running
            if world_state.number_of_observations_since_last_state > 0:
                msg = world_state.observations[-1].text
                ob = json.loads(msg)
                return ob['entities']
        return []

    def position_states(self, entities):
        """
            Returns the states of all the entities [villager_alive, px, pz, pyaw,(zx, zz, zyaw)*]
            all positions relative
        """
        enemystate = []
        mystate = []
        seen_villager, x, z, yaw = False,0.,0.,0.
        health = 0 #Villager's health
        for ent in entities:
            name = ent['name']
            if name == PROTECTEE:
                x, z, health = ent['x'], ent['z'], ent['life']
                #seen_villager = True
            elif name == ENEMY:
                enemystate.extend([str(elem) for elem in [ent['x'] - x, ent['z'] - z, ent['life']]])
            elif name == 'The Hunted':
                mystate = [str(elem) for elem in [ent['x'] - x, ent['z'] - z, health]]
        x = []
        # print "VILLAGER_HP: ", health
        # print "mystate: ", mystate
        # print "enemystate: ", enemystate
        # print "x before extend: ", x

        if len(enemystate) > 3:
            enemystate = [0.0, 0.0, 0.0]
        x.extend(mystate)
        x.extend(enemystate)
        return x

    def is_solution(reward):
        """If the reward equals to the maximum reward possible returns True, False otherwise. """
        return reward == 100

    def get_possible_actions(self):
        """Returns all possible actions that can be done at the current state. """
        return ['up', 'down', 'turn-left', 'turn-right', 'swing']

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

    def act(self, agent_host, action, entities):
        print "===", action, "==="

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
        return self.score(entities, action)

    def move(self, action, agent_host):
        if action == 'up':
            agent_host.sendCommand('move 1')
        elif action == 'down':
            agent_host.sendCommand('move -1')
        elif action == 'left':
            agent_host.sendCommand('strafe 1')
        elif action == 'right':
            agent_host.sendCommand('strafe -1')
        elif action == 'turn-right':
            agent_host.sendCommand('turn 1')
        elif action == 'turn-left':
            agent_host.sendCommand('turn -1')

    def attack(self, agent_host):
        agent_host.sendCommand('attack 1')

    def score(self, entities, action):
        '''
        A dummy reward function. High if no zombies, but villager is alive, very negative if the opposite
        :param entities:
        :return: number score (see above)
        '''
        score = -1
        state = self.position_states(entities)
        zx, zz = 0., 0.
        vx, vz = 0., 0.

        for ent in entities:
            if ent['name'] == PROTECTEE:
                #Negative reward based on damage villager has taken
                score -= (20 - ent['life'])
                vx -= ent['x']
                vz -= ent['z']
            elif ent['name'] == ENEMY:
                #Positive reward based on damage to the zombie
                score -= (ent['life'] - 20)*2
                zx -= ent['x']
                zz -= ent['z']
            elif ent['name'] == 'The Hunted':
                zx += ent['x']
                zz += ent['z']
                vx += ent['x']
                vz += ent['z']

        zdistance = math.sqrt((zx**2) + (zz**2))
        vdistance = math.sqrt((vx**2) + (vz**2))
        # save the episode's distance between player and zombie
        self.distances.append(zdistance)
        print "ZOMB DISTANCE:", zdistance
        print "VILL DISTANCE:", vdistance

        score -= abs(zx)
        score -= abs(zz)
        score -= abs(vx)
        score -= abs(vz)

        #score -= zdistance
        #score -= vdistance * 0.5

        if action in ['up', 'down']:
            score -= 1
        elif action in ['turn-left', 'turn-right']:
            score -= 1
        elif action == 'swing':
            score -= 5

        if not self.has(entities, PROTECTEE):
           score -= 100
        if not self.has(entities, ENEMY):
            score += 100
        return score

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

        # Episode start time
        start_time = time.time()

        while not done_update:
            entities = self.get_entities(agent_host)
            s0 = self.get_curr_state(entities)
            possible_actions = self.get_possible_actions()
            # a0 = self.choose_action(s0, possible_actions, self.epsilon)
            a0 = 'swing'
            S.append(s0)
            A.append(a0)
            R.append(0)
            entities = self.get_entities(agent_host)

            T = sys.maxint
            for t in xrange(sys.maxint):
                world_state = agent_host.getWorldState()
                policy = make_epsilon_greedy_policy(
                    self.estimator, self.epsilon * self.gamma**t, len(possible_actions))
                time.sleep(0.1)
                if t < T:
                    current_r = self.act(agent_host, A[-1], entities)
                    entities = self.get_entities(agent_host)
                    R.append(current_r)
                    last_state = S[-1]
                    s = self.get_curr_state(entities)

                    print "Took action ", A[-1], " at state ", S[-1], " got reward ", current_r
                    if current_r <= -100 or current_r >= 100 or not world_state.is_mission_running:
                        # Terminating state
                        print " <= -100 got triggered"
                        T = t + 1
                        S.append('Term State')
                        present_reward = current_r
                        R.append(current_r)
                        print "Reward:", present_reward
                        print self.q_table
                        agent_host.sendCommand("quit")

                    else:
                        S.append(s)
                        action_taken = A[-1]
                        s_prime = self.get_curr_state(self.get_entities(agent_host))

                        # picking next action
                        possible_actions = self.get_possible_actions()
                        if len(s_prime) == 0:
                            break
                        action_probs = policy(s_prime)
                        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

                        A.append(ACTIONS[action])

                        # TD Update
                        q_values_current = estimator.predict(s_prime)
                        td_target = current_r + self.gamma * np.argmax(q_values_current)

                        # Update the function approximator using our target
                        print "Updating estimator for state ", s, " action ", action_taken, " target ", td_target
                        estimator.update(last_state, possible_actions.index(action_taken), td_target)


                tau = t - self.n + 1
                # if tau >= 0:
                #     self.update_q_table(tau, S, A, R, T)

                if tau == T - 1:
                    # while len(S) > 1:
                    #     tau = tau + 1
                        # self.update_q_table(tau, S, A, R, T)
                    done_update = True
                    break
        # Episode end time
        end_time = time.time() - start_time
        # dump episode time to a file
        self.time_file.write(str(end_time) + '\n')
        # dump the reward function to a file.
        self.reward_file.write(','.join([str(r) for r in R]) + '\n')
        # Average the difference of distances between each episode
        differences = [self.distances[i+1] - self.distances[i] for i in range(len(self.distances) - 1)]
        avg_difference = sum(differences)/len(differences)
        self.distance_file.write(str(avg_difference) + '\n')

    def has(self, entities, name):
        return sum([1 if entity['name'] == name else 0 for entity in entities]) > 0

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
            if not world_state.is_mission_running:
                return False
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


def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        print q_values
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        print epsilon
        return A
    return policy_fn




def GetMissionXML(summary):
    ''' Build an XML mission string.'''
    return '''<?xml version="1.0" encoding="UTF-8" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <About>
            <Summary>''' + summary + '''</Summary>
        </About>

        <ModSettings>
            <MsPerTick>20</MsPerTick>
            <PrioritiseOffscreenRendering>0</PrioritiseOffscreenRendering>
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

                    <!-- <DrawEntity x="20.5" y="207.0" z="20.5" type="Zombie" /> -->
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
                        <InventoryItem slot="8" type="diamond_sword"/>
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
    n,gamma,alpha = 10, .5,.7
    estimator = Estimator()
    # stats = plotting.EpisodeStats(
    # episode_lengths=np.zeros(num_episodes),
    # episode_rewards=np.zeros(num_episodes))


    ai = Protector(alpha, gamma, n, estimator)

    # episodes
    for iRepeat in range(num_reps):
        my_mission = MalmoPython.MissionSpec(GetMissionXML("Defend #" + str(iRepeat)), True)
        my_mission_record = MalmoPython.MissionRecordSpec()  # Records nothing by default
        my_mission.requestVideo(1280, 720)
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

        villager_x = random.randint(4,26)
        villager_z = random.randint(4,26)
        zombie_x = random.randint(4,26)
        zombie_z = random.randint(4,26)

        agent_host.sendCommand("chat /summon Villager "+ str(villager_x) +" 207.0 " +str(villager_z) + " {NoAI:1}")
        agent_host.sendCommand("chat /summon Zombie " + str(zombie_x) + " 207.0 " + str(zombie_z))
        #agent_host.sendCommand("chat /effect @p resistance 99999 255")
        agent_host.sendCommand("chat /effect @p invisibility 99999 255")
        agent_host.sendCommand("hotbar.9 1")  # Press the hotbar key
        agent_host.sendCommand("hotbar.9 0")  # Release hotbar key - agent should now be holding diamond_pickaxe
        time.sleep(1)

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