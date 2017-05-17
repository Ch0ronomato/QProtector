import MalmoPython
import os
import random
import sys
import time
import json
import random
import errno
import math
import Tkinter as tk
from collections import namedtuple
from collections import defaultdict, deque
from timeit import default_timer as timer

MOB_TYPE = "Endermite"  # Change for fun, but note that spawning conditions have to be correct - eg spiders will require darker conditions.
ARENA_WIDTH = 60
ARENA_BREADTH = 60

def getCorner(index,top,left,expand=0,y=206):
    ''' Return part of the XML string that defines the requested corner'''
    x = str(-(expand+ARENA_WIDTH/2)) if left else str(expand+ARENA_WIDTH/2)
    z = str(-(expand+ARENA_BREADTH/2)) if top else str(expand+ARENA_BREADTH/2)
    return 'x'+index+'="'+x+'" y'+index+'="' +str(y)+'" z'+index+'="'+z+'"'

def GetMissionXML(summary):
    ''' Build an XML mission string.'''
    spawn_end_tag = 'type="Pig"/>'
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
                    <StartTime>6000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
            </ServerInitialConditions>
            <ServerHandlers>
                <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1" />
                <DrawingDecorator>
                    <DrawCuboid ''' + getCorner("1",True,True,expand=1) + " " + getCorner("2",False,False,y=226,expand=1) + ''' type="stone"/>
                    <DrawCuboid ''' + getCorner("1",True,True,y=207) + " " + getCorner("2",False,False,y=226) + ''' type="air"/>

                    <DrawEntity x="2.5" y="207.0" z="0.5" type="Pig" />
                </DrawingDecorator>
                <ServerQuitWhenAnyAgentFinishes />
            </ServerHandlers>
        </ServerSection>

        <AgentSection mode="Survival">
            <Name>The Hunted</Name>
            <AgentStart>
                <Placement x="0.5" y="207.0" z="0.5"/>
                <Inventory>
                </Inventory>
            </AgentStart>
            <AgentHandlers>
                <ChatCommands/>
                <ContinuousMovementCommands turnSpeedDegs="360"/>
                <AbsoluteMovementCommands/>
                <ObservationFromNearbyEntities>
                    <Range name="entities" xrange="'''+str(ARENA_WIDTH)+'''" yrange="2" zrange="'''+str(ARENA_BREADTH)+'''" />
                </ObservationFromNearbyEntities>
                <ObservationFromFullStats/>
            </AgentHandlers>
        </AgentSection>

    </Mission>'''


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
            print world_state

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