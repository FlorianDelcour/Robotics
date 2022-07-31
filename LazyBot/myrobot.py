"""
    University of Liege
    INFO0948-2 - Introduction to intelligent robotics
    Authors : 
        BOVEROUX Laurie
        DELCOUR Florian
"""

from manipulation import manipulation
import sim as vrep
import numpy as np
import sys
from cleanup_vrep import cleanup_vrep
from youbot_init import youbot_init
from youbot_hokuyo_init import youbot_hokuyo_init
from navigation import navigation
from manipulation import manipulation
from MapController import Map
from RobotController import Robot

#######################################
#         Project parameters          #    
#######################################
width = 15
height = 15
resolution = 5
numTables = 3
radiusTables = 0.4
navDifficulty = 1 # 1: easy (GPS), 2: medium (beacon)
mapDifficulty = 2 # 1: easy, 2: hard


#######################################
#         Simulator connection        #    
#######################################
timestep = .05

vrep.simxFinish(-1)
clientID = vrep.simxStart('127.0.0.1',  19997, True, True, 2000, 5)
returnCode = vrep.simxSynchronous(clientID, True)

if clientID < 0:
    sys.exit('Failed connecting to remote API server. Exiting.')

print('Connection ' + str(clientID) + ' to remote API server open')
vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)

for i in range(int(1./timestep)):
    vrep.simxSynchronousTrigger(clientID)
    vrep.simxGetPingTime(clientID)

h = youbot_init(vrep, clientID)
h = youbot_hokuyo_init(vrep, h)

for i in range(int(1./timestep)):
    vrep.simxSynchronousTrigger(clientID)
    vrep.simxGetPingTime(clientID)


#######################################
#           Map/Robot init            #    
#######################################
map = Map(height, width, resolution, numTables, radiusTables)
robot = Robot(vrep, clientID, h)

#######################################
#           Milestones                #    
#######################################
# navigation(vrep, clientID, h, robot, map, navDifficulty)
# with open('mapNavBeacon.npy', 'wb') as f:
#     np.save(f, map.map)
manipulation(vrep, clientID, h, robot, map, navDifficulty, mapDifficulty)


#######################################
#           End simulator             #    
#######################################
cleanup_vrep(vrep, clientID)
print('Simulation has stopped')