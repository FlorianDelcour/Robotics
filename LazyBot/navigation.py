"""
    University of Liege
    INFO0948-2 - Introduction to intelligent robotics
    Authors : 
        BOVEROUX Laurie
        DELCOUR Florian
"""

import time
import numpy as np
import sys
import matplotlib.pyplot as plt
from cleanup_vrep import cleanup_vrep
from vrchk import vrchk
from youbot_drive import youbot_drive
from beacon import beacon_init

def navigation(vrep, clientID, h, robot, map, navDifficulty):

    print("-----------------Navigation-----------------")
    if navDifficulty == 3:
        print("Navigation difficulty: GPS")
    else:
        print("Navigation difficulty: Beacon")
    print("---------------------------------------------")

    t_run = []

    #######################################
    #           Initialization            #    
    #######################################
    forwBackVel = 0
    rightVel = 0
    rotateRightVel = 0

    # LazyBot init pos
    res, youbotPos = vrep.simxGetObjectPosition(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
    h = youbot_drive(vrep, h, forwBackVel, rightVel, rotateRightVel)

    # beacons init
    beacons_handle = beacon_init(vrep, clientID, h)
    bPos = np.zeros((len(beacons_handle), 3))
    for i, beacon in enumerate(beacons_handle):   
        res, bPos[i] = vrep.simxGetObjectPosition(clientID, beacon, -1, vrep.simx_opmode_oneshot_wait)
        vrchk(vrep, res, True) 

    #######################################
    #        State machine variables      #    
    #######################################
    pathList = []
    objective = []
    currObjAccomplished = True
    maxExplored = False

    # show the map every 50 iterations
    mapCounter = 0
    map.show()

    # print variable
    movePrint = True
    movePrint2 = True


    #######################################
    #              Start loop             #    
    #######################################
    while True:
        try:

            t_loop = time.perf_counter()
            # Check the connection with the simulator
            if vrep.simxGetConnectionId(clientID) == -1:
                sys.exit('Lost connection to remote API.')


            #############################################
            #  Update robot position and orientation    #    
            #############################################
            # Using GPS
            if navDifficulty == 1:
                robot.updatePosAndOrient()
            # Using beacons
            else:
                robot.updatePosAndOrientBeacon(beacons_handle, bPos)


            #############################################
            #       Update data from hokuyo sensors     #    
            #############################################
            map.updateDataHokuyo(robot)


            #############################################
            #    Check if LazyBot has front obstacle    #    
            #############################################
            if robot.frontObstacle(map):
                print("Obs : front obstacle -> reset path and objective")
                movePrint = True
                robot.stop()
                pathList = []
                objective = []
                currObjAccomplished = True


            ########################################################
            #  Define new path and objective if pathList is empty  #    
            ########################################################
            if currObjAccomplished:
                if not pathList:

                    if robot.getDistFront() >= 3:
                        if movePrint2:
                            movePrint2 = False
                            print('State : forward')
                        objective = []
                        robot.forward(map, 0, nav=True)
                    
                    else:
                        print('State : compute a new objective')
                        movePrint = True
                        movePrint2 = True
                        robot.stop()
                        fromPt = map.hokuyoPt[:,int(map.hokuyoPt.shape[1] / 2)]
                        pathList = map.getNextPath(robot, fromPt)
                        if len(pathList) == 0:
                            print("Error during path computation")  

                        # Map explored
                        elif np.isinf(pathList[0][0]):
                            maxExplored = True

                else:
                    print("State : new target point")
                    movePrint = True
                    objective = pathList[0]
                    del(pathList[0])


            ####################################################
            #    Check if current objective is accomplished    #    
            ####################################################            
            if objective:
                # check if robot has accomplished its current objective
                currObjAccomplished = robot.checkCurrentObj(objective, map)

			    # if robot has not accomplished its objective, go to it
                if not currObjAccomplished:
                    if movePrint:
                        print("State : move")
                        movePrint = False
                    robot.move(objective, map)


            ######################
            #     Display map    #    
            ######################
            if mapCounter == 50:
                pathCopy = pathList.copy()
                pathCopy.insert(0, objective)
                if len(pathCopy[0]) != 0:
                   map.show(pathCopy, robot)
                mapCounter = 0


            ############################
            #    Navigation finished   #
            ############################
            if maxExplored:
                map.show()
                print('State : map is fully explored')
                break

            
            mapCounter += 1
            end_time = time.perf_counter()
            t_run.append((end_time-t_loop)*1000.)  # In ms
            vrep.simxSynchronousTrigger(clientID)
            vrep.simxGetPingTime(clientID)

        except KeyboardInterrupt:
            cleanup_vrep(vrep, clientID)
            sys.exit('Stop simulation')

    map.show()

    
    # Histogram of time loop
    n, x, _ = plt.hist(t_run, bins=100)
    plt.hist(t_run, bins=100)
    plt.vlines(np.min(t_run), 0, np.max(n), linewidth=1.5, colors="r")
    #plt.vlines(np.max(t_run), 0, np.max(n), linewidth=1.5, colors="k")
    plt.xlabel(r"time $t_{\rm{loop}}$ (ms)")
    plt.ylabel("Number of loops (-)")
    plt.savefig("t_loop_nav.png")
    # plt.show()