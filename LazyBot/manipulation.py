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
from utils_sim import angdiff
from beacon import beacon_init
from astar import *

def manipulation(vrep, clientID, h, robot, map, navDifficulty, mapDifficulty):
    
    print("-----------------Manipulation-----------------")
    if navDifficulty == 3:
        print("Navigation difficulty: GPS")
    else:
        print("Navigation difficulty: Beacon")
    if mapDifficulty == 3:
        print("Manipulation difficulty: easy Table") 
    else:
        print("Manipulation difficulty: hard Table")
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
    # init state
    fsm = 'objectTable'
    STATE = 'objectTable'

    # variales for 'goToPoint' state
    pathList = []
    objective = []
    currObjAccomplished = True

    # current object center to grasp
    objectCenter = None

    # variable for 'grasp' state
    hasAngle = False

    # load scene from navigation milestone
    map.map = np.load('mapNavBeacon.npy')

    # Init gridExtend1 = map with extended obstacles (1)
    grid = np.zeros((map.map.shape[0], map.map.shape[1]))      
    ind = np.where(map.map[:,:] > 0)
    grid[ind] = 1
    struct2 = sp.generate_binary_structure(2, 2)
    grid = sp.binary_dilation(grid, structure=struct2, iterations = 1).astype(grid.dtype)
    map.gridExtend1 = grid.copy()

    # Init gridExtend2 = map with extended obstacles (2)
    grid = np.zeros((map.map.shape[0], map.map.shape[1]))      
    grid[ind] = 1
    grid = sp.binary_dilation(grid, structure=struct2, iterations = 2).astype(grid.dtype)
    map.gridExtend2 = grid.copy()

    # find center of each table
    print("Analyze map to find target table") 
    map.tableCenterFinding(robot)
    print('Target center found:', map.tableCenter['target'])
    if mapDifficulty == 1:
        tableObject = 'easy'
    else: 
        tableObject = 'hard'

    # print variable
    prevPrint = None


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

            # Robot pos in map
            youbotPosMap = (np.round((robot.youbotPos  - robot.youbotPosInit + np.array([map.height, map.width,0])) * map.resolution)).astype(int)


            #############################################
            #            Finite State Machine           #    
            #############################################
            """
            LazyBot will transiate in some states. Some general variables : 
            'ptToGo' : point to reach 
            'ptToAlign' : point to align with
            'fsm' : current state machine
            'STATE' buffer state
            """
            
            if fsm == 'objectTable':
                """Go to nearest point around table containing objects"""

                if prevPrint != fsm:
                    print('State objectTable : go to table with objects')
                    prevPrint = fsm

                ct = (np.array(map.tableCenterScene[tableObject]) - robot.youbotPosInit[:2] + np.array([map.height, map.width]))*map.resolution
                ptToGo = map.getPtAroundTableCenter(map.tableCenter[tableObject], youbotPosMap)
                ptToAlign = ct

                fsm = 'goToPoint'
                STATE = 'objectTable'
                pathList = []
                objective = []
                currObjAccomplished = True
                pathDone = False


            elif fsm == 'goToPoint':
                """Go to point 'ptToGo'"""

                if prevPrint != fsm:
                    print('State goToPoint : go to point \'ptToGo\'')
                    prevPrint = fsm

                if currObjAccomplished:
                    if not pathList:
                        if pathDone:
                            print("Point reached")
                            fsm = 'align'

                        else:

                            # Larger security of path if robot goes to another table 
                            if STATE == 'objectTable' or STATE == 'targetTable':

                                grid = np.zeros((map.matHeight, map.matWidth))      
                                ind = np.where(map.map[:,:]>0) #where there are obstacles
                                grid[ind] = 1
                                # do the inflation using SciPy
                                struct2 = sp.generate_binary_structure(2, 2)
                                grid = sp.binary_dilation(grid, structure=struct2, iterations=2).astype(grid.dtype)
                                cop = map.map.copy()
                                ind = np.where(grid[:,:]==1)
                                cop[ind] = 10
                                for x in range(-5,10):
                                    for y in range(-5,5):
                                        cop[ptToGo[0]+x, ptToGo[1]+y] = -1
                                
                                for a in range(-2,2):
                                    for b in range(-2,2):
                                        cop[youbotPosMap[0]+a, youbotPosMap[1]+b] = -1

                                # import matplotlib.pyplot as plt
                                # mapCopie = np.copy(cop)
                                # indNeg = np.where(mapCopie[:,:] < 0)
                                # mapCopie[indNeg] = -1
                                # indPos = np.where(mapCopie[:,:] > 0)
                                # mapCopie[indPos] = 1
                                # from matplotlib.colors import ListedColormap
                                # vec_col = ListedColormap(["darkslateblue", "lightblue", "yellow"])
                                # plt.clf()
                                # plt.pcolormesh(mapCopie[:,:],vmax = 1,vmin = -1,cmap=vec_col)
                                # plt.ylim(max(plt.ylim()), min(plt.ylim()))
                                # plt.show(block=True)
                                # plt.pause(0.05)
                                
                                pathList= astar(youbotPosMap[0], youbotPosMap[1], ptToGo[0], ptToGo[1], cop, extend=0)
                                if not pathList:
                                    robot.forwBackVel = -0.1
                                    robot.drive()
                                    pathDone = False
                                else:
                                    robot.stop()
                                    pathDone = True

                            if STATE == 'object':
                                robot.stop()
                                pathList= astar(youbotPosMap[0], youbotPosMap[1], ptToGo[0], ptToGo[1], map.map, extend=1)
                                pathDone = True

                        # optimize path
                        if len(pathList) >=3:
                            pathList = map.optimizePath(pathList)

                    else:
                        # Get intermediate objective
                        objective = pathList[0]
                        del(pathList[0])

                if objective:
                    # check if LazyBot has accomplished its current objective
                    if STATE == 'object':
                        prec = 0.2
                    else:
                        prec = 0.5
                    currObjAccomplished = robot.checkCurrentObj(objective, map, prec)

                    # If LazyBot has not accomplished its objective, go to it
                    if not currObjAccomplished:
                        robot.move(objective, map)

            elif fsm == 'align':
                """Align with point 'ptToAlign'"""

                if prevPrint != fsm:
                    print('State align : align with point \'ptToAlign\'')
                    prevPrint = fsm

                # angle between point to align and robot
                anglePt = np.array([ptToAlign[0]/map.resolution, ptToAlign[1]/map.resolution])
                rotAngl = robot.getAngle(anglePt, map)
                distRot = abs(angdiff(robot.youbotEuler[2], rotAngl))

                # precision changes depending on buffer state
                if STATE == 'objectTable':
                    prec = 0.03
                else: 
                    prec = 0.001

                if distRot < prec or abs(distRot - 2*np.pi) < prec:
                    robot.stop()

                    if STATE == 'objectTable':
                        fsm = 'getCloser'
                    elif STATE == 'targetTable':
                        fsm = 'dropForward'
                    elif STATE == 'object':
                        fsm = 'graspForward'
                    elif STATE == 'graspPointCloud':
                        fsm = 'graspAdjust'
                    else:
                        print('Unknown state')

                else:
                    robot.rotate(rotAngl)


            elif fsm == 'getCloser':
                """Get closer to object table"""

                if prevPrint != fsm:
                    print('State getCloser : align with point \'ptToAlign\'')
                    prevPrint = fsm

                distPt = np.array([ptToGo[0]/map.resolution,ptToGo[1]/map.resolution])
                dist = robot.getDistFront()

                # 0.6 meters from table
                if dist <= 0.6:
                    robot.stop()
                    fsm = 'object'
                else :
                    robot.forward(map, distPt)
                    

            elif fsm == 'object':
                """Make 3D pointcloud of object table and go to nearest object center""" 

                if prevPrint != fsm:
                    print('State object : analyze table and get the nearest object center')
                    prevPrint = fsm

                objectCenter = robot.getPointCloud(map, mapDifficulty)

                # if no object center found, table is empty and manipulation is finished
                if objectCenter[0] == np.inf:
                    print("No more objects - > Finish")
                    break

                ptToGo = map.getPointNearObject(objectCenter, map.tableCenter[tableObject])
                ptToAlign = objectCenter

                STATE = 'object'
                fsm = 'goToPoint'
                pathList = []
                objective = []
                currObjAccomplished = True
                pathDone = False
                
                
            elif fsm == 'graspForward':
                """Get closer to table to prepare grasp""" 

                if prevPrint != fsm:
                    print('State graspForward : Get closer to table to prepare grasp')
                    prevPrint = fsm

                if robot.getDistFront() <= 0.5:
                    robot.stop()
                    fsm = 'graspPointCloud'

                else:
                    objectCent = np.array(objectCenter)/map.resolution
                    robot.forward(map, objectCent)
            
            
            elif fsm == 'graspPointCloud':
                """Get a more precise pointCloud of the front object""" 

                if prevPrint != fsm:
                    print('State graspPointCloud : Precise PointCloud of front object')
                    prevPrint = fsm

                objectCenter = robot.getPointCloud(map, mapDifficulty, bigAngle=False)
                STATE = 'graspPointCloud'
                fsm = 'align'
                ptToAlign = objectCenter


            elif fsm == 'graspAdjust':
                """Adjust the position of the robot to grab the object"""

                if prevPrint != fsm:
                    print('State graspAdjust : Adjust distance between LazyBot and object')
                    prevPrint = fsm

                STATE = 'object'
                youbotPos = (robot.youbotPos  - robot.youbotPosInit + np.array([map.height, map.width,0]))[:2]
                objectCent = np.array(objectCenter)/map.resolution
                distRobObj = np.linalg.norm(youbotPos-objectCent)

                # Check if object is too far from robot due to error
                if distRobObj > 1:
                    print('ERROR : Too far from object, restart at object')
                    fsm = 'object'

                # Robot must be at exactly 0.59 meters from object
                # -> determined experimentally
                displacement = distRobObj - (0.585)

                if abs(displacement) < 0.01:
                    robot.stop()
                    fsm = 'graspHalf'
                    hasAngle = False
                else: 
                    robot.forwBackVel = -np.sign(displacement)/20
                    robot.drive()
            
            
            elif fsm == 'graspHalf':
                """Half return to grasp the object"""

                if prevPrint != fsm:
                    print('State graspHalf : Half return')
                    prevPrint = fsm

                if not hasAngle:
                    rotAngl = robot.youbotEuler[2] - np.pi
                    hasAngle = True

                if abs(angdiff(robot.youbotEuler[2], rotAngl)) < 0.01:
                    robot.stop()
                    hasAngle = False
                    fsm = 'graspMoveArm1'
                else:
                    prev_fsm = fsm
                    fsm = 'rotate'
                    rotateObj = rotAngl


            elif fsm == 'graspMoveArm1':
                """First move of the arm"""

                if prevPrint != fsm:
                    print('State graspMoveArm1 : First move of the arm')
                    prevPrint = fsm

                # Open the gripper
                res = vrep.simxSetIntegerSignal(clientID, 'gripper_open', 1, vrep.simx_opmode_oneshot_wait)
                vrep.simxSetIntegerSignal(clientID, 'km_mode', 0, vrep.simx_opmode_oneshot_wait)

                # Determined experimentally
                target_1 = [0, -np.pi/4, -2.9*np.pi/4, np.pi/2]
                cond = True

                for i in range(len(target_1)):
                    vrep.simxSetJointTargetPosition(clientID, h["armJoints"][i], target_1[i], vrep.simx_opmode_oneshot)            
                    _, curr = vrep.simxGetJointPosition(clientID, h["armJoints"][i], vrep.simx_opmode_buffer)
                    if abs(angdiff(curr, target_1[i])) > 0.001:
                        cond = False

                if cond:
                    fsm = 'graspMoveArm2'
            

            elif fsm == 'graspMoveArm2':
                """Second move of the arm"""

                if prevPrint != fsm:
                    print('State graspMoveArm2 : Second move of the arm')
                    prevPrint = fsm

                # Second move of the arm
                target_2 = [0, -np.pi/4, -np.pi/2, np.pi/4]
                cond = True

                for i in range(len(target_2)):
                    vrep.simxSetJointTargetPosition(clientID, h["armJoints"][i], target_2[i], vrep.simx_opmode_oneshot)            
                    _, curr = vrep.simxGetJointPosition(clientID, h["armJoints"][i], vrep.simx_opmode_buffer)
                    if abs(angdiff(curr, target_2[i])) > 0.001:
                        cond = False

                if cond:
                    fsm = 'graspMoveArm3'


            elif fsm == 'graspMoveArm3':
                """Third move of the arm"""

                if prevPrint != fsm:
                    print('State graspMoveArm3 : Third move of the arm')
                    prevPrint = fsm
                
                # Third move of the arm
                target_3 = [0, -13*np.pi/40, -np.pi/2, 13*np.pi/40]
                cond = True

                for i in range(len(target_3)):
                    vrep.simxSetJointTargetPosition(clientID, h["armJoints"][i], target_3[i], vrep.simx_opmode_oneshot)            
                    _, curr = vrep.simxGetJointPosition(clientID, h["armJoints"][i], vrep.simx_opmode_buffer)
                    if abs(angdiff(curr, target_3[i])) > 0.001:
                        cond = False

                if cond:
                    fsm = 'closeGripper'
                    time_to_close = time.time()


            elif fsm == 'closeGripper':
                """Close gripper to grasp the object"""

                if prevPrint != fsm:
                    print('State closeGripper : Closing gripper')
                    prevPrint = fsm

                vrep.simxSetIntegerSignal(clientID, 'gripper_open', 0, vrep.simx_opmode_oneshot_wait)

                if time.time()-time_to_close > 3.:
                    fsm = 'liftUpReset'
                    next_fsm = 'graspCheck'


            elif fsm == 'liftUpReset':
                """Reset arm position after grasp or drop"""

                if prevPrint != fsm:
                    print('State liftUpReset : Reset arm position')
                    prevPrint = fsm
                    
                # Initial angles of the arm
                reset = [0, 0.54, 0.915, 1.269]
                cond = True

                for i in range(len(reset)):
                    vrep.simxSetJointTargetPosition(clientID, h["armJoints"][i], reset[i], vrep.simx_opmode_oneshot)            
                    _, curr = vrep.simxGetJointPosition(clientID, h["armJoints"][i], vrep.simx_opmode_buffer)
                    if abs(angdiff(curr, reset[i])) > 0.001:
                        cond = False

                if cond:
                    if next_fsm == 'graspCheck':
                        time.sleep(2)
                        fsm = next_fsm
                    else:
                        fsm = next_fsm


            elif fsm == 'graspCheck':
                """Check if the object has been grasped"""

                if prevPrint != fsm:
                    print('State graspCheck : check if object is grasped')
                    prevPrint = fsm

                if robot.checkGrasp():
                    fsm = 'targetTable'
                    print('Object is grasped !')
                else:
                    fsm = 'objectTable'
                    print('Object is not grasped, LazyBot will try again !')


            elif fsm == 'targetTable':
                """When the object is grasped, go to the target table"""

                if prevPrint != fsm:
                    print('State targetTable : Go to target table')
                    prevPrint = fsm

                ct = map.tableCenter['target']
                ptInd = np.argmin([np.linalg.norm([youbotPosMap[0]-map.posAroundTarget[i][0], youbotPosMap[1]-map.posAroundTarget[i][1]]) for i in range(len(map.posAroundTarget))])
                ptToGo = map.posAroundTarget[ptInd]
                del(map.posAroundTarget[ptInd])
                ptToAlign = ct

                STATE = 'targetTable'
                fsm = 'goToPoint'
                pathList = []
                objective = []
                currObjAccomplished = True
                pathDone = False


            elif fsm == 'dropForward':
                """Get closer to table to drop the object"""

                if prevPrint != fsm:
                    print('State dropForward : get closer to target table')
                    prevPrint = fsm

                # Stop when target table is at 0.5 meters from LazyBot
                if robot.getDistFront() <= 0.50:
                    robot.stop()
                    fsm = 'dropHalf'
                    hasAngle = False
                else:
                    targetCenter = np.array(map.tableCenter['target'])/map.resolution
                    robot.forward(map, targetCenter)


            elif fsm == 'dropHalf':
                """Half return to drop the object"""

                if prevPrint != fsm:
                    print('State dropHalf : half return')
                    prevPrint = fsm

                if not hasAngle:
                    rotAngl = robot.youbotEuler[2] - np.pi
                    hasAngle = True

                if abs(angdiff(robot.youbotEuler[2], rotAngl)) < 0.001:
                    robot.stop()
                    hasAngle = False
                    fsm = 'dropMoveArm'
                else:
                    prev_fsm = fsm
                    fsm = 'rotate'
                    rotateObj = rotAngl


            elif fsm == 'dropMoveArm':
                """Move the arm to drop the object"""

                if prevPrint != fsm:
                    print('State dropMoveArm : Move arm')
                    prevPrint = fsm

                # This time, we can go in 1 step
                drop_angles = [0, -13*np.pi/40, -np.pi/2, 13*np.pi/40]
                cond = True

                for i in range(len(drop_angles)):
                    vrep.simxSetJointTargetPosition(clientID, h["armJoints"][i], drop_angles[i], vrep.simx_opmode_oneshot)            
                    _, curr = vrep.simxGetJointPosition(clientID, h["armJoints"][i], vrep.simx_opmode_buffer)
                    if abs(angdiff(curr, drop_angles[i])) > 0.001:
                        cond = False

                if cond:
                    fsm = 'openGripper'
                    time_to_close = time.time()


            elif fsm == 'openGripper':
                """Open gripper to drop the object"""

                if prevPrint != fsm:
                    print('State openGripper : Open gripper')
                    prevPrint = fsm

                vrep.simxSetIntegerSignal(clientID, 'gripper_open', 1, vrep.simx_opmode_oneshot_wait)

                if time.time()-time_to_close > 3.:
                    fsm = 'liftUpReset'
                    next_fsm = 'objectTable'


            elif fsm == 'rotate':
                """Rotate to reach 'rotateObj' angle"""

                if prevPrint != fsm:
                    print('State rotate')
                    prevPrint = fsm

                distRot = abs(angdiff(robot.youbotEuler[2], rotateObj))
                if distRot < 0.001 or abs(distRot - 2*np.pi) < 0.001:
                    fsm = prev_fsm
                else:
                    robot.rotate(rotateObj)


            else:
                sys.exit('Unknown state ' + fsm)


            end_time = time.perf_counter()
            t_run.append((end_time-t_loop)*1000.)  # In ms
                
            vrep.simxSynchronousTrigger(clientID)
            vrep.simxGetPingTime(clientID)

        except KeyboardInterrupt:
            cleanup_vrep(vrep, clientID)
            sys.exit('Stop simulation')


    # Histogram of time loop
    plt.figure()
    n, x, _ = plt.hist(t_run, bins=100)
    plt.vlines(np.min(t_run), 0, np.max(n), linewidth=1.5, colors="r")
    # plt.vlines(np.max(t_run), 0, np.max(n), linewidth=1.5, colors="k")
    plt.xlabel(r"time $t_{\rm{loop}}$ (ms)")
    plt.ylabel("Number of loops (-)")
    plt.savefig("t_loop_manip.png")

    #plt.show()