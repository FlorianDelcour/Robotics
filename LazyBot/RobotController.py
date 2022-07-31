"""
    University of Liege
    INFO0948-2 - Introduction to intelligent robotics
    Authors : 
        BOVEROUX Laurie
        DELCOUR Florian
"""

import numpy as np
from vrchk import vrchk
from youbot_drive import youbot_drive
from utils_sim import angdiff
from beacon import youbot_beacon
from scipy.optimize import least_squares
from youbot_hokuyo import youbot_hokuyo
from youbot_xyz_sensor import youbot_xyz_sensor
import matplotlib.pyplot as plt
import open3d
import scipy.cluster.hierarchy as hcluster
from robopy.base.transforms import transl, trotx, troty, trotz

class Robot:

    def __init__(self, vrep, clientID, h):

        # simulator robot link
        self.vrep = vrep
        self.clientID = clientID
        self.h = h

        # LazyBot position and orientation
        res, tmpYoubotPosInit = vrep.simxGetObjectPosition(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
        vrchk(vrep, res, True)
        self.youbotPosInit = np.array(tmpYoubotPosInit)
        res, tmpYoubotEulerInit = vrep.simxGetObjectOrientation(clientID, h['ref'], -1, vrep.simx_opmode_streaming)
        vrchk(vrep, res, True)
        self.youbotEulerInit = tmpYoubotEulerInit
        self.youbotPos = self.youbotPosInit
        self.youbotEuler = self.youbotEulerInit

        # velocities of robot's wheels
        self.forwBackVel = 0
        self.rightVel = 0
        self.rotateRightVel = 0

        # parameter when the robot is near an obstacle
        self.nearCounter = 0
        self.isNear = False
        self.nearTresh = 1
        self.prevGuess = None
        

    def update_h(self, forwBackVel, rightVel, rotateRightVel):
        """Update robot's wheel velocities"""

        self.h = youbot_drive(self.vrep, self.h, forwBackVel, rightVel, rotateRightVel)


    def updatePosAndOrient(self):
        """Update robot's position and orientation with GPS"""

        res, tmpYoubotPos = self.vrep.simxGetObjectPosition(self.clientID, self.h['ref'], -1, self.vrep.simx_opmode_buffer)
        vrchk(self.vrep, res, True)
        self.youbotPos = np.array(tmpYoubotPos)
        
        res, tmpYoubotEuler = self.vrep.simxGetObjectOrientation(self.clientID, self.h['ref'], -1, self.vrep.simx_opmode_streaming)
        vrchk(self.vrep, res, True)
        self.youbotEuler = np.array(tmpYoubotEuler)


    def updatePosAndOrientBeacon(self, beacons_handle, bPos):
        """Update robot's position and orientation with beacons"""

        beacon_dist = youbot_beacon(self.vrep, self.clientID, beacons_handle, self.h, flag=0)
        # simply perform triangulation
        p1 = [bPos[0][0],bPos[0][1],beacon_dist[0]]
        p2 = [bPos[1][0],bPos[1][1],beacon_dist[1]]
        p3 = [bPos[2][0],bPos[2][1],beacon_dist[2]]
        ans = self.intersectionPoint(p1,p2,p3)
        arrayAns = np.array([ans.x[0], ans.x[1], 0])
        self.prevGuess = (arrayAns[0], arrayAns[1])
        self.youbotPos = arrayAns
        res, tmpYoubotEuler = self.vrep.simxGetObjectOrientation(self.clientID, self.h['ref'], -1, self.vrep.simx_opmode_streaming)
        vrchk(self.vrep, res, True)
        self.youbotEuler = np.array(tmpYoubotEuler)


    def intersectionPoint(self,p1,p2,p3): 
        """
        Get the intersection point of the three points
        ref : https://www.javaer101.com/fr/article/339358.html
        """

        x1, y1, dist_1 = (p1[0], p1[1], p1[2])
        x2, y2, dist_2 = (p2[0], p2[1], p2[2])
        x3, y3, dist_3 = (p3[0], p3[1], p3[2])

        def eq(g):
            x, y = g
            return (
                (x - x1)**2 + (y - y1)**2 - dist_1**2,
                (x - x2)**2 + (y - y2)**2 - dist_2**2,
                (x - x3)**2 + (y - y3)**2 - dist_3**2)

        if self.prevGuess == None:
            guess = (x1, y1 + dist_1)
        else:
            guess = self.prevGuess
        ans = least_squares(eq, guess, ftol=1e-3, xtol=1e-3, bounds = (-7.3, 7.3))
        return ans

    def get_pos_from_beacon(self):
        """
        Returns the youBot position using the 3 beacons.
        """
        z_pos = 0  # We ignore z position.

        x0, y0, z0 = self.beacons_world_pos[0]
        x1, y1, z1 = self.beacons_world_pos[1]
        x2, y2, z2 = self.beacons_world_pos[2]
        z_beacons = [z0, z1, z2]
        our_dist = []
        for i in range(len(self.beacon_dist)):
            our_dist.append(
                np.sqrt(self.beacon_dist[i]**2 - (z_pos - z_beacons[i])**2))
        d0, d1, d2 = our_dist
        d0_2 = d0**2
        d1_2 = d1**2
        d2_2 = d2**2
        x0_2, y0_2 = x0**2, y0**2
        x1_2, y1_2 = x1**2, y1**2
        x2_2, y2_2 = x2**2, y2**2

        youbot_y = (1 / (2 * (-y0 + y1 -
                              (((-y0 + y2) * (-x0 + x1)) / (-x0 + x2))))) * \
                   (d0_2 - d1_2 + y1_2 - y0_2 + x1_2 - x0_2 -
                    ((-x0 + x1) / (-x0 + x2)) *
                    (d0_2 - d2_2 - x0_2 + x2_2 - y0_2 + y2_2))
        youbot_x = (1 / (2 * (-x0 + x2))) * (d0_2 - d2_2 - x0_2 +
                                             x2_2 - 2 * youbot_y * (-y0 + y2)
                                             - y0_2 + y2_2)

        return youbot_x, youbot_y, 


    def getAngle(self, intPt, map):
        """Get the angle between robot and point 'intPt'"""

        intPtx = intPt[0]
        intPty = intPt[1]
        youbotPosMap = (self.youbotPos  - self.youbotPosInit + np.array([map.height, map.width,0]))
        a = np.array([intPtx, intPty])
        b = np.array([youbotPosMap[0], youbotPosMap[1]])
        delta = a - b
        cos_theta = delta[0] / (np.linalg.norm(delta))
        sign = 1 if delta[1] > 0 else -1
        theta = sign * np.arccos(cos_theta) + np.pi/ 2

        return theta


    def move(self, obj, map):
        """
        Set the velocities of the robot according to its position
        and its orientation so that the robot can reach the
        objective 'obj'.
        """

        objective = np.array([obj[0]/map.resolution, obj[1]/map.resolution])
        youbotPosMap = (self.youbotPos  - self.youbotPosInit + np.array([map.height, map.width,0]))[:2]

        # get angle between robot position and objective position
        rotAngl = self.getAngle(objective, map)

        # get distance to objective, for position and rotation
        distObjPos = np.linalg.norm([objective-youbotPosMap])
        distObjRot = abs(angdiff(self.youbotEuler[2], rotAngl))

        # define default values
        self.forwBackVel = 0
        self.rightVel = 0
        self.rotateRightVel = 0

        forward = -6 * distObjPos
        rotation = angdiff(self.youbotEuler[2], rotAngl)

        # 360° = 0°
        if abs(distObjRot - 2*np.pi) < 0.03:
            distObjRot = 0

        # set velocities of the robot according to its objective
        if distObjRot > np.pi / 6 : # 30°
            if rotation >  np.pi:
                self.rotateRightVel = 2.5 *(rotation - 2*np.pi)
            else:
                self.rotateRightVel = 2.5 * rotation
        elif distObjRot > np.pi / 8: # 22.5°
            self.rotateRightVel = rotation
            self.forwBackVel = forward / 2
        elif distObjRot > np.pi / 18: # 10°
            self.rotateRightVel = rotation / 2
            self.forwBackVel = forward
        else:
            self.forwBackVel = forward

        self.drive()


    def checkCurrentObj(self, intPt, map, prec = 0.5):
        """Check if robot has reached its objective : 'intPt'"""

        youbotPosMap = (self.youbotPos  - self.youbotPosInit + np.array([map.height, map.width,0]))[:2]
        objective = np.array([intPt[0]/map.resolution, intPt[1]/map.resolution])

        dist = np.linalg.norm([objective[0]-youbotPosMap[0], objective[1]-youbotPosMap[1]])

        if dist < prec:
            return True
        else:
            return False


    def frontObstacle(self, map):
        """Check if there is an obstacle in front of the robot (dist < nearTresh)"""

        obstacle = False

        # if robot is not moving, don't check
        if self.forwBackVel == 0:
            return

        # counter to let time to the robot to move
        if self.isNear:
            if self.nearCounter == 50:
                self.isNear = False
                self.nearCounter = 0
            
            else:
                self.nearCounter += 1
                return obstacle
        
        # check if robot is too close to something
        if self.checkFront(self.nearTresh, map):
            obstacle = True
            self.isNear = True

        return obstacle


    def checkFront(self, maxDist, map):
        """Check if front laser detects something under 'maxDist'"""

		# By default, no obstacle
        near = False
        sizeCts = int(map.hokuyoPt.shape[1] / 2)
        # Distance to some elements in front of the robot
        nb_rays = 85
        distFront = []
        youbotPosMap = (self.youbotPos  - self.youbotPosInit + np.array([map.height, map.width,0]))[:2]
        for i in range(sizeCts-nb_rays, sizeCts+nb_rays):
            a = [map.hokuyoPt[0,i] - youbotPosMap[0], map.hokuyoPt[1,i] - youbotPosMap[1]]
            distFront.append(np.linalg.norm(a))

        # Check if robot is too close to something
        if all(dist < maxDist for dist in distFront):
            near = True
        return near


    def stop(self):
        """Stop the robot"""

        self.forwBackVel = 0
        self.rightVel = 0
        self.rotateRightVel = 0
        
        while self.h['previousForwBackVel'] != 0 or  self.h['previousLeftRightVel'] != 0 or self.h['previousRotVel'] != 0:
            self.drive()
    

    def drive(self):
        """Drive the robot"""
        self.h = youbot_drive(self.vrep, self.h, self.forwBackVel, self.rightVel, self.rotateRightVel)


    def rotate(self, rotAngl):
        """Rotate the robot by 'rotAngl'"""

        # get distance to objective, for rotation
        distObjRot = abs(angdiff(self.youbotEuler[2], rotAngl))

        # define default values
        self.forwBackVel = 0
        self.rightVel = 0
        self.rotateRightVel = 0

        rotation = angdiff(self.youbotEuler[2], rotAngl)

        # 360° = 0°
        if abs(distObjRot - 2*np.pi) < 0.001:
            distObjRot = 0

        # set velocities of the robot according to its objective
        if distObjRot > np.pi / 6 : # 30°

            if rotation >  np.pi:
                self.rotateRightVel = 2.5 *(rotation - 2*np.pi)
            else:
                self.rotateRightVel = 2.5 * rotation

        elif distObjRot > np.pi / 8: # 22.5°
            self.rotateRightVel = rotation
        elif distObjRot > np.pi / 18: # 10°
            self.rotateRightVel = rotation / 2
        else:
            self.rotateRightVel = rotation / 2

        self.drive()


    def getDistFront(self):
        """Get distance to the nearest obstacle in front of the robot"""

        scanned_points, _ = youbot_hokuyo(self.vrep, self.h, self.vrep.simx_opmode_buffer)
        dist = np.inf
        cst = int(scanned_points.shape[1])
        for i in range(86): #10° = 28 rays ~ 30 rays
            a = abs(scanned_points[1,cst-1-i])
            b = abs(scanned_points[4,i])
            if a < dist:
                dist = a
            if b < dist:
                dist = b

        return dist


    def forward(self, map, objective, nav=False):
        """Move the robot forward until it reaches its objective"""

        if nav:
            distObjPos = 1
        else:
            youbotPosMap = (self.youbotPos  - self.youbotPosInit + np.array([map.height, map.width,0]))[:2]
            # get distance to objective, for position and rotation
            distObjPos = np.linalg.norm([objective-youbotPosMap])

        # define default values
        self.forwBackVel = 0
        self.rightVel = 0
        self.rotateRightVel = 0

        self.forwBackVel = -6 * distObjPos
        self.drive()


    def get_transform(self,handle1, handle2):
        """Return the transform matrix (4x4)."""

        _, pos = self.vrep.simxGetObjectPosition(self.clientID, handle1, handle2, self.vrep.simx_opmode_oneshot_wait)
        _, euler_angles = self.vrep.simxGetObjectOrientation(self.clientID, handle1, handle2, self.vrep.simx_opmode_oneshot_wait)
        T = np.eye(4)
        T[:3, :3] = open3d.geometry.TriangleMesh.create_coordinate_frame().get_rotation_matrix_from_xyz(euler_angles)
        T[:3, 3] = np.array(pos).T
        return T


    def getPointCloud(self, map, mapDifficulty, bigAngle=True):
        """Make 3D PointCloud of the object table"""

        # According to easy or hard table, set different scan angle
        if bigAngle and mapDifficulty == 1:
            scanAngles = np.arange(-np.pi/3, np.pi/3, np.pi/27)
            viewAngle = np.pi/27
            tresh = 0.1
        elif not bigAngle and mapDifficulty == 1:
            scanAngles = np.arange(-np.pi/8, np.pi/8, np.pi/32)
            viewAngle = np.pi/32
            tresh = 0.1
        elif bigAngle and mapDifficulty == 2:
            scanAngles = np.arange(-np.pi/6, np.pi/6, np.pi/32)
            viewAngle = np.pi/32
            tresh = 0.002
        else:
            scanAngles = np.arange(-np.pi/12, np.pi/12, np.pi/64)
            viewAngle = np.pi/64
            tresh = 0.002

        pointCloud  = None

        for i in range(len(scanAngles)):
            # rotate the sensor
            self.vrep.simxSetObjectOrientation(self.clientID, self.h['rgbdCasing'], self.h['ref'], np.array([0, 0, (-np.pi / 2) + scanAngles[i]]), self.vrep.simx_opmode_oneshot_wait)
            self.vrep.simxSetFloatSignal(self.clientID, 'rgbd_sensor_scan_angle', viewAngle, self.vrep.simx_opmode_oneshot_wait)
            self.vrep.simxSetIntegerSignal(self.clientID, 'handle_xyz_sensor', 1, self.vrep.simx_opmode_oneshot_wait)

            self.vrep.simxSynchronousTrigger(self.clientID)
            self.vrep.simxGetPingTime(self.clientID)
            pts = youbot_xyz_sensor(self.vrep, self.h, self.vrep.simx_opmode_oneshot_wait)
     
            self.vrep.simxSynchronousTrigger(self.clientID)
            self.vrep.simxGetPingTime(self.clientID)

            # take only points where d < 1.2 meters
            ind = np.where(pts[:,3] <= 1.2)
            if len(ind[0]) == 0:
                continue
            tmp = pts[ind].copy()

            # take only points where z > -0.04 meters (to avoid point on table)
            ind = np.where(tmp[:,1] >= -0.04)
            if len(ind[0]) == 0:
                continue
            tmp = tmp[ind].copy()

            xyzd = tmp
            xyzd[:, 3] = 1
            trf1 = self.get_transform(self.h['xyzSensor'], self.h['ref'])
            trf2 = transl(self.youbotPos[0],self.youbotPos[1],self.youbotPos[2]) * trotx(self.youbotEuler[0]) * troty(self.youbotEuler[1]) * trotz(self.youbotEuler[2])
            pts = (trf1 @ xyzd.T).T
            pts = (trf2 @ pts.T).T

            if pointCloud is None:
                pointCloud = pts
            else:
                pointCloud = np.concatenate((pointCloud, pts))
        
        if type(pointCloud) == type(None):
            return [np.inf,np.inf,np.inf]

        # clustering with a minimal distance between the clusters of 0.1 meters
        if mapDifficulty == 1:
            clusters = hcluster.fclusterdata(pointCloud, tresh, criterion="distance")
        else:
            clusters = hcluster.fclusterdata(pointCloud, 5, criterion="maxclust")

        # get nearest center
        numClusters = len(np.unique(clusters))
        dist = np.inf
        center = None

        for j in range(numClusters):
            ind = np.where(clusters == j+1)
            centerTmp = [pointCloud[ind,0].mean(), pointCloud[ind,1].mean(), pointCloud[ind,2].mean()]
            distTmp = np.linalg.norm([self.youbotPos[0]-centerTmp[0], self.youbotPos[1]-centerTmp[1]])

            if distTmp <= dist:
                dist = distTmp
                center = centerTmp

        return [(center[0]- self.youbotPosInit[0] + map.height)*map.resolution,
                (center[1]- self.youbotPosInit[1] + map.height)*map.resolution]


    def checkGrasp(self):
        """Check if the gripper is empty or not using mean distance of 3D PointCloud"""

        self.vrep.simxSetObjectOrientation(self.clientID, self.h['rgbdCasing'], self.h['ref'], [0, 0, np.pi/2], self.vrep.simx_opmode_oneshot_wait)
        self.vrep.simxSetFloatSignal(self.clientID, 'rgbd_sensor_scan_angle', np.pi/12, self.vrep.simx_opmode_oneshot_wait)
        self.vrep.simxSynchronousTrigger(self.clientID)
        self.vrep.simxGetPingTime(self.clientID)

        self.vrep.simxSetIntegerSignal(self.clientID, 'handle_xyz_sensor', 1, self.vrep.simx_opmode_oneshot_wait)
        self.vrep.simxSynchronousTrigger(self.clientID)
        self.vrep.simxGetPingTime(self.clientID)

        xyzd = youbot_xyz_sensor(self.vrep, self.h, self.vrep.simx_opmode_oneshot_wait)

        # take only points where d < 0.6 meters
        xyzd = xyzd[xyzd[:,3] < 0.6]
        mean_d = np.mean(xyzd[:,3])
        
        # 0.20 has been determined experimentally
        if mean_d > 0.20:
            return False
        else:
            return True