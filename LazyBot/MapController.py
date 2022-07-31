"""
    University of Liege
    INFO0948-2 - Introduction to intelligent robotics
    Authors : 
        BOVEROUX Laurie
        DELCOUR Florian
"""

import numpy as np
from robopy.base.transforms import transl, trotx, troty, trotz
from youbot_hokuyo import youbot_hokuyo
from matplotlib import path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from astar import *
from skimage import color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
import warnings
warnings.filterwarnings("ignore")

class Map:

    def __init__(self, height, width, resolution,numTables,radiusTables):
        
        ####################################
        #     variables for navigation     #
        ####################################

        # map parameters
        self.height = height
        self.width = width
        self.resolution = resolution
        self.matHeight = 2*self.height*self.resolution
        self.matWidth = 2*self.width*self.resolution
        self.map = np.zeros((self.matHeight, self.matWidth))
        
        # scanned points from hokuyo sensors
        self.hokuyoPt = 0

        # contact points from hokuyo sensors
        self.hokuyoContacts = 0

        # previous goal point
        self.prevGoal = []


        ####################################
        #    variables for manipulation    #
        ####################################
        
        self.numTables = numTables
        self.radiusTables = radiusTables
        self.tableCenter = {'easy': [None, None], 'hard': [None, None], 'target' : [None, None]}
        self.tableCenterScene = {'easy' : [-5.575,2.925], 'hard' : [-5.55,5.2], 'target' : [None,None]}

        # x and y coordinates of the tables perimeter
        self.perimTabx = []
        self.perimTaby = []

        # copy of the map without tables as obstacles
        self.mapWithoutTables = np.zeros((self.matHeight, self.matWidth))

        # extended grids obstacles
        self.gridExtend1 = None
        self.gridExtend2 = None

        # coordinates around target table
        self.posAroundTarget = []


    def updateDataHokuyo(self, robot):
        """Update the map with the data from the hokuyo sensors"""

        # transformation to robot position in the meshgrid
        youbotPosMap = robot.youbotPos  - robot.youbotPosInit + np.array([self.height, self.width,0])
        trf = transl(youbotPosMap[0],youbotPosMap[1],youbotPosMap[2]) * trotx(robot.youbotEuler[0]) * troty(robot.youbotEuler[1]) * trotz(robot.youbotEuler[2])

        # get data from the hokuyo - return empty if data is not captured
        scanned_points, contacts = youbot_hokuyo(robot.vrep, robot.h, robot.vrep.simx_opmode_buffer,trans = trf)

        # coordinates of the sensors
        tmph1 = trf * np.array([[robot.h['hokuyo1Pos'][0],robot.h['hokuyo1Pos'][1],robot.h['hokuyo1Pos'][2],1]]).T
        coordH1 = np.array([tmph1[0][0],tmph1[1][0]],dtype=object)
        
        tmph2 = trf * np.array([[robot.h['hokuyo2Pos'][0],robot.h['hokuyo2Pos'][1],robot.h['hokuyo2Pos'][2],1]]).T
        coordH2 = np.array([tmph2[0][0],tmph2[1][0]],dtype=object)

        all_X_scanned_points = np.concatenate((scanned_points[0,:],scanned_points[3,:]), axis=1)
        all_X = np.insert(all_X_scanned_points,0,coordH1[0],axis=1)
        all_X = np.insert(all_X,len(all_X),coordH2[0],axis=1)

        all_Y_scanned_points = np.concatenate((scanned_points[1,:],scanned_points[4,:]), axis=1)
        all_Y = np.insert(all_Y_scanned_points,0,coordH1[1],axis=1)
        all_Y = np.insert(all_Y,len(all_Y),coordH2[1],axis=1)

        self.hokuyoPt = np.stack((all_X_scanned_points, all_Y_scanned_points), axis=0)

        # create the test meshgrid
        xMax = np.max(all_X)
        xMin = np.min(all_X)
        yMax = np.max(all_Y)
        yMin = np.min(all_Y)
        a = np.arange(xMin,xMax,1/self.resolution)
        b = np.arange(yMin,yMax,1/self.resolution)
        mx, my = np.meshgrid(a,b)
        catalog_test_points = np.stack((mx.flatten(),my.flatten()),axis=1)  

        # create the polygon
        points  = np.transpose(np.stack((all_X, all_Y),axis=0))
        poly = path.Path(points)

        pointsRes = poly.contains_points(catalog_test_points)
        
        # Update map with point in polygon
        # point = free -> -1
        # point = obstacle -> +10
        # point = unobserved -> 0

        # If points in the polygon, there in no obstacle 
        indMinus = np.where(pointsRes)
        self.map[(np.round((catalog_test_points[indMinus,0])*self.resolution)).astype(int), (np.round((catalog_test_points[indMinus,1])*self.resolution)).astype(int)] -=1

        # Update the map in function of the points from the hokuyo sensors
        all_contacts = np.concatenate((contacts[0,:],contacts[1,:]), axis=0)
        self.hokuyoContacts = all_contacts

        indPlus = np.where(all_contacts) #obstacle
        indMinus = np.where(all_contacts == False) #free
        self.map[(np.round((all_X_scanned_points[0,indPlus])*self.resolution)).astype(int), (np.round((all_Y_scanned_points[0,indPlus])*self.resolution)).astype(int)] += 10
        self.map[(np.round((all_X_scanned_points[0,indMinus])*self.resolution)).astype(int), (np.round((all_Y_scanned_points[0,indMinus])*self.resolution)).astype(int)] -= 1


    def show(self, *args):
        """Display the map along with robot pos and trajectory"""
        
        mapCopie = np.copy(self.map)
        indNeg = np.where(mapCopie[:,:] < 0)
        mapCopie[indNeg] = -1
        indPos = np.where(mapCopie[:,:] > 0)
        mapCopie[indPos] = 1
        vec_col = ListedColormap(["darkslateblue", "lightblue", "yellow"])

        if len(args) == 0:
            plt.pcolormesh(mapCopie[:,:],vmax = 1,vmin = -1,cmap=vec_col)
            plt.ylim(max(plt.ylim()), min(plt.ylim()))
            plt.show(block=False)
            plt.pause(0.05)
            plt.savefig("mapExplored.pdf")

        if len(args) == 2:
            if(not len(args[0])):
                return
            plt.clf()
            plt.pcolormesh(mapCopie[:,:],vmax = 1,vmin = -1,cmap=vec_col)
            for point in args[0]:
                plt.scatter(point[1], point[0], c='red')
            robot = args[1]
            youbotPosMap = (robot.youbotPos  - robot.youbotPosInit + np.array([self.height, self.width,0]))[:2] * self.resolution
            plt.scatter(round(youbotPosMap[1]), round(youbotPosMap[0]), c='black')
            plt.ylim(max(plt.ylim()), min(plt.ylim()))
            plt.show(block=False)
            plt.pause(0.05)

        if len(args) == 1:
            plt.clf()
            plt.pcolormesh(mapCopie[:,:],vmax = 1,vmin = -1,cmap=vec_col)
            for i in range(self.numTables):
                for point in zip(self.perimTabx[i], self.perimTaby[i]):
                    plt.scatter(point[1], point[0], 5, c='red')
            plt.ylim(max(plt.ylim()), min(plt.ylim()))
            plt.show(block=True)
            plt.pause(0.05)


    def getNextPath(self, robot, fromPt):
        """Return the next path to follow using astar algorithm"""

        while True:

            youbotPosMap = (np.round((robot.youbotPos  - robot.youbotPosInit + np.array([self.height, self.width,0])) * self.resolution)).astype(int)
            fromPt = fromPt * self.resolution
            gp = self.getNextPoint(youbotPosMap)

            if gp[0] == np.inf:
                print('ERROR : no goal point')
                return [[np.inf,np.inf]]
            
            sp = youbotPosMap[:2]
            path = astar(sp[0], sp[1], gp[0], gp[1], self.map)

            if len(path) == 0:
                self.prevGoal.append((gp[0], gp[1]))
                continue
            
            if len(path) >=3:
                path = self.optimizePath(path)
            
            self.prevGoal = []
            return path


    def optimizePath(self, path):
        """Remove colinear points to optimize path"""

        newPath = [path[0]]
        
        p_1 = path[0]
        p_2 = path[1]
        p_3 = path[2]

        for i in range(2, len(path)):
            x1, y1 = p_1[0], p_1[1]
            x2, y2 = p_2[0], p_2[1]
            x3, y3 = p_3[0], p_3[1] 

            if ((y3 - y2) * (x2 - x1) != (y2 - y1) * (x3 - x2)):
                newPath.append(p_2)

            p_1 = p_2
            p_2 = p_3
            p_3 = path[i]
        
        newPath.append(path[-1])
        return newPath
   

    def getNextPoint(self,youbotPosMap):
        """
        Return the next point to go. It is the closest point greater than 0.5 meters to the robot.
        This point is a unobserved point next to one free point.
        """

        maxDist = np.inf
        nextPt = np.array([np.inf,np.inf])
        grid = np.zeros((self.matHeight, self.matWidth))      
        ind = np.where(self.map[:,:] > 0)
        grid[ind] = 1

        # inflate obstacles of the grid
        struct2 = sp.generate_binary_structure(2, 2)
        grid = sp.binary_dilation(grid, structure=struct2, iterations = 2).astype(grid.dtype)
        ind = np.where((grid==0) & (self.map[:,:] < 0))
    
        for k in range(len(ind[0])):
            i = ind[0][k]
            j = ind[1][k]

            if (i,j) in self.prevGoal:
                continue

            # At least 1 free point around the next target point
            count = 0
            for x in [i-1, i, i+1]:
                for y in [j-1, j, j+1]:
                    if x >= 0 and y >= 0 and x < self.matHeight  and y < self.matWidth and not(x==i and y ==j):
                        if self.map[x,y] == 0 : #There is a unexplored neighboor
                            count += 1
                            break
            
            if count >= 1:
                coordNextPt = np.array([i,j])
                dist = np.linalg.norm(youbotPosMap[:2] - coordNextPt)
                if dist < maxDist and dist > 5*self.resolution:
                    maxDist = dist
                    nextPt = coordNextPt

        return nextPt


    def tableCenterFinding(self,robot):
        """
        Find the center of the target table. 
        Generate some points around object table and target table
        """
        
        # Coordinates of easy and hard table are known and fixed.
        self.tableCenter['easy'] = [round((-5.575 - robot.youbotPosInit[0] + 15)*self.resolution), round((2.925 - robot.youbotPosInit[1] + 15)*self.resolution)]
        self.tableCenter['hard'] = [round((-5.55 - robot.youbotPosInit[0] + 15)*self.resolution), round((5.2 - robot.youbotPosInit[1] + 15)*self.resolution)]

        mapCopy = self.map.copy()
        indOne = np.where(mapCopy > 0)
        indZero = np.where(mapCopy < 0)
        mapCopy[indOne] = 1
        mapCopy[indZero] = 0

        edges = canny(mapCopy)
        # detect two radii
        #hough_radii = np.array([round(self.radiusTables * self.resolution),round((self.radiusTables * self.resolution)+1)])
        hough_radii = np.array([round((self.radiusTables * self.resolution)+1)])
        hough_res = hough_circle(edges, hough_radii)
        
        # select the most prominent 3 circles
        _, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=self.numTables)
        
        # draw them
        image = color.gray2rgb(mapCopy)
        self.mapWithoutTables = self.map.copy()

        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = circle_perimeter(center_y, center_x, radius,shape=image.shape)
            self.perimTabx.append(circy)
            self.perimTaby.append(circx)

            xMax = np.max(circy)
            xMin = np.min(circy)
            yMax = np.max(circx)
            yMin = np.min(circx)
            self.mapWithoutTables[xMin:xMax+1, yMin:yMax+1] = -1

        self.show(True)

        tabInd = []
        for tab in ['easy', 'hard']:
            a = np.linalg.norm([self.tableCenter[tab][0]-cy[0], self.tableCenter[tab][1]-cx[0]])
            b = np.linalg.norm([self.tableCenter[tab][0]-cy[1], self.tableCenter[tab][1]-cx[1]])
            c = np.linalg.norm([self.tableCenter[tab][0]-cy[2], self.tableCenter[tab][1]-cx[2]])
            tabInd.append(np.argmin((a,b,c)))

        targetInd = (set(range(self.numTables))- set(tabInd)).pop()
        self.tableCenter['target'] = [cy[targetInd], cx[targetInd]]

        for j in range(self.perimTabx[targetInd].shape[0]):
            x =  self.perimTabx[targetInd][j]
            y = self.perimTaby[targetInd][j]

            while self.gridExtend1[x,y] != 0:
                x, y = self.extendPointOutTable(cy[targetInd], cx[targetInd], x, y)
            x, y = self.extendPointOutTable(cy[targetInd], cx[targetInd], x, y)

            if [x, y] not in self.posAroundTarget:
                self.posAroundTarget.append([x,y])

        a = len(self.posAroundTarget)
        self.posAroundTarget = self.posAroundTarget[:int(a/3)]

        return
    

    def extendPointOutTable(self,cx,cy,x,y):
        """Get next point out of the table"""
        if x > cx:
            if y > cy:
                return x+1, y+1
            elif y < cy:
                return x+1, y-1
            else:
                return x+1, y
        elif x < cx:
            if y > cy:
                return x-1, y+1
            elif y < cy:
                return x-1, y-1
            else:
                return x-1, y
        else:
            if y > cy:
                return x, y+1
            elif y < cy:
                return x, y-1
            else:
                return x, y


    def getPointNearObject(self,objTar,centerTab):
        """Get nearest free point around the object"""

        # Extend the point out of the table
        cx = centerTab[0]
        cy = centerTab[1]      
        ox = round(objTar[0])
        oy = round(objTar[1])

        align = np.inf

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                newI = ox+i
                newJ = oy+j
                
                if np.linalg.norm([cx-newI, cy-newJ]) < self.radiusTables*self.resolution:
                    continue

                a =  ((cy - oy) * (ox-newI) - (oy-newJ) * (cx - ox))

                if abs(a) < align:
                    align = abs(a)

                    while self.gridExtend1[newI,newJ] != 0:
                        newI += i
                        newJ += j

                    xObject = newI + i
                    yObject = newJ + j
        
        mapCopie = np.copy(self.map)
        indNeg = np.where(mapCopie[:,:] < 0)
        mapCopie[indNeg] = -1
        indPos = np.where(mapCopie[:,:] > 0)
        mapCopie[indPos] = 1
        vec_col = ListedColormap(["darkslateblue", "lightblue", "yellow"])

        return [round(xObject), round(yObject)]


    def getPtAroundTableCenter(self, target, source):
        """
        Get nearest point from robot around the table center
            'target' : point to extend where we want to go (ex: center)
            'source' : (ex : youBot)
        """

        i = target[0]
        j = target[1]
        iMin = None
        jMin = None
        dist = np.inf

        for x in [i-4, i, i+4]:
            for y in [j-4, j, j+4]:
                if x >= 0 and y >= 0 and x < self.matHeight  and y < self.matWidth and not(x==i and y==j):

                    distCurr = np.linalg.norm([x-source[0], y-source[1]])

                    if distCurr < dist and self.gridExtend2[x,y] != 1:
                        iMin = x
                        jMin = y
                        dist = distCurr
        
        cx = target[0]
        cy = target[1]        
        x = iMin
        y = jMin
        dist = np.linalg.norm([cx-x, cy-y])

        while(dist < (2*self.radiusTables*self.resolution)+1):     
            x,y = self.extendPointOutTable(cx,cy,x,y)
            dist = np.linalg.norm([cx-x, cy-y])

        return [round(x),round(y)]


