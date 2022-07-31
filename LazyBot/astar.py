"""
    University of Liege
    INFO0948-2 - Introduction to intelligent robotics
    Authors : 
        BOVEROUX Laurie
        DELCOUR Florian
"""

import numpy as np
import scipy.ndimage as sp
import heapq

class State:
    def __init__(self, spx, spy):
        self.spx = spx
        self.spy = spy

def generateSuccessors(state, map, extend):
    """Generate successors to a given state. Extend the obstacle map by 'extend'"""

    grid = np.zeros((map.shape[0], map.shape[1]))      
    ind = np.where(map[:,:]>0) #where there are obstacles
    grid[ind] = 1
    # do the inflation using SciPy
    if extend !=0:
        struct2 = sp.generate_binary_structure(2, 2)
        grid = sp.binary_dilation(grid, structure=struct2, iterations = extend).astype(grid.dtype)

    s = []
    i = state.spx
    j = state.spy
    for x in [i-1, i, i+1]:
        for y in [j-1, j, j+1]:
            if x >= 0 and y >= 0 and x < map.shape[0]  and y < map.shape[1] and not(x==i and y ==j):               
                if grid[x,y] == 0:
                    s.append(State(x, y))
                    
    return s


def key(state):
    """Get key of a state"""

    return (state.spx, state.spy)


def heuristic(state, gpx, gpy):
    """Heuristic function of astar algorithm: euclidean distance"""

    return np.linalg.norm([state.spx-gpx, state.spy-gpy])


def astar(spx, spy, gpx, gpy, map, extend=2):
    """A* algorithm"""

    state = State(spx, spy)
    path = []
    fringe = PriorityQueue()
    fringe.push((state, path, 0), 0)
    closed = set()

    while True:
        if fringe.isEmpty():
            return []  # failure

        priority, item = fringe.pop()
        current, path, cost = item

        if current.spx==gpx and current.spy==gpy:
            return path

        current_key = key(current)
        
        if current_key not in closed:
            closed.add(current_key)
            
            for next_state in generateSuccessors(current, map, extend):
                newCost = cost + np.linalg.norm([next_state.spx-current.spx, next_state.spy-current.spy])
                item = (next_state, path  + [[next_state.spx, next_state.spy]], newCost)
                fringe.push(item, heuristic(next_state, gpx,gpy)+newCost)


class PriorityQueue:
    """Priority Queue"""

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (priority, _, item) = heapq.heappop(self.heap)
        return (priority, item)

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)