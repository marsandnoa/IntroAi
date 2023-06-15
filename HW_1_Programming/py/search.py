# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util
from util import heappush, heappop,PriorityQueue
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
      """
      Returns the start state for the search problem
      """
      util.raiseNotDefined()

    def isGoalState(self, state):
      """
      state: Search state

      Returns True if and only if the state is a valid goal state
      """
      util.raiseNotDefined()

    def getSuccessors(self, state):
      """
      state: Search state

      For a given state, this should return a list of triples,
      (successor, action, stepCost), where 'successor' is a
      successor to the current state, 'action' is the action
      required to get there, and 'stepCost' is the incremental
      cost of expanding to that successor
      """
      util.raiseNotDefined()

    def getCostOfActions(self, actions):
      """
      actions: A list of actions to take

      This method returns the total cost of a particular sequence of actions.  The sequence must
      be composed of legal moves
      """
      util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
# a set used to keep track of expanded nodes
    closed=set()
# a dict to keep track of the path of nodes to end node
    expandednode={}
#set of possible expansion nodes
    fringe=[]
    fringe.insert(0, problem.getStartState())
#list for output of actions
    actions=[]

#continues till no nodes left to expand to
    while len(fringe)>0:
#since DFS, the leftmost node is always the first node(nodes are added LIFO) fringe
      node=fringe.pop(0)
#exit condition when goal state
      if (problem.isGoalState(node)):
#looping through the path to create list of actions
        while (node!=problem.getStartState()):
          temptuple=expandednode[node]
          actions.insert(0,temptuple[1])
          node=temptuple[0]
        return actions
      else:
#checking to see if node was expanded
          if (node in closed):
          	pass
          else:
#adding to expanded list
            closed.add(node)
#loop for successors
            for (next_state,action,cost) in problem.getSuccessors(node):
#add successors to list of possible expansions, adding to front of list for LIFO
              fringe.insert(0,next_state)
#dont add any successors that were already expanded, needed for manypaths repeat adding to fringe
              if (next_state in closed):
                pass 
              else:
#creating path of nodes and actions
                expandednode[next_state]=(node,action)

def breadthFirstSearch(problem):
    closed=set()
    expandednode={}
    fringe=[]
    fringe.insert(0, problem.getStartState())
    actions=[]
    while len(fringe)>0:
#pulling next node for expansion, since BFS,nodes are added FIFO to fringe
      node=fringe.pop(0)
      if (problem.isGoalState(node)):
        while (node!=problem.getStartState()):
          temptuple=expandednode[node]
          actions.insert(0,temptuple[1])
          node=temptuple[0]
        return actions
      else:
          if (node in closed):
          	pass
          else:
            closed.add(node)

            for (next_state,action,cost) in problem.getSuccessors(node):
              if (next_state in closed):
                pass 
              else:
                if (fringe.count(next_state)==0):
                  expandednode[next_state]=(node,action)
#appending is needed for FIFO
                fringe.append(next_state)

def uniformCostSearch(problem):
    closed=set()
    expandednode={}
#the fringe list was replaced with a priorityQ to pull the lowest cost edge
    fringe=PriorityQueue()
#format of item in priorityQ is (node/state,parent node,action from parent to node, cost to current node)
    fringe.push((problem.getStartState(),'parent','action',0),0)
    actions=[]

    while not(fringe.isEmpty()):
#pulling apart next priorityQ item
#pulling next state from q
      tempthree=fringe.pop()
#pulling cost from q
      currentcost=tempthree[3]
#checking if this is shortest path to the node
      if (expandednode.get(tempthree[0],0)==0):
        expandednode[tempthree[0]]=(tempthree[1],tempthree[2],currentcost)
      else:
        if (expandednode.get(tempthree[0])[2]>currentcost):
          expandednode[tempthree[0]]=(tempthree[1],tempthree[2],currentcost)
#pulling node
      node=tempthree[0]
      if (problem.isGoalState(node)):
        while (node!=problem.getStartState()):
          temptuple=expandednode[node]
          actions.insert(0,temptuple[1])
          node=temptuple[0]
        return actions
      else:
          if (node in closed):
          	pass
          else:
            closed.add(node)

            for (next_state,action,cost) in problem.getSuccessors(node):
              if (next_state in closed):
                pass 
              else:
#determining successor cost using the current nodes cost and edge cost
                truecost=currentcost+cost
#adding to fringe
                fringe.update((next_state,node,action,truecost),truecost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    closed=set()
    expandednode={}
    fringe=PriorityQueue()
    fringe.push((problem.getStartState(),'parent','action',0,problem),heuristic(problem.getStartState(),problem))
    actions=[]

    while not(fringe.isEmpty()):
      tempthree=fringe.pop()
      currentcost=tempthree[3]
      if (expandednode.get(tempthree[0],0)==0):
        expandednode[tempthree[0]]=(tempthree[1],tempthree[2],currentcost)
      else:
        if (expandednode.get(tempthree[0])[2]>currentcost):
          expandednode[tempthree[0]]=(tempthree[1],tempthree[2],currentcost)

      node=tempthree[0]
      if (problem.isGoalState(node)):
        while (node!=problem.getStartState()):
          temptuple=expandednode[node]
          actions.insert(0,temptuple[1])
          node=temptuple[0]
        return actions
      else:
          if (node in closed):
          	pass
          else:
            closed.add(node)

            for (next_state,action,cost) in problem.getSuccessors(node):
              if (next_state in closed):
                pass 
              else:
#calculating actual cost to node, and heuristic+actual cost
                truecost=currentcost+cost
                heuristiccost=truecost+heuristic(next_state,problem)
                fringe.update((next_state,node,action,truecost),heuristiccost)



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
