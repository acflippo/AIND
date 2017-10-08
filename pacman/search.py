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
DEBUG = True


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
  """
  Search the deepest nodes in the search tree first
  [2nd Edition: p 75, 3rd Edition: p 87]

  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm
  [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:

  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())
  """
  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())
  #util.raiseNotDefined()

def breadthFirstSearch(problem):
  """
  Search the shallowest nodes in the search tree first.
  [2nd Edition: p 73, 3rd Edition: p 82]
  """
  "*** YOUR CODE HERE ***"

  # initialize frontier as a stack and explored nodes as a set
  frontier = util.Stack()
  explored = set()
  start_node = problem.getStartState()

  # frontier is a tuple of (successor, action, stepCost)
  frontier.push((start_node, []))

  while not frontier.isEmpty():
      # Get a node from the frontier
      (curr_node, curr_path) = frontier.pop()

      if DEBUG:
          print "curr_node:", curr_node, " curr_path: ", curr_path

      # check if the current node is at the Goal
      if problem.isGoalState(curr_node):
          return curr_path
      else:
          # add the node to the explored set
          explored.add(curr_node)
          successors = problem.getSuccessors(curr_node)

          if DEBUG:
              print "successors", successors
          for (node, action, stepCost) in successors:
              # if node is not in visited then put it in the frontier stack
              if node not in explored:
                  frontier.push((node, curr_path + [action]))

  return []


def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  "*** YOUR CODE HERE ***"
  # initialize frontier as a stack and visited nodes as a set
  frontier = util.PriorityQueue()
  explored = []

  # start_node is a tuple of (state, action, cost)
  start_node = ((problem.getStartState(), None, 0), [], 0)
  frontier.push(start_node, None)

  if DEBUG:
      print "start_node:", start_node


  while not frontier.isEmpty():
      # Get a node from the frontier
      curr_state = frontier.pop()
      curr_node = curr_state[0][0]
      curr_dir = curr_state[0][1]
      curr_path = curr_state[1]
      curr_cost = curr_state[2]

      if DEBUG:
          print "curr_node:", curr_node, " curr_dir: ", curr_dir, " curr_path: ", curr_path, "curr_cost: ", curr_cost

      if curr_node not in explored:
          explored.append(curr_node)

          # check if the current node is at the Goal
          if problem.isGoalState(curr_node):
              return curr_path

          successors = problem.getSuccessors(curr_node)

          if DEBUG:
              print "successors", successors

          # Check each successor nodes to see if it reached the Goal
          for (node, action, stepCost) in successors:
              print "node: ", node, "action: ", action, "stepCost:", stepCost
              if node not in explored:
                  if problem.isGoalState(node):
                      return curr_path + [action]
              new_state = ((node, action, stepCost), curr_path+[action], curr_cost + stepCost)
              frontier.push(new_state, curr_cost + stepCost)

  return []


def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  "*** YOUR CODE HERE ***"
  frontier = util.PriorityQueue()
  explored = []

  # h - estimated distance from Goal, g - path cost
  h = heuristic(problem.getStartState(), problem)
  g = 0
  f = g + h

  # start_node is a tuple of (state, action, cost)
  start_node = (problem.getStartState(), None, g, [])
  frontier.push(start_node, f)

  if DEBUG:
      print "start_node:", start_node

  while not frontier.isEmpty():
      # Get a node from the frontier
      curr_state = frontier.pop()
      curr_node = curr_state[0]
      curr_dir  = curr_state[1]
      curr_cost = curr_state[2]
      curr_path = curr_state[3]

      if DEBUG:
          print "curr_node:", curr_node, " curr_path: ", curr_path, "curr_cost: ", curr_cost

      if curr_node not in explored:
          explored.append(curr_node)

          successors = problem.getSuccessors(curr_node)

          if DEBUG:
              print "successors", successors

          # Check each successor nodes to see if it reached the Goal
          for (node, action, stepCost) in successors:
              print "node: ", node, "action: ", action, "stepCost:", stepCost
              if node not in explored:
                  if problem.isGoalState(node):
                      return curr_path + [action]

              h = heuristic(node, problem)
              g = curr_cost + stepCost
              f = g + h

              new_state = (node, action, g, curr_path+[action])
              frontier.push(new_state, f)

  return []




# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
