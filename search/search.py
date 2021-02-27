# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    """startingNode = problem.getStartState()
    if problem.isGoalState(startingNode):
        return []

    frontier = util.Stack()
    visitedNodes = []
    frontier.push((startingNode, []))

    while not frontier.isEmpty():
        currentNode, actions = myQueue.pop()
        if currentNode not in visitedNodes:
            visitedNodes.append(currentNode)

            if problem.isGoalState(currentNode):
                return actions

            for nextNode, action, cost in problem.getSuccessors(currentNode):
                newAction = actions + [action]
                frontier.push((nextNode, newAction))"""

    frontier = problem.getStartState();

    stack = util.Stack();
    visitedNodes = []

    if problem.isGoalState(frontier):
        return []

    node = GraphNode(frontier, None, [], 0)
    stack.push(node)

    while not stack.isEmpty():
        node = stack.pop()

        if node.getState() not in visitedNodes:
            visitedNodes.append(node.getState())
            if problem.isGoalState(node.getState()):
                return node.getAction()

            for state, action, cost in problem.getSuccessors(node.getState()):
                newAction = node.getAction() + [action]
                newCost = cost + node.getCost()
                newNode = GraphNode(state, node.getState(), newAction, newCost)
                stack.push(newNode)


    #print("Start:", problem.getStartState())
    #print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    #print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    #Laborator 2
    #(state, action, cost) = problem.getSuccessors(problem.getStartState())
    """(next_state, action, _) = problem.getSuccessors(problem.getStartState())[0] #primul succesor
    (next_next, next_action, _) = problem.getSuccessors(next_state)[0]
    print ("A possible solution could start with actions ", action, next_action)
    return [action, next_action]

    "*** YOUR CODE HERE ***"

    #2.9
    node1 = CustomNode(" first ", 3)  # creates a new object
    node2 = CustomNode(" second ", 10)
    print(" Create a stack ")
    my_stack = util.Stack()  # creates a new object of the class Stack defined in file util.py
    print(" Push the new node into the stack ")
    my_stack.push(node1)
    my_stack.push(node2)
    print(" Pop an element from the stack ")
    extracted = my_stack.pop()  # call a method of the object
    print(" Extracted node is ", extracted.getName(), " ", extracted.getCost())"""


    #util.raiseNotDefined()



def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    startPoint = problem.getStartState()
    queue = util.Queue()
    visitedNodes = []
    node = GraphNode(startPoint, None, [], 0)
    queue.push(node)
    '''successor=node.getState()
    action=node.getAction()
    cost=node.getCost()'''
    while not queue.isEmpty():
        node = queue.pop()
        if problem.isGoalState(node.getState()):
            return node.getAction()
        if node.getState() not in visitedNodes:
            visitedNodes.append(node.getState())
            for state, action, cost in problem.getSuccessors(node.getState()):
                newAction = node.getAction() + [action]
                newCost = cost + node.getCost()
                newNode = GraphNode(state, node.getState(), newAction, newCost)
                queue.push(newNode)


    #util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    startPoint = problem.getStartState()
    ucs = util.PriorityQueue()
    dictionary = util.Counter()
    visitedNodes = []
    node = GraphNode(startPoint, None, [], 0)
    ucs.push(node, dictionary[str(startPoint[0])])
    while not ucs.isEmpty():
        node = ucs.pop()
        if problem.isGoalState(node.getState()):
            return node.getAction()
        if node.getState() not in visitedNodes:
            visitedNodes.append(node.getState())
            for state, action, cost in problem.getSuccessors(node.getState()):
                newAction = node.getAction() + [action]
                newCost = cost + node.getCost()
                newNode = GraphNode(state, node.getState(), newAction, newCost)
                dictionary[str(state)] = problem.getCostOfActions(node.getAction() + [action])
                ucs.push(newNode, dictionary[str(state)])
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    startPoint = problem.getStartState()
    astar = util.PriorityQueue()
    dictionary = util.Counter()
    visitedNodes = []
    node = GraphNode(startPoint, None, [], 0)
    dictionary[str(startPoint[0])] += heuristic(startPoint, problem)
    astar.push(node, dictionary[str(startPoint[0])])
    while not astar.isEmpty():
        node = astar.pop()
        if problem.isGoalState(node.getState()):
            return node.getAction()
        if node.getState() not in visitedNodes:
            visitedNodes.append(node.getState())
            for state, action, cost in problem.getSuccessors(node.getState()):
                newAction = node.getAction() + [action]
                newCost = cost + node.getCost()
                newNode = GraphNode(state, node.getState(), newAction, newCost)
                dictionary[str(state)] = problem.getCostOfActions(node.getAction() + [action]) + heuristic(state, problem)
                astar.push(newNode, dictionary[str(state)])
    #util.raiseNotDefined()

#2.9
"""class CustomNode:

    def __init__(self, name, cost):
        self.name =  name #attribute name
        self.cost = cost #attribute cost

    def getName(self):
        return self.name

    def getCost(self):
        return self.cost"""

#2.10
"""def randomSearch (problem) :
    current = problem .getStartState ()
    solution =[]
    while (not (problem .isGoalState(current))) :
        succ = problem . getSuccessors (current)
        no_of_successors = len (succ)
        random_succ_index = int (random.random () * no_of_successors)
        next = succ [ random_succ_index ]
        current = next [0]
        solution . append ( next [1])
    print (" The solution is " , solution)
    return solution"""

#for DFS
class GraphNode:
    def __init__(self, state, parent, action, cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost

    def getState(self):
        return self.state

    def getParent(self):
        return self.parent

    def getAction(self):
        return self.action

    def getCost(self):
        return self.cost

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
