# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class RandomAgent(Agent):
    def getAction(selfself, gameState):
        legalMoves = gameState.getLegalActions()
        chosenAction = random.choice(legalMoves)
        return chosenAction

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #Laborator 5
        # print("Evaluating current position ", currentGameState.getPacmanPosition(), ": action", action, " results in ", newPos);
        # print("Remaining food: \n", newFood.asList());
        # print(newFood);
        # print("New scared times: ", newScaredTimes);
        #
        # ghost_position = currentGameState.getGhostPosition()#distanta fiecarei fantome
        # distToGhost = [manhattanDistance(newPos, ghost_position) for ghost_pos in ghost_position]
        # distToFood = [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
        #
        # d1 = min(distToFood) if len(distToFood) > 0 else 0
        # d2 = min(distToGhost) if len(distToGhost) > 0 else 0
        #
        # return successorGameState.getScore() - d1 + d2

        # focusing on eating food.When ghost near don't go,
        newFood = successorGameState.getFood().asList()
        minFoodist = float("inf")
        for food in newFood:
            minFoodist = min(minFoodist, manhattanDistance(newPos, food))

        # avoid ghost if too close
        for ghost in successorGameState.getGhostPositions():
            if (manhattanDistance(newPos, ghost) < 2):
                return -float('inf')
        # reciprocal
        return successorGameState.getScore() + 1.0 / minFoodist


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        action, score = self.minimax(gameState, 0, 0)  # Get the action and score for pacman (agent_index=0)
        return action  # Return the action to be done as per minimax algorithm

    def minimax(self, gameState, depth, agentIndex = True):
        '''
        Returns the best score for an agent using the minimax algorithm. For max player (agentIndex=0), the best
        score is the maximum score among its successor states and for the min player (agentIndex!=0), the best
        score is the minimum score among its successor states. Recursion ends if there are no successor states
        available or depth equals the max depth to be searched until.
        '''

        if gameState.getNumAgents() <= agentIndex:
            agentIndex = 0
            depth = depth + 1

        if depth == self.depth or gameState.isWin():
            return None, self.evaluationFunction(gameState)

        bestScore, bestAction = None, None
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex == False:  # If it is max player's (pacman) turn
            for action in actions:  # For each legal action of pacman
                _, score = self.minimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
                if bestScore is None or bestScore < score:
                    bestScore = score
                    bestAction = action
        else:  # If it is min player's (ghost) turn
            for action in actions:  # For each legal action of ghost agent
                _, score = self.minimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
                if bestScore is None or bestScore > score:
                    bestScore = score
                    bestAction = action

        if bestScore is None:
            return None, self.evaluationFunction(gameState)
        return bestAction, bestScore  # Return the best_action and best_score
        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        inf = float('inf')
        action, score = self.alpha_beta(gameState, 0, -inf, inf, 0)  # Get the action and score for pacman (max)
        return action  # Return the action to be done as per alpha-beta algorithm

    def alpha_beta(self, gameState, depth, alpha, beta, agentIndex=True):
        '''
        Returns the best score for an agent using the minimax algorithm. For max player (agentIndex=0), the best
        score is the maximum score among its successor states and for the min player (agentIndex!=0), the best
        score is the minimum score among its successor states. Recursion ends if there are no successor states
        available or depth equals the max depth to be searched until.
        '''

        if gameState.getNumAgents() <= agentIndex:
            agentIndex = 0
            depth = depth + 1

        if depth == self.depth or gameState.isWin():
            return None, self.evaluationFunction(gameState)

        bestScore, bestAction = None, None
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex == False:  # If it is max player's (pacman) turn
            for action in actions:  # For each legal action of pacman
                _, score = self.alpha_beta(gameState.generateSuccessor(agentIndex, action), depth, alpha, beta,
                                           agentIndex + 1)
                if bestScore is None or bestScore < score:
                    bestScore = score
                    bestAction = action
                alpha = max(alpha, score)
                if alpha > beta:
                    break;
        else:  # If it is min player's (ghost) turn
            for action in actions:  # For each legal action of ghost agent
                _, score = self.alpha_beta(gameState.generateSuccessor(agentIndex, action), depth, alpha, beta,
                                           agentIndex + 1)
                if bestScore is None or bestScore > score:
                    bestScore = score
                    bestAction = action
                beta = min(beta, score)
                if beta < alpha:
                    break;

        if bestScore is None:
            return None, self.evaluationFunction(gameState)
        return bestAction, bestScore  # Return the best_action and best_score
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
